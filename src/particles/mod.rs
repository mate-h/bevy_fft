//! Water FX particles on the GPU: spray, droplets and splashes.
//!
//! [`WaterParticlesPlugin`] keeps a fixed pool of particles in world space in a storage buffer.
//! Each frame a compute pass writes the frame's spawns at an atomic cursor that runs around the
//! pool, overwriting the oldest slots once it is full, and moves every live particle under gravity
//! and quadratic air drag against [`WaterParticleSettings::wind`]. The pool is drawn as
//! camera-facing quads with premultiplied alpha in the transparent pass, one quad per slot, read
//! by vertex index. Free slots collapse to nothing.
//!
//! With [`WaterParticleSettings::collide`] on, a particle that falls through the water surface in
//! the [`WaterLodData`] dies there and leaves active foam in the foam sim's
//! [`WaterFoamDeposit`](crate::water::WaterFoamDeposit), which it adds that same frame. The
//! simulation runs in [`WaterLodGraphSystems::Simulate`], before the foam sim.
//!
//! Spawns come from [`WaterParticleSpawns`], which anything can push to during the frame, and
//! from [`WaterParticleEmitter`] components, a CPU emitter for tests and examples. Later emitters
//! spawn from the water data on the GPU.

use bevy::{
    asset::{RenderAssetUsages, embedded_asset},
    camera::visibility::NoFrustumCulling,
    ecs::world::{FromWorld, World},
    mesh::{Indices, PrimitiveTopology},
    prelude::*,
    render::{
        Render, RenderApp, RenderSystems,
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        render_asset::RenderAssets,
        render_resource::{
            AsBindGroup,
            binding_types::{
                storage_buffer, storage_buffer_read_only, storage_buffer_sized, texture_2d_array,
                uniform_buffer,
            },
            encase::internal::{WriteInto, Writer},
            *,
        },
        renderer::{RenderContext, RenderDevice, RenderGraph, RenderQueue},
        storage::{GpuShaderBuffer, ShaderBuffer},
        texture::GpuImage,
    },
    shader::ShaderRef,
};

use crate::water::{
    WaterFoamDepositSystems, WaterFoamSim, WaterLodData, WaterLodGpu, WaterLodGraphSystems,
    WaterLodSlices, WaterPlugin,
};

pub const WATER_PARTICLES_SIMULATE_SHADER: &str = "embedded://bevy_fft/particles/simulate.wesl";
pub const WATER_PARTICLES_DRAW_SHADER: &str = "embedded://bevy_fft/particles/draw.wesl";

/// Bytes per particle in the pool: position and size, velocity and remaining life.
pub const WATER_PARTICLE_BYTES: u64 = 32;
/// Most spawns one frame takes. Spawns past it wait for the next frame.
pub const WATER_PARTICLE_MAX_SPAWNS: usize = 8192;
/// Most particles GPU passes can queue in [`WaterParticleQueue`] per frame.
pub const WATER_PARTICLE_QUEUE_CAPACITY: u64 = 4096;

/// A render-world queue that other GPU passes append particles to, such as floating bodies'
/// splashes. It is a storage buffer of an `atomic<u32>` count, three `u32` of padding, then
/// [`WATER_PARTICLE_QUEUE_CAPACITY`] particles laid out like [`WaterParticle`]. A pass appends with
/// `atomicAdd` on the count and writes the slot it got while it is below the capacity. The
/// particle pass spawns them on its next run and resets the count.
#[derive(Resource)]
pub struct WaterParticleQueue {
    pub buffer: Buffer,
}

impl FromWorld for WaterParticleQueue {
    fn from_world(world: &mut World) -> Self {
        let buffer = world
            .resource::<RenderDevice>()
            .create_buffer(&BufferDescriptor {
                label: Some("water_particle_queue"),
                size: 16 + WATER_PARTICLE_QUEUE_CAPACITY * WATER_PARTICLE_BYTES,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        Self { buffer }
    }
}

/// One particle as the pool and the spawn queue store it.
#[derive(Clone, Copy, Debug, Default, ShaderType, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct WaterParticle {
    pub position: Vec3,
    /// Quad edge in meters.
    pub size: f32,
    pub velocity: Vec3,
    /// Seconds left to live.
    pub life: f32,
}

/// Pool size and the forces on every particle.
#[derive(Resource, Clone, ExtractResource, Reflect)]
#[extract_app(RenderApp)]
pub struct WaterParticleSettings {
    /// Slots in the pool, read when the plugin starts.
    pub capacity: u32,
    /// Acceleration in m/s².
    pub gravity: Vec3,
    /// Quadratic air drag: a particle moving at `v` through air moving at [`Self::wind`] slows
    /// by `drag |v - wind| (v - wind)`. For a water drop of diameter d it is about
    /// `0.47 · 3/4 · ρ_air / ρ_water / d`, 0.4 per meter for a 1 mm drop.
    pub drag: f32,
    /// Wind velocity in m/s.
    pub wind: Vec3,
    /// Premultiplied colour and opacity of a particle's centre.
    pub color: LinearRgba,
    /// Seconds over which a particle fades before it dies.
    pub fade: f32,
    /// Runs the simulation. Off holds every particle where it is.
    pub enabled: bool,
    /// Tests particles against the water surface in the [`WaterLodData`].
    /// A particle that falls through it dies there and leaves foam, and one past the data's reach
    /// dies too.
    pub collide: bool,
    /// Height of the water surface mesh plane in meters, which the LOD heights are relative to.
    pub surface_y: f32,
    /// Active foam a landing particle leaves over its own footprint, from 0 to 1. A particle as
    /// wide as an LOD texel with 1 foams that texel fully.
    pub foam_per_drop: f32,
    /// Spray from breaking crests. See [`WaterCrestSpray`].
    pub crest_spray: WaterCrestSpray,
}

/// Spray that breaking crests in the water LOD data throw up. The ocean, shallow water grids and
/// dispersive patches all write their breaking intensity into the LOD foam lane, and each texel
/// of the finest [`Self::slices`] slices with breaking emits particles. A texel a finer slice
/// also covers is left to it, so nested slices emit a patch once.
#[derive(Clone, Copy, Debug, Reflect)]
pub struct WaterCrestSpray {
    /// Particles per m² per second at full breaking. Zero turns the emitter off. The rate rises
    /// to its full value where the surface rises as fast as [`Self::speed`] and falls to half
    /// where it stands still.
    pub rate: f32,
    pub slices: u32,
    /// Most particles one frame emits.
    pub cap: u32,
    /// Upward launch speed at full breaking in m/s, on top of the water's own velocity, and the
    /// random speed added in any direction.
    pub speed: f32,
    pub spread: f32,
    /// Mean quad edge in meters and mean lifetime in seconds. Each particle takes 0.5 to 1.5
    /// times them.
    pub size: f32,
    pub lifetime: f32,
}

impl Default for WaterCrestSpray {
    fn default() -> Self {
        Self {
            rate: 0.0,
            slices: 4,
            cap: 4096,
            speed: 2.5,
            spread: 1.0,
            size: 0.04,
            lifetime: 1.5,
        }
    }
}

impl Default for WaterParticleSettings {
    fn default() -> Self {
        Self {
            capacity: 1 << 16,
            gravity: Vec3::new(0.0, -9.81, 0.0),
            drag: 0.4,
            wind: Vec3::ZERO,
            color: LinearRgba::new(0.8, 0.85, 0.9, 0.9),
            fade: 0.3,
            enabled: true,
            collide: true,
            surface_y: 0.0,
            foam_per_drop: 1.0,
            crest_spray: WaterCrestSpray::default(),
        }
    }
}

/// Particles to spawn, in world space. Push to it at any time; the render world takes everything
/// in it at the frame's extraction, so each spawn reaches the GPU once.
#[derive(Resource, Clone, Default)]
pub struct WaterParticleSpawns(pub Vec<WaterParticle>);

/// The pool's buffers. `particles` holds [`WaterParticleSettings::capacity`] particles, and
/// `state` eight `u32` counters, the first four of which are the spawn cursor, particles spawned,
/// particles that died, by age or by being overwritten, and particles alive after the last frame. Spawned minus died is
/// alive. Both can be read back with [`Readback::buffer`](bevy::render::gpu_readback::Readback).
#[derive(Resource, Clone, ExtractResource)]
#[extract_app(RenderApp)]
pub struct WaterParticles {
    pub particles: Handle<ShaderBuffer>,
    pub state: Handle<ShaderBuffer>,
    pub capacity: u32,
}

/// The frame's simulated time step, the same the foam simulation takes.
#[derive(Resource, Clone, Default, ExtractResource)]
#[extract_app(RenderApp)]
struct WaterParticleFrame {
    dt: f32,
}

/// Sprays particles from its entity's position. A CPU emitter for tests and examples.
#[derive(Component, Clone, Debug, Reflect)]
pub struct WaterParticleEmitter {
    /// Mean launch velocity in m/s.
    pub velocity: Vec3,
    /// Random speed added in any direction, in m/s.
    pub spread: f32,
    /// Particles per second.
    pub rate: f32,
    pub lifetime: f32,
    /// Quad edge in meters.
    pub size: f32,
    /// Fraction of a particle carried to the next frame.
    #[reflect(ignore)]
    pub carry: f32,
    #[reflect(ignore)]
    pub seed: u32,
}

impl Default for WaterParticleEmitter {
    fn default() -> Self {
        Self {
            velocity: Vec3::new(0.0, 6.0, 0.0),
            spread: 1.0,
            rate: 2000.0,
            lifetime: 2.0,
            size: 0.05,
            carry: 0.0,
            seed: 1,
        }
    }
}

/// Draws the pool. See `draw.wesl`.
#[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
pub struct WaterParticleMaterial {
    #[storage(0, read_only)]
    pub particles: Handle<ShaderBuffer>,
    #[uniform(1)]
    pub look: WaterParticleLook,
}

#[derive(Clone, Copy, Debug, Default, ShaderType)]
pub struct WaterParticleLook {
    pub color: Vec4,
    pub fade: f32,
    pub _pad: Vec3,
}

impl Material for WaterParticleMaterial {
    fn vertex_shader() -> ShaderRef {
        WATER_PARTICLES_DRAW_SHADER.into()
    }

    fn fragment_shader() -> ShaderRef {
        WATER_PARTICLES_DRAW_SHADER.into()
    }

    fn alpha_mode(&self) -> AlphaMode {
        AlphaMode::Premultiplied
    }

    fn enable_prepass() -> bool {
        false
    }

    fn enable_shadows() -> bool {
        false
    }
}

/// Marks the entity that draws the pool.
#[derive(Component)]
pub struct WaterParticlesDraw;

pub struct WaterParticlesPlugin;

impl Plugin for WaterParticlesPlugin {
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<WaterPlugin>() {
            app.add_plugins(WaterPlugin);
        }
        embedded_asset!(app, "simulate.wesl");
        embedded_asset!(app, "draw.wesl");
        app.init_resource::<WaterParticleSettings>()
            .init_resource::<WaterParticleSpawns>()
            .init_resource::<WaterParticleFrame>()
            .register_type::<WaterParticleSettings>()
            .register_type::<WaterParticleEmitter>()
            .add_plugins((
                MaterialPlugin::<WaterParticleMaterial>::default(),
                ExtractResourcePlugin::<WaterParticleSettings>::default(),
                ExtractResourcePlugin::<WaterParticles>::default(),
                ExtractResourcePlugin::<WaterParticleFrame>::default(),
            ))
            .add_systems(Startup, setup_water_particles)
            .add_systems(
                PostUpdate,
                (emit_water_particles, record_water_particle_frame),
            )
            .add_systems(Update, sync_water_particle_look);
    }

    fn finish(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .init_resource::<WaterParticlePipeline>()
            .init_resource::<WaterParticleGpu>()
            .init_resource::<WaterParticleSpawns>()
            .init_resource::<WaterParticleQueue>()
            .add_systems(bevy::render::ExtractSchedule, extract_water_particle_spawns)
            .add_systems(
                Render,
                prepare_water_particles.in_set(RenderSystems::PrepareBindGroups),
            )
            .add_systems(
                RenderGraph,
                run_water_particles
                    .in_set(WaterLodGraphSystems::Simulate)
                    .in_set(WaterFoamDepositSystems),
            );
    }
}

fn setup_water_particles(
    mut commands: Commands,
    settings: Res<WaterParticleSettings>,
    mut buffers: ResMut<Assets<ShaderBuffer>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<WaterParticleMaterial>>,
) {
    let capacity = settings.capacity.max(1);
    // Zeroed, so every slot starts free.
    let particles = buffers.add(ShaderBuffer::new(
        vec![0u8; (capacity as u64 * WATER_PARTICLE_BYTES) as usize],
        RenderAssetUsages::RENDER_WORLD,
    ));
    let state = buffers.add(ShaderBuffer::new(
        vec![0u32; 8],
        RenderAssetUsages::RENDER_WORLD,
    ));
    // Six vertices per slot. Their positions are unused: the vertex shader reads the pool.
    let vertices = capacity as usize * 6;
    let mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::RENDER_WORLD,
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, vec![[0.0f32; 3]; vertices])
    .with_inserted_indices(Indices::U32((0..vertices as u32).collect()));
    commands.spawn((
        WaterParticlesDraw,
        Mesh3d(meshes.add(mesh)),
        MeshMaterial3d(materials.add(WaterParticleMaterial {
            particles: particles.clone(),
            look: look(&settings),
        })),
        Transform::default(),
        NoFrustumCulling,
        bevy::light::NotShadowCaster,
    ));
    commands.insert_resource(WaterParticles {
        particles,
        state,
        capacity,
    });
}

fn look(settings: &WaterParticleSettings) -> WaterParticleLook {
    WaterParticleLook {
        color: settings.color.to_vec4(),
        fade: settings.fade,
        _pad: Vec3::ZERO,
    }
}

fn sync_water_particle_look(
    settings: Res<WaterParticleSettings>,
    draws: Query<&MeshMaterial3d<WaterParticleMaterial>, With<WaterParticlesDraw>>,
    mut materials: ResMut<Assets<WaterParticleMaterial>>,
) {
    if !settings.is_changed() {
        return;
    }
    for handle in &draws {
        if let Some(mut material) = materials.get_mut(&handle.0) {
            material.look = look(&settings);
        }
    }
}

/// Moves the main world's spawns into the render world.
fn extract_water_particle_spawns(
    mut main: ResMut<bevy::render::MainWorld>,
    mut spawns: ResMut<WaterParticleSpawns>,
) {
    let Some(mut source) = main.get_resource_mut::<WaterParticleSpawns>() else {
        spawns.0.clear();
        return;
    };
    spawns.0 = std::mem::take(&mut source.0);
}

fn record_water_particle_frame(
    time: Res<Time>,
    settings: Res<WaterParticleSettings>,
    mut frame: ResMut<WaterParticleFrame>,
) {
    frame.dt = if settings.enabled {
        time.delta_secs()
    } else {
        0.0
    };
}

/// A small integer hash for the emitter's jitter.
fn hash(mut x: u32) -> u32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb_352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846c_a68b);
    x ^ (x >> 16)
}

fn unit(seed: &mut u32) -> f32 {
    *seed = hash(seed.wrapping_add(0x9e37_79b9));
    (*seed >> 8) as f32 / (1u32 << 24) as f32
}

fn emit_water_particles(
    time: Res<Time>,
    mut emitters: Query<(&mut WaterParticleEmitter, &GlobalTransform)>,
    mut spawns: ResMut<WaterParticleSpawns>,
) {
    for (mut emitter, transform) in &mut emitters {
        let wanted = emitter.rate * time.delta_secs() + emitter.carry;
        let count = wanted.floor();
        emitter.carry = wanted - count;
        let origin = transform.translation();
        for _ in 0..count as u32 {
            let mut seed = emitter.seed;
            // A uniform direction in the unit ball, scaled by the spread.
            let jitter = loop {
                let v = Vec3::new(unit(&mut seed), unit(&mut seed), unit(&mut seed)) * 2.0 - 1.0;
                if v.length_squared() <= 1.0 {
                    break v;
                }
            };
            emitter.seed = seed;
            spawns.0.push(WaterParticle {
                position: origin,
                size: emitter.size,
                velocity: emitter.velocity + jitter * emitter.spread,
                life: emitter.lifetime,
            });
        }
    }
}

#[derive(Clone, Copy, Default, ShaderType)]
struct GpuParticleParams {
    gravity: Vec3,
    dt: f32,
    wind: Vec3,
    drag: f32,
    capacity: u32,
    spawn_count: u32,
    collide: u32,
    surface_y: f32,
    foam_per_drop: f32,
    spray_rate: f32,
    spray_slices: u32,
    spray_cap: u32,
    spray_speed: f32,
    spray_spread: f32,
    spray_size: f32,
    spray_lifetime: f32,
    seed: u32,
    _pad: UVec3,
}

#[derive(Resource)]
struct WaterParticlePipeline {
    layout: BindGroupLayoutDescriptor,
    begin: CachedComputePipelineId,
    spawn: CachedComputePipelineId,
    simulate: CachedComputePipelineId,
    spray: CachedComputePipelineId,
    spawn_queued: CachedComputePipelineId,
    clear_queue: CachedComputePipelineId,
}

impl FromWorld for WaterParticlePipeline {
    fn from_world(world: &mut World) -> Self {
        let cache = world.resource::<PipelineCache>();
        let shader = world
            .resource::<AssetServer>()
            .load(WATER_PARTICLES_SIMULATE_SHADER);
        let layout = BindGroupLayoutDescriptor::new(
            "water_particles_layout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (
                    storage_buffer::<WaterParticle>(false),
                    storage_buffer::<[u32; 8]>(false),
                    storage_buffer_read_only::<WaterParticle>(false),
                    uniform_buffer::<GpuParticleParams>(false),
                    uniform_buffer::<WaterLodSlices>(false),
                    texture_2d_array(TextureSampleType::Float { filterable: false }),
                    texture_2d_array(TextureSampleType::Float { filterable: false }),
                    storage_buffer::<u32>(false),
                    texture_2d_array(TextureSampleType::Float { filterable: false }),
                    storage_buffer_sized(false, None),
                ),
            ),
        );
        let queue = |label: &'static str, entry: &'static str| {
            cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some(label.into()),
                layout: vec![layout.clone()],
                immediate_size: 0,
                shader: shader.clone(),
                shader_defs: vec![],
                entry_point: Some(entry.into()),
                zero_initialize_workgroup_memory: false,
                constants: vec![],
            })
        };
        Self {
            begin: queue("water_particles_begin", "beginFrame"),
            spawn: queue("water_particles_spawn", "spawnParticles"),
            simulate: queue("water_particles_simulate", "simulateParticles"),
            spray: queue("water_particles_crest_spray", "emitCrestSpray"),
            spawn_queued: queue("water_particles_spawn_queued", "spawnQueued"),
            clear_queue: queue("water_particles_clear_queue", "clearQueue"),
            layout,
        }
    }
}

/// The render world's spawn queue, uniform and bind group.
#[derive(Resource, Default)]
struct WaterParticleGpu {
    spawns: Option<Buffer>,
    params: Option<Buffer>,
    group: Option<BindGroup>,
    capacity: u32,
    spawn_count: u32,
    frame: u32,
    /// Workgroups of the crest emitter, when it runs this frame.
    spray: Option<UVec3>,
}

#[allow(clippy::too_many_arguments)]
fn prepare_water_particles(
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    cache: Res<PipelineCache>,
    pipeline: Res<WaterParticlePipeline>,
    buffers: Res<RenderAssets<GpuShaderBuffer>>,
    pool: Option<Res<WaterParticles>>,
    settings: Res<WaterParticleSettings>,
    spawns: Res<WaterParticleSpawns>,
    frame: Res<WaterParticleFrame>,
    lod: Option<Res<WaterLodData>>,
    lod_gpu: Res<WaterLodGpu>,
    foam: Res<WaterFoamSim>,
    particle_queue: Res<WaterParticleQueue>,
    images: Res<RenderAssets<GpuImage>>,
    mut gpu: ResMut<WaterParticleGpu>,
) {
    gpu.group = None;
    let (Some(pool), Some(lod)) = (pool, lod) else {
        return;
    };
    let (Some(particles), Some(state)) = (buffers.get(&pool.particles), buffers.get(&pool.state))
    else {
        return;
    };
    let (Some(displacement), Some(foam_lane), Some(flow), Some(slices), Some(deposit)) = (
        images.get(&lod.displacement),
        images.get(&lod.foam),
        images.get(&lod.flow),
        lod_gpu.slices.binding(),
        foam.deposit.buffer.as_ref(),
    ) else {
        return;
    };
    let spawn_buffer = gpu
        .spawns
        .get_or_insert_with(|| {
            device.create_buffer(&BufferDescriptor {
                label: Some("water_particle_spawns"),
                size: WATER_PARTICLE_MAX_SPAWNS as u64 * WATER_PARTICLE_BYTES,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        })
        .clone();
    let params_buffer = gpu
        .params
        .get_or_insert_with(|| {
            device.create_buffer(&BufferDescriptor {
                label: Some("water_particle_params"),
                size: GpuParticleParams::min_size().get(),
                usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        })
        .clone();
    let count = spawns.0.len().min(WATER_PARTICLE_MAX_SPAWNS);
    if count > 0 {
        queue.write_buffer(&spawn_buffer, 0, bytemuck::cast_slice(&spawns.0[..count]));
    }
    let params = GpuParticleParams {
        gravity: settings.gravity,
        dt: frame.dt,
        wind: settings.wind,
        drag: settings.drag.max(0.0),
        capacity: pool.capacity,
        spawn_count: count as u32,
        collide: u32::from(settings.collide),
        surface_y: settings.surface_y,
        foam_per_drop: settings.foam_per_drop.max(0.0),
        spray_rate: settings.crest_spray.rate.max(0.0),
        spray_slices: settings.crest_spray.slices,
        spray_cap: settings.crest_spray.cap,
        spray_speed: settings.crest_spray.speed,
        spray_spread: settings.crest_spray.spread,
        spray_size: settings.crest_spray.size,
        spray_lifetime: settings.crest_spray.lifetime,
        seed: gpu.frame.wrapping_mul(0x9e37_79b9),
        _pad: UVec3::ZERO,
    };
    gpu.frame = gpu.frame.wrapping_add(1);
    gpu.spray = (settings.crest_spray.rate > 0.0 && frame.dt > 0.0).then(|| {
        let slices = settings.crest_spray.slices.min(lod.slices.count);
        let res = lod.slices.resolution;
        UVec3::new(res.div_ceil(8), res.div_ceil(8), slices)
    });
    let mut bytes = Vec::new();
    params.write_into(&mut Writer::new(&params, &mut bytes, 0).unwrap());
    queue.write_buffer(&params_buffer, 0, &bytes);
    gpu.capacity = pool.capacity;
    gpu.spawn_count = count as u32;
    gpu.group = Some(device.create_bind_group(
        "water_particles",
        &cache.get_bind_group_layout(&pipeline.layout),
        &BindGroupEntries::sequential((
            particles.buffer.as_entire_buffer_binding(),
            state.buffer.as_entire_buffer_binding(),
            spawn_buffer.as_entire_buffer_binding(),
            params_buffer.as_entire_buffer_binding(),
            slices,
            &displacement.texture_view,
            &foam_lane.texture_view,
            deposit.as_entire_buffer_binding(),
            &flow.texture_view,
            particle_queue.buffer.as_entire_buffer_binding(),
        )),
    ));
}

fn run_water_particles(
    mut ctx: RenderContext,
    pipeline: Res<WaterParticlePipeline>,
    cache: Res<PipelineCache>,
    gpu: Res<WaterParticleGpu>,
) {
    let Some(group) = &gpu.group else {
        return;
    };
    let get = |id| cache.get_compute_pipeline(id);
    let (Some(begin), Some(spawn), Some(simulate), Some(spray), Some(queued), Some(clear)) = (
        get(pipeline.begin),
        get(pipeline.spawn),
        get(pipeline.simulate),
        get(pipeline.spray),
        get(pipeline.spawn_queued),
        get(pipeline.clear_queue),
    ) else {
        return;
    };
    let encoder = ctx.command_encoder();
    let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
        label: Some("water_particles_pass"),
        timestamp_writes: None,
    });
    pass.set_bind_group(0, group, &[]);
    pass.set_pipeline(begin);
    pass.dispatch_workgroups(1, 1, 1);
    if let Some(groups) = gpu.spray {
        pass.set_pipeline(spray);
        pass.dispatch_workgroups(groups.x, groups.y, groups.z);
    }
    if gpu.spawn_count > 0 {
        pass.set_pipeline(spawn);
        pass.dispatch_workgroups(gpu.spawn_count.div_ceil(64), 1, 1);
    }
    pass.set_pipeline(queued);
    pass.dispatch_workgroups((WATER_PARTICLE_QUEUE_CAPACITY as u32).div_ceil(64), 1, 1);
    pass.set_pipeline(clear);
    pass.dispatch_workgroups(1, 1, 1);
    pass.set_pipeline(simulate);
    pass.dispatch_workgroups(gpu.capacity.div_ceil(64), 1, 1);
}
