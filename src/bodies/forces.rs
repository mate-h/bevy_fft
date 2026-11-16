//! Water forces on voxelized floating bodies, evaluated on the GPU.
//!
//! Add [`FloatingBodyPlugin`] after [`WaterPlugin`](crate::water::WaterPlugin) and spawn
//! [`FloatingBody`] on entities with a [`GlobalTransform`]. Each frame one compute pass samples the
//! shared water LOD data at every voxel of every body, so currents, wakes and surf push bodies
//! along with the waves, and sums the forces per body. Bevy's [`Readback`] returns them a few
//! frames later into [`FloatingBody::force`] and [`FloatingBody::torque`], for a physics engine to
//! apply. Only 32 bytes per body come back.
//!
//! `forces.wesl` documents the force model. [`FloatingBodySettings`] holds its coefficients.

use std::{collections::VecDeque, sync::Arc};

use bevy::{
    asset::RenderAssetUsages,
    prelude::*,
    render::{
        Render, RenderApp, RenderSystems,
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        gpu_readback::{Readback, ReadbackComplete},
        render_asset::RenderAssets,
        render_resource::{binding_types::*, *},
        renderer::{RenderContext, RenderDevice, RenderGraph, RenderQueue},
        storage::{GpuShaderBuffer, ShaderBuffer},
        texture::GpuImage,
    },
};

use super::VoxelizedMesh;
use crate::water::{WaterLodData, WaterLodGpu, WaterLodGraphSystems, WaterLodSlices};

pub const FLOATING_BODY_SHADER: &str = "embedded://bevy_fft/bodies/forces.wesl";

/// Bytes of one voxel and of one `Patch` in `forces.wesl`.
const VOXEL_BYTES: usize = 16;
const PATCH_BYTES: usize = 32;
/// Bytes of one `Body` in `forces.wesl`.
const BODY_BYTES: usize = 96;
/// Bytes of the output header and of one `BodyForce` in `forces.wesl`.
const OUTPUT_HEADER_BYTES: usize = 16;
const OUTPUT_BODY_BYTES: usize = 32;
/// Most bodies one frame evaluates. Further bodies get no forces.
pub const FLOATING_BODY_MAX: usize = 256;
/// Submitted frames kept for pairing readbacks with the bodies they were computed for.
const PENDING_FRAMES: usize = 64;

/// Coefficients of the force model in `forces.wesl`.
#[derive(Resource, Clone, Copy, Debug, Reflect)]
#[reflect(Resource)]
pub struct FloatingBodySettings {
    /// m/s².
    pub gravity: f32,
    /// Water density in kg/m³: about 1025 for sea water and 1000 for fresh.
    pub density: f32,
    /// World Y of the water LOD data's zero height, the surface mesh plane.
    pub surface_y: f32,
    /// m²/s, about 1.19e-6 for sea water at 15 °C.
    pub kinematic_viscosity: f32,
    /// Pressure drag on surface moving into the water, in N/m² per m/s and per (m/s)².
    pub pressure_linear: f32,
    pub pressure_quadratic: f32,
    /// Exponent of the cosine of the angle of attack. Below 1 drag rises quickly off the
    /// tangent.
    pub pressure_falloff: f32,
    /// Suction on surface moving away from the water, as for pressure drag.
    pub suction_linear: f32,
    pub suction_quadratic: f32,
    pub suction_falloff: f32,
    /// Growth rate of wetted area times normal speed, per unit hull area, in m/s², at which
    /// slamming reaches full strength. Zero turns slamming off.
    pub slam_rate_max: f32,
    /// Exponent applied to the slamming strength below that rate.
    pub slam_power: f32,
}

impl Default for FloatingBodySettings {
    fn default() -> Self {
        Self {
            gravity: 9.81,
            density: 1025.0,
            surface_y: 0.0,
            kinematic_viscosity: 1.19e-6,
            pressure_linear: 100.0,
            pressure_quadratic: 500.0,
            pressure_falloff: 0.5,
            suction_linear: 50.0,
            suction_quadratic: 250.0,
            suction_falloff: 0.5,
            slam_rate_max: 50.0,
            slam_power: 2.0,
        }
    }
}

/// A body that floats on the water LOD data.
///
/// The physics engine writes [`Self::linear_velocity`] and [`Self::angular_velocity`] each frame,
/// or leaves [`Self::estimate_velocity`] on to have them estimated from the [`GlobalTransform`].
/// The forces in [`Self::force`] and [`Self::torque`] are in world space, the torque about the
/// world-space centre of mass. They arrive a few frames late. [`Self::frame`] counts readbacks.
#[derive(Component, Clone, Debug, Reflect)]
#[require(Transform)]
pub struct FloatingBody {
    #[reflect(ignore)]
    pub hull: Arc<VoxelizedMesh>,
    /// Kilograms. Slamming uses it to scale the momentum it stops.
    pub mass: f32,
    /// Local centre of mass. [`Self::new`] uses the hull's centroid.
    pub center_of_mass: Vec3,
    /// Length along the flow in meters, for the Reynolds number of the friction line.
    pub length: f32,
    /// World velocity of the centre of mass in m/s.
    pub linear_velocity: Vec3,
    /// World angular velocity in rad/s.
    pub angular_velocity: Vec3,
    /// Estimate the velocities from how the [`GlobalTransform`] moves between frames.
    pub estimate_velocity: bool,
    /// Latest water force in newtons.
    pub force: Vec3,
    /// Latest torque about the centre of mass in newton meters.
    pub torque: Vec3,
    /// Hull volume below the surface in cubic meters.
    pub submerged_volume: f32,
    /// Hull surface area below the surface in square meters.
    pub wetted_area: f32,
    /// Readbacks so far. It changes when the forces do.
    pub frame: u32,
    #[reflect(ignore)]
    last: Option<(Vec3, Quat, f32)>,
}

impl FloatingBody {
    /// A body with this hull and mass, centred on the hull's centroid, with its length taken
    /// from the hull's longest horizontal extent.
    pub fn new(hull: impl Into<Arc<VoxelizedMesh>>, mass: f32) -> Self {
        let hull = hull.into();
        let (lo, hi) = hull
            .voxels
            .iter()
            .fold((Vec3::MAX, Vec3::MIN), |(lo, hi), v| {
                (lo.min(v.center), hi.max(v.center))
            });
        let extent = (hi - lo).max(Vec3::ZERO) + hull.voxel_size;
        Self {
            center_of_mass: hull.centroid(),
            length: extent.x.max(extent.z),
            hull,
            mass,
            linear_velocity: Vec3::ZERO,
            angular_velocity: Vec3::ZERO,
            estimate_velocity: true,
            force: Vec3::ZERO,
            torque: Vec3::ZERO,
            submerged_volume: 0.0,
            wetted_area: 0.0,
            frame: 0,
            last: None,
        }
    }
}

/// Registers [`FloatingBody`] and its force pass. Add it after
/// [`WaterPlugin`](crate::water::WaterPlugin).
pub struct FloatingBodyPlugin;

impl Plugin for FloatingBodyPlugin {
    fn build(&self, app: &mut App) {
        bevy::asset::embedded_asset!(app, "forces.wesl");
        app.register_type::<FloatingBody>()
            .register_type::<FloatingBodySettings>()
            .init_resource::<FloatingBodySettings>()
            .init_resource::<FloatingBodyFrame>()
            .add_plugins(ExtractResourcePlugin::<FloatingBodyFrame>::default())
            .add_systems(Startup, setup_floating_bodies)
            .add_systems(
                PostUpdate,
                submit_floating_bodies.after(bevy::transform::TransformSystems::Propagate),
            );
        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .init_resource::<FloatingBodyBindGroup>()
                .add_systems(
                    Render,
                    prepare_floating_bodies.in_set(RenderSystems::PrepareBindGroups),
                )
                .add_systems(
                    RenderGraph,
                    run_floating_bodies.in_set(WaterLodGraphSystems::Read),
                );
        }
    }

    fn finish(&self, app: &mut App) {
        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app.init_resource::<FloatingBodyPipeline>();
        }
    }
}

/// Buffers and pairing state of the force pass.
#[derive(Resource)]
struct FloatingBodyBuffers {
    voxels: Handle<ShaderBuffer>,
    patches: Handle<ShaderBuffer>,
    bodies: Handle<ShaderBuffer>,
    state: Handle<ShaderBuffer>,
    output: Handle<ShaderBuffer>,
    /// Entities and hulls the voxel buffers were built for.
    layout: Vec<(Entity, usize)>,
    frame: u32,
    pending: VecDeque<(u32, Vec<Entity>)>,
}

/// Marks the entity that reads back the body forces.
#[derive(Component)]
struct FloatingBodyReadback;

fn setup_floating_bodies(mut commands: Commands, mut buffers: ResMut<Assets<ShaderBuffer>>) {
    let output = buffers.add(ShaderBuffer::with_size(
        (OUTPUT_HEADER_BYTES + FLOATING_BODY_MAX * OUTPUT_BODY_BYTES) as u64,
        RenderAssetUsages::RENDER_WORLD,
    ));
    let empty = |buffers: &mut Assets<ShaderBuffer>, bytes: usize| {
        buffers.add(ShaderBuffer::new(
            vec![0u8; bytes],
            RenderAssetUsages::default(),
        ))
    };
    let resource = FloatingBodyBuffers {
        voxels: empty(&mut buffers, VOXEL_BYTES),
        patches: empty(&mut buffers, PATCH_BYTES),
        bodies: empty(&mut buffers, FLOATING_BODY_MAX * BODY_BYTES),
        state: empty(&mut buffers, 4),
        output,
        layout: Vec::new(),
        frame: 0,
        pending: VecDeque::new(),
    };
    commands.insert_resource(resource);
    commands
        .spawn((FloatingBodyReadback, Name::new("Floating body readback")))
        .observe(accept_floating_body_forces);
}

fn accept_floating_body_forces(
    event: On<ReadbackComplete>,
    mut buffers: ResMut<FloatingBodyBuffers>,
    mut bodies: Query<&mut FloatingBody>,
) {
    let data = &event.data;
    if data.len() < OUTPUT_HEADER_BYTES {
        return;
    }
    let frame = read_u32(data, 0);
    let count = read_u32(data, 4) as usize;
    let Some(pos) = buffers.pending.iter().position(|(f, _)| *f == frame) else {
        return;
    };
    let (_, entities) = buffers.pending.remove(pos).unwrap();
    buffers.pending.retain(|(f, _)| *f > frame);
    for (i, entity) in entities.iter().enumerate().take(count) {
        let Ok(mut body) = bodies.get_mut(*entity) else {
            continue;
        };
        let o = OUTPUT_HEADER_BYTES + i * OUTPUT_BODY_BYTES;
        let Some(bytes) = data.get(o..o + OUTPUT_BODY_BYTES) else {
            break;
        };
        let f = |lane: usize| read_f32(bytes, lane * 4);
        body.force = Vec3::new(f(0), f(1), f(2));
        body.submerged_volume = f(3);
        body.torque = Vec3::new(f(4), f(5), f(6));
        body.wetted_area = f(7);
        body.frame = body.frame.wrapping_add(1);
    }
}

fn read_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
}

fn read_f32(bytes: &[u8], offset: usize) -> f32 {
    f32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
}

/// Render-world view of one frame of the force pass.
#[derive(Resource, Clone, Default, ExtractResource)]
#[extract_app(RenderApp)]
pub struct FloatingBodyFrame {
    body_count: u32,
    params: ForceParams,
    voxels: Handle<ShaderBuffer>,
    patches: Handle<ShaderBuffer>,
    bodies: Handle<ShaderBuffer>,
    state: Handle<ShaderBuffer>,
    output: Handle<ShaderBuffer>,
    lod_displacement: Handle<Image>,
    lod_foam: Handle<Image>,
    lod_flow: Handle<Image>,
}

/// Uniform `ForceParams` in `forces.wesl`.
#[derive(Clone, Copy, Default, ShaderType)]
struct ForceParams {
    gravity: f32,
    density: f32,
    surface_y: f32,
    dt: f32,
    kinematic_viscosity: f32,
    pressure_linear: f32,
    pressure_quadratic: f32,
    pressure_falloff: f32,
    suction_linear: f32,
    suction_quadratic: f32,
    suction_falloff: f32,
    slam_rate_max: f32,
    slam_power: f32,
    body_count: u32,
    frame: u32,
    _pad: u32,
}

fn push_vec4(bytes: &mut Vec<u8>, v: [f32; 4]) {
    for x in v {
        bytes.extend_from_slice(&x.to_le_bytes());
    }
}

#[allow(clippy::too_many_arguments)]
fn submit_floating_bodies(
    mut commands: Commands,
    mut state: ResMut<FloatingBodyBuffers>,
    mut frame_out: ResMut<FloatingBodyFrame>,
    mut shader_buffers: ResMut<Assets<ShaderBuffer>>,
    mut bodies: Query<(Entity, &mut FloatingBody, &GlobalTransform)>,
    readback: Query<(Entity, Has<Readback>), With<FloatingBodyReadback>>,
    settings: Res<FloatingBodySettings>,
    lod: Option<Res<WaterLodData>>,
    time: Res<Time>,
) {
    frame_out.body_count = 0;
    let Ok((readback_entity, has_readback)) = readback.single() else {
        return;
    };
    let now = time.elapsed_secs();
    let dt = time.delta_secs();
    let mut list = Vec::new();
    for (entity, mut body, transform) in &mut bodies {
        let (_, rotation, translation) = transform.to_scale_rotation_translation();
        let com = translation + rotation * body.center_of_mass;
        if body.estimate_velocity
            && let Some((last_com, last_rotation, last_time)) = body.last
            && now > last_time
        {
            let step = now - last_time;
            body.linear_velocity = (com - last_com) / step;
            let (axis, angle) = (rotation * last_rotation.inverse()).to_axis_angle();
            let angle = if angle > std::f32::consts::PI {
                angle - std::f32::consts::TAU
            } else {
                angle
            };
            body.angular_velocity = axis * angle / step;
        }
        body.last = Some((com, rotation, now));
        if list.len() < FLOATING_BODY_MAX && !body.hull.voxels.is_empty() {
            list.push((
                entity,
                body.into_inner().clone(),
                rotation,
                translation,
                com,
            ));
        }
    }
    let (Some(lod), false) = (lod, list.is_empty()) else {
        if has_readback {
            commands.entity(readback_entity).remove::<Readback>();
        }
        return;
    };

    // Rebuild the voxel and patch buffers when the bodies or their hulls change.
    let layout: Vec<(Entity, usize)> = list
        .iter()
        .map(|(e, b, ..)| (*e, Arc::as_ptr(&b.hull) as usize))
        .collect();
    let mut first = Vec::with_capacity(list.len());
    let (mut voxel_total, mut patch_total) = (0u32, 0u32);
    for (_, body, ..) in &list {
        first.push((voxel_total, patch_total));
        voxel_total += body.hull.voxels.len() as u32;
        patch_total += body.hull.surface.len() as u32;
    }
    if layout != state.layout {
        let mut voxels = Vec::with_capacity(voxel_total as usize * VOXEL_BYTES);
        let mut patches = Vec::with_capacity(patch_total as usize * PATCH_BYTES);
        for (_, body, ..) in &list {
            for v in &body.hull.voxels {
                push_vec4(&mut voxels, v.center.extend(v.volume).to_array());
            }
            for p in &body.hull.surface {
                push_vec4(&mut patches, p.position.extend(p.area).to_array());
                push_vec4(&mut patches, p.normal.extend(0.0).to_array());
            }
        }
        voxels.resize(voxels.len().max(VOXEL_BYTES), 0);
        patches.resize(patches.len().max(PATCH_BYTES), 0);
        state.voxels = shader_buffers.add(ShaderBuffer::new(voxels, RenderAssetUsages::default()));
        state.patches =
            shader_buffers.add(ShaderBuffer::new(patches, RenderAssetUsages::default()));
        state.state = shader_buffers.add(ShaderBuffer::new(
            vec![0u8; patch_total.max(1) as usize * 4],
            RenderAssetUsages::default(),
        ));
        state.layout = layout;
    }

    let mut bytes = Vec::with_capacity(list.len() * BODY_BYTES);
    for (i, (_, body, rotation, translation, com)) in list.iter().enumerate() {
        let area = body.hull.area();
        push_vec4(&mut bytes, rotation.to_array());
        push_vec4(
            &mut bytes,
            translation.extend(body.hull.voxel_size).to_array(),
        );
        push_vec4(&mut bytes, com.extend(body.mass).to_array());
        push_vec4(
            &mut bytes,
            body.linear_velocity.extend(body.length).to_array(),
        );
        push_vec4(&mut bytes, body.angular_velocity.extend(area).to_array());
        let ranges = [
            first[i].0,
            body.hull.voxels.len() as u32,
            first[i].1,
            body.hull.surface.len() as u32,
        ];
        for v in ranges {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
    }
    bytes.resize(FLOATING_BODY_MAX * BODY_BYTES, 0);
    // A new asset each frame, as the ocean queries do for their points.
    state.bodies = shader_buffers.add(ShaderBuffer::new(bytes, RenderAssetUsages::default()));

    state.frame = state.frame.wrapping_add(1).max(1);
    let frame = state.frame;
    state
        .pending
        .push_back((frame, list.iter().map(|(e, ..)| *e).collect()));
    while state.pending.len() > PENDING_FRAMES {
        state.pending.pop_front();
    }

    let s = *settings;
    *frame_out = FloatingBodyFrame {
        body_count: list.len() as u32,
        params: ForceParams {
            gravity: s.gravity,
            density: s.density,
            surface_y: s.surface_y,
            dt: dt.max(1.0e-4),
            kinematic_viscosity: s.kinematic_viscosity,
            pressure_linear: s.pressure_linear,
            pressure_quadratic: s.pressure_quadratic,
            pressure_falloff: s.pressure_falloff,
            suction_linear: s.suction_linear,
            suction_quadratic: s.suction_quadratic,
            suction_falloff: s.suction_falloff,
            slam_rate_max: s.slam_rate_max,
            slam_power: s.slam_power,
            body_count: list.len() as u32,
            frame,
            _pad: 0,
        },
        voxels: state.voxels.clone(),
        patches: state.patches.clone(),
        bodies: state.bodies.clone(),
        state: state.state.clone(),
        output: state.output.clone(),
        lod_displacement: lod.displacement.clone(),
        lod_foam: lod.foam.clone(),
        lod_flow: lod.flow.clone(),
    };
    if !has_readback {
        commands
            .entity(readback_entity)
            .insert(Readback::buffer(state.output.clone()));
    }
}

#[derive(Resource)]
struct FloatingBodyPipeline {
    layout: BindGroupLayoutDescriptor,
    pipeline: CachedComputePipelineId,
}

impl FromWorld for FloatingBodyPipeline {
    fn from_world(world: &mut World) -> Self {
        let pipeline_cache = world.resource::<PipelineCache>();
        let asset_server = world.resource::<AssetServer>();
        let unfiltered = TextureSampleType::Float { filterable: false };
        let entries = BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                uniform_buffer::<ForceParams>(false),
                storage_buffer_read_only_sized(false, None),
                storage_buffer_read_only_sized(false, None),
                storage_buffer_read_only_sized(false, None),
                storage_buffer_sized(false, None),
                storage_buffer_sized(false, None),
                uniform_buffer::<WaterLodSlices>(false),
                texture_2d_array(unfiltered),
                texture_2d_array(unfiltered),
                texture_2d_array(unfiltered),
            ),
        );
        let layout = BindGroupLayoutDescriptor::new("floating_body_layout", &entries);
        let pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("floating_body_forces".into()),
            layout: vec![layout.clone()],
            immediate_size: 0,
            shader: asset_server.load(FLOATING_BODY_SHADER),
            shader_defs: vec![],
            entry_point: Some("body_forces".into()),
            zero_initialize_workgroup_memory: false,
            constants: vec![],
        });
        Self { layout, pipeline }
    }
}

#[derive(Resource, Default)]
struct FloatingBodyBindGroup {
    group: Option<BindGroup>,
    params: UniformBuffer<ForceParams>,
}

#[allow(clippy::too_many_arguments)]
fn prepare_floating_bodies(
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    pipeline_cache: Res<PipelineCache>,
    pipeline: Option<Res<FloatingBodyPipeline>>,
    frame: Res<FloatingBodyFrame>,
    buffers: Res<RenderAssets<GpuShaderBuffer>>,
    images: Res<RenderAssets<GpuImage>>,
    lod_gpu: Res<WaterLodGpu>,
    mut bind_group: ResMut<FloatingBodyBindGroup>,
) {
    let state = &mut *bind_group;
    state.group = None;
    let Some(pipeline) = pipeline else {
        return;
    };
    if frame.body_count == 0 {
        return;
    }
    let (Some(voxels), Some(patches), Some(bodies), Some(patch_state), Some(output)) = (
        buffers.get(&frame.voxels),
        buffers.get(&frame.patches),
        buffers.get(&frame.bodies),
        buffers.get(&frame.state),
        buffers.get(&frame.output),
    ) else {
        return;
    };
    let (Some(displacement), Some(foam), Some(flow), Some(slices)) = (
        images.get(&frame.lod_displacement),
        images.get(&frame.lod_foam),
        images.get(&frame.lod_flow),
        lod_gpu.slices.binding(),
    ) else {
        return;
    };
    state.params.set(frame.params);
    state.params.write_buffer(&device, &queue);
    let Some(params) = state.params.binding() else {
        return;
    };
    let layout = pipeline_cache.get_bind_group_layout(&pipeline.layout);
    state.group = Some(device.create_bind_group(
        "floating_body_bind_group",
        &layout,
        &BindGroupEntries::sequential((
            params,
            voxels.buffer.as_entire_binding(),
            patches.buffer.as_entire_binding(),
            bodies.buffer.as_entire_binding(),
            patch_state.buffer.as_entire_binding(),
            output.buffer.as_entire_binding(),
            slices,
            &displacement.texture_view,
            &foam.texture_view,
            &flow.texture_view,
        )),
    ));
}

fn run_floating_bodies(
    mut ctx: RenderContext,
    pipeline: Option<Res<FloatingBodyPipeline>>,
    cache: Res<PipelineCache>,
    frame: Res<FloatingBodyFrame>,
    bind_group: Res<FloatingBodyBindGroup>,
) {
    let (Some(pipeline), Some(group)) = (pipeline, &bind_group.group) else {
        return;
    };
    let Some(compute) = cache.get_compute_pipeline(pipeline.pipeline) else {
        return;
    };
    let mut pass = ctx
        .command_encoder()
        .begin_compute_pass(&ComputePassDescriptor {
            label: Some("floating_body_forces_pass"),
            timestamp_writes: None,
        });
    pass.set_pipeline(compute);
    pass.set_bind_group(0, group, &[]);
    pass.dispatch_workgroups(frame.body_count, 1, 1);
}
