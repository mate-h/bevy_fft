//! Render-world core of the shallow water solver.
//!
//! [`ShallowWaterSolver`] owns the grid's bind groups and uniform buffers and encodes the
//! solver's kernels into a compute pass that the caller begins. It does not schedule anything, so
//! another crate can step it as often as it likes per frame, read or write the face velocities
//! between steps with its own kernels, and change the bed while the water keeps flowing.
//! [`ShallowWaterPlugin`](super::ShallowWaterPlugin) builds one from
//! [`ShallowWaterController`](super::ShallowWaterController) and drives it once per frame.

use bevy::{
    ecs::world::{FromWorld, World},
    prelude::*,
    render::{
        render_asset::RenderAssets,
        render_resource::{
            binding_types::{storage_buffer, texture_2d, texture_storage_2d, uniform_buffer},
            encase::internal::{WriteInto, Writer},
            *,
        },
        renderer::{RenderDevice, RenderQueue},
        texture::GpuImage,
    },
};

use super::{
    GpuInteractionUniform, GpuParticle, GpuSimulationUniform, ShallowWaterBorder,
    ShallowWaterImages, shaders,
};

/// Compute pipelines for every solver kernel, queued once in the render world by
/// [`ShallowWaterSolverPlugin`](super::ShallowWaterSolverPlugin).
#[derive(Resource)]
pub struct ShallowWaterPipelines {
    pub layout_buffers: BindGroupLayoutDescriptor,
    pub layout_settings: BindGroupLayoutDescriptor,
    pub clear: CachedComputePipelineId,
    pub load_preset: CachedComputePipelineId,
    pub sync_bed: CachedComputePipelineId,
    pub interact: CachedComputePipelineId,
    pub ocean_band: CachedComputePipelineId,
    pub mac_u_copy: CachedComputePipelineId,
    pub mac_u_sl_forward: CachedComputePipelineId,
    pub mac_u_sl_reverse: CachedComputePipelineId,
    pub mac_u_combine: CachedComputePipelineId,
    pub mac_w_copy: CachedComputePipelineId,
    pub mac_w_sl_forward: CachedComputePipelineId,
    pub mac_w_sl_reverse: CachedComputePipelineId,
    pub mac_w_combine: CachedComputePipelineId,
    pub integrate_height: CachedComputePipelineId,
    pub integrate_velocity_u: CachedComputePipelineId,
    pub integrate_velocity_w: CachedComputePipelineId,
    pub apply_domain_boundaries: CachedComputePipelineId,
    pub wet_dry_u: CachedComputePipelineId,
    pub wet_dry_w: CachedComputePipelineId,
    pub friction_u: CachedComputePipelineId,
    pub friction_w: CachedComputePipelineId,
    pub pml_step: CachedComputePipelineId,
    pub pml_damp_u: CachedComputePipelineId,
    pub pml_damp_w: CachedComputePipelineId,
    pub overshoot_reduce: CachedComputePipelineId,
    pub reconstruct_cell_velocity: CachedComputePipelineId,
    pub update_particles: CachedComputePipelineId,
    pub seed_height: CachedComputePipelineId,
    pub seed_faces: CachedComputePipelineId,
}

impl FromWorld for ShallowWaterPipelines {
    fn from_world(world: &mut World) -> Self {
        let pipeline_cache = world.resource::<PipelineCache>();
        let asset_server = world.resource::<AssetServer>();

        // Eight storage textures is Metal's limit per kernel. New inputs go in the settings group.
        let layout_buffers = BindGroupLayoutDescriptor::new(
            "shallow_water_buffers_layout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (
                    texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::ReadWrite),
                    texture_storage_2d(TextureFormat::R32Float, StorageTextureAccess::ReadWrite),
                    texture_storage_2d(TextureFormat::R32Float, StorageTextureAccess::ReadWrite),
                    texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::ReadWrite),
                    texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::ReadWrite),
                    texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::ReadWrite),
                    texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::ReadWrite),
                    storage_buffer::<GpuParticle>(false),
                    texture_storage_2d(TextureFormat::R32Float, StorageTextureAccess::ReadWrite),
                ),
            ),
        );

        let layout_settings = BindGroupLayoutDescriptor::new(
            "shallow_water_settings_layout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (
                    uniform_buffer::<GpuInteractionUniform>(false),
                    uniform_buffer::<GpuSimulationUniform>(false),
                    texture_2d(TextureSampleType::Float { filterable: false }),
                    texture_2d(TextureSampleType::Float { filterable: false }),
                ),
            ),
        );

        let layout = vec![layout_buffers.clone(), layout_settings.clone()];
        let shader = asset_server.load(shaders::SHALLOW_WATER_SIMULATOR);

        let q = |label: &'static str, entry: &'static str| {
            pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some(label.into()),
                layout: layout.clone(),
                immediate_size: 0,
                shader: shader.clone(),
                shader_defs: vec![],
                entry_point: Some(entry.into()),
                zero_initialize_workgroup_memory: false,
                constants: vec![],
            })
        };

        Self {
            clear: q("shallow_water_clear", "clearBuffers"),
            load_preset: q("shallow_water_preset", "loadPreset"),
            sync_bed: q("shallow_water_sync_bed", "syncBed"),
            interact: q("shallow_water_interact", "interact"),
            ocean_band: q("shallow_water_ocean_band", "oceanBand"),
            mac_u_copy: q("shallow_water_mac_u_copy", "macU_copy"),
            mac_u_sl_forward: q("shallow_water_mac_u_fwd", "macU_sl_forward"),
            mac_u_sl_reverse: q("shallow_water_mac_u_rev", "macU_sl_reverse"),
            mac_u_combine: q("shallow_water_mac_u_comb", "macU_combine"),
            mac_w_copy: q("shallow_water_mac_w_copy", "macW_copy"),
            mac_w_sl_forward: q("shallow_water_mac_w_fwd", "macW_sl_forward"),
            mac_w_sl_reverse: q("shallow_water_mac_w_rev", "macW_sl_reverse"),
            mac_w_combine: q("shallow_water_mac_w_comb", "macW_combine"),
            integrate_height: q("shallow_water_height", "integrateHeight"),
            integrate_velocity_u: q("shallow_water_vel_u", "integrateVelocityU"),
            integrate_velocity_w: q("shallow_water_vel_w", "integrateVelocityW"),
            apply_domain_boundaries: q("shallow_water_borders", "applyDomainBoundaries"),
            wet_dry_u: q("shallow_water_wet_u", "applyWetDryReflectU"),
            wet_dry_w: q("shallow_water_wet_w", "applyWetDryReflectW"),
            friction_u: q("shallow_water_fric_u", "applyFrictionU"),
            friction_w: q("shallow_water_fric_w", "applyFrictionW"),
            pml_step: q("shallow_water_pml", "pmlStep"),
            pml_damp_u: q("shallow_water_pml_u", "pmlDampFaceU"),
            pml_damp_w: q("shallow_water_pml_w", "pmlDampFaceW"),
            overshoot_reduce: q("shallow_water_overshoot", "overshootReduce"),
            reconstruct_cell_velocity: q("shallow_water_recon_vel", "reconstructCellVelocity"),
            update_particles: q("shallow_water_particles", "updateParticles"),
            seed_height: q("shallow_water_seed_height", "seedOceanHeight"),
            seed_faces: q("shallow_water_seed_faces", "seedOceanFaces"),
            layout_buffers,
            layout_settings,
        }
    }
}

impl ShallowWaterPipelines {
    /// Every kernel resolved from the pipeline cache, or `None` while any is still compiling.
    pub fn kernels(&self, cache: &PipelineCache) -> Option<ShallowWaterKernels> {
        let get = |id: CachedComputePipelineId| cache.get_compute_pipeline(id).cloned();
        Some(ShallowWaterKernels {
            clear: get(self.clear)?,
            load_preset: get(self.load_preset)?,
            sync_bed: get(self.sync_bed)?,
            interact: get(self.interact)?,
            ocean_band: get(self.ocean_band)?,
            mac_u_copy: get(self.mac_u_copy)?,
            mac_u_sl_forward: get(self.mac_u_sl_forward)?,
            mac_u_sl_reverse: get(self.mac_u_sl_reverse)?,
            mac_u_combine: get(self.mac_u_combine)?,
            mac_w_copy: get(self.mac_w_copy)?,
            mac_w_sl_forward: get(self.mac_w_sl_forward)?,
            mac_w_sl_reverse: get(self.mac_w_sl_reverse)?,
            mac_w_combine: get(self.mac_w_combine)?,
            integrate_height: get(self.integrate_height)?,
            integrate_velocity_u: get(self.integrate_velocity_u)?,
            integrate_velocity_w: get(self.integrate_velocity_w)?,
            apply_domain_boundaries: get(self.apply_domain_boundaries)?,
            wet_dry_u: get(self.wet_dry_u)?,
            wet_dry_w: get(self.wet_dry_w)?,
            friction_u: get(self.friction_u)?,
            friction_w: get(self.friction_w)?,
            pml_step: get(self.pml_step)?,
            pml_damp_u: get(self.pml_damp_u)?,
            pml_damp_w: get(self.pml_damp_w)?,
            overshoot_reduce: get(self.overshoot_reduce)?,
            reconstruct_cell_velocity: get(self.reconstruct_cell_velocity)?,
            update_particles: get(self.update_particles)?,
            seed_height: get(self.seed_height)?,
            seed_faces: get(self.seed_faces)?,
        })
    }
}

/// The solver's compiled pipelines, from [`ShallowWaterPipelines::kernels`].
#[derive(Clone)]
pub struct ShallowWaterKernels {
    clear: ComputePipeline,
    load_preset: ComputePipeline,
    sync_bed: ComputePipeline,
    interact: ComputePipeline,
    ocean_band: ComputePipeline,
    mac_u_copy: ComputePipeline,
    mac_u_sl_forward: ComputePipeline,
    mac_u_sl_reverse: ComputePipeline,
    mac_u_combine: ComputePipeline,
    mac_w_copy: ComputePipeline,
    mac_w_sl_forward: ComputePipeline,
    mac_w_sl_reverse: ComputePipeline,
    mac_w_combine: ComputePipeline,
    integrate_height: ComputePipeline,
    integrate_velocity_u: ComputePipeline,
    integrate_velocity_w: ComputePipeline,
    apply_domain_boundaries: ComputePipeline,
    wet_dry_u: ComputePipeline,
    wet_dry_w: ComputePipeline,
    friction_u: ComputePipeline,
    friction_w: ComputePipeline,
    pml_step: ComputePipeline,
    pml_damp_u: ComputePipeline,
    pml_damp_w: ComputePipeline,
    overshoot_reduce: ComputePipeline,
    reconstruct_cell_velocity: ComputePipeline,
    update_particles: ComputePipeline,
    seed_height: ComputePipeline,
    seed_faces: ComputePipeline,
}

/// Views of the grid textures the solver binds for storage. For a grid of `nx × ny` cells,
/// `bed_water`, `velocity`, `pml_state` and `ocean_eta` are `nx × ny`, `flow_x` and `mac_u_temps`
/// are `(nx + 1) × ny`, and `flow_y` and `mac_w_temps` are `nx × (ny + 1)`. `flow_x`, `flow_y` and
/// `ocean_eta` are `R32Float` and the rest `Rgba32Float`. Each texture needs storage binding.
///
/// `bed_water.x` is the bed height H and `.y` the water depth h. `flow_x` holds the face velocity
/// u on vertical faces and `flow_y` the face velocity w on horizontal faces, positive along +x and
/// +y. `velocity.xy` is the cell-centered velocity that
/// [`ShallowWaterSolver::encode_reconstruct_velocity`] writes.
#[derive(Clone)]
pub struct ShallowWaterTextures {
    pub bed_water: TextureView,
    pub flow_x: TextureView,
    pub flow_y: TextureView,
    pub mac_u_temps: TextureView,
    pub mac_w_temps: TextureView,
    pub velocity: TextureView,
    pub pml_state: TextureView,
    pub ocean_eta: TextureView,
}

impl ShallowWaterTextures {
    /// Creates zeroed grid textures for `cells`, usable for storage, sampling, and copies in and
    /// out. A zero `ocean_eta` is only valid with ocean forcing off. [`ShallowWaterImages`] fills
    /// it with the unset marker instead.
    pub fn create(device: &RenderDevice, cells: UVec2) -> Self {
        let view = |label: &'static str, size: UVec2, format: TextureFormat| {
            device
                .create_texture(&TextureDescriptor {
                    label: Some(label),
                    size: Extent3d {
                        width: size.x,
                        height: size.y,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: TextureDimension::D2,
                    format,
                    usage: TextureUsages::STORAGE_BINDING
                        | TextureUsages::TEXTURE_BINDING
                        | TextureUsages::COPY_SRC
                        | TextureUsages::COPY_DST,
                    view_formats: &[],
                })
                .create_view(&TextureViewDescriptor::default())
        };
        let u_faces = cells + UVec2::X;
        let w_faces = cells + UVec2::Y;
        Self {
            bed_water: view("shallow_water_bed_water", cells, TextureFormat::Rgba32Float),
            flow_x: view("shallow_water_flow_x", u_faces, TextureFormat::R32Float),
            flow_y: view("shallow_water_flow_y", w_faces, TextureFormat::R32Float),
            mac_u_temps: view("shallow_water_mac_u", u_faces, TextureFormat::Rgba32Float),
            mac_w_temps: view("shallow_water_mac_w", w_faces, TextureFormat::Rgba32Float),
            velocity: view("shallow_water_velocity", cells, TextureFormat::Rgba32Float),
            pml_state: view("shallow_water_pml_state", cells, TextureFormat::Rgba32Float),
            ocean_eta: view("shallow_water_ocean_eta", cells, TextureFormat::R32Float),
        }
    }

    /// Views of main-world images, or `None` until every one has reached the GPU.
    pub fn from_images(images: &ShallowWaterImages, gpu: &RenderAssets<GpuImage>) -> Option<Self> {
        let view = |handle: &Handle<Image>| gpu.get(handle).map(|g| g.texture_view.clone());
        Some(Self {
            bed_water: view(&images.bed_water)?,
            flow_x: view(&images.flow_x)?,
            flow_y: view(&images.flow_y)?,
            mac_u_temps: view(&images.mac_u_temps)?,
            mac_w_temps: view(&images.mac_w_temps)?,
            velocity: view(&images.velocity)?,
            pml_state: view(&images.pml_state)?,
            ocean_eta: view(&images.ocean_eta)?,
        })
    }

    fn ids(&self) -> [TextureViewId; 8] {
        [
            self.bed_water.id(),
            self.flow_x.id(),
            self.flow_y.id(),
            self.mac_u_temps.id(),
            self.mac_w_temps.id(),
            self.velocity.id(),
            self.pml_state.id(),
            self.ocean_eta.id(),
        ]
    }
}

/// Read-only textures the solver samples, one texel per cell. Missing inputs read as zero.
#[derive(Clone, Default)]
pub struct ShallowWaterInputs {
    /// Bed height per cell, `R32Float`. [`ShallowWaterSolver::encode_sync_bed`] copies it into
    /// `bed_water.x`, and `loadPreset` reads it for
    /// [`SHALLOW_WATER_PRESET_BED_SOURCE`](super::SHALLOW_WATER_PRESET_BED_SOURCE).
    pub bed: Option<TextureView>,
    /// Water sources and sinks per cell, `Rgba32Float`. `.x` is depth added per second, negative
    /// to remove water. `.y` is a sink weight from 0 to 1, the fraction of the depth drained each
    /// step, so 1 drains the cell dry. `.zw` are unused.
    pub source: Option<TextureView>,
}

impl ShallowWaterInputs {
    fn ids(&self) -> [Option<TextureViewId>; 2] {
        [
            self.bed.as_ref().map(|v| v.id()),
            self.source.as_ref().map(|v| v.id()),
        ]
    }
}

/// Parameters for [`ShallowWaterSolver::write_settings`].
#[derive(Clone, Copy, Debug)]
pub struct ShallowWaterSettings {
    /// Time step in seconds. Keep `√(g h) dt / dx` under about 0.5.
    pub dt: f32,
    /// Cell width. Heights, velocities, and gravity share its length unit.
    pub cell_size: f32,
    pub gravity: f32,
    /// Linear drag, the fraction of face velocity lost per second, from 0 to 1.
    pub friction: f32,
    /// Manning roughness n. Zero turns Manning friction off. Faces are slowed implicitly by
    /// `u / (1 + g dt n² |u| / h^(4/3))` with h the mean depth of the two cells.
    pub manning_n: f32,
    /// Fraction of the water depth that evaporates per second.
    pub evaporation: f32,
    pub left_border: ShallowWaterBorder,
    pub right_border: ShallowWaterBorder,
    pub bottom_border: ShallowWaterBorder,
    pub top_border: ShallowWaterBorder,
    /// PML strip width in cells. Zero turns the PML off.
    pub pml_width: u32,
    /// Target free-surface height in the PML and the ocean band.
    pub pml_eta_rest: f32,
    /// PML absorption ramp, 2 for quadratic and 3 for cubic.
    pub pml_sigma_exponent: f32,
    /// PML scaling by the cosine of the flow direction, from 0 for none to 1 for full.
    pub pml_cosine_blend: f32,
    /// Width in cells of the band where [`ShallowWaterBorder::Ocean`] edges blend toward the
    /// target surface.
    pub ocean_band_cells: f32,
    /// Reads the target surface from `ocean_eta` instead of `pml_eta_rest`.
    pub ocean_forcing: bool,
    /// Limits overshoot where the surface steps sharply between cells.
    pub overshoot_limiter: bool,
    /// Unit wind direction on the grid, for the ocean seed.
    pub wind_dir: Vec2,
    /// Step counter for the wave maker edge and particle respawns.
    pub timestamp: u32,
}

impl Default for ShallowWaterSettings {
    fn default() -> Self {
        Self {
            dt: 0.08,
            cell_size: 1.0,
            gravity: 10.0,
            friction: 0.0,
            manning_n: 0.0,
            evaporation: 0.0,
            left_border: ShallowWaterBorder::Wall,
            right_border: ShallowWaterBorder::Wall,
            bottom_border: ShallowWaterBorder::Wall,
            top_border: ShallowWaterBorder::Wall,
            pml_width: 0,
            pml_eta_rest: 2.0,
            pml_sigma_exponent: 2.0,
            pml_cosine_blend: 0.0,
            ocean_band_cells: 5.0,
            ocean_forcing: false,
            overshoot_limiter: true,
            wind_dir: Vec2::X,
            timestamp: 0,
        }
    }
}

impl ShallowWaterSettings {
    pub fn packed_border_mask(&self) -> u32 {
        self.left_border.bits()
            | (self.right_border.bits() << 3)
            | (self.bottom_border.bits() << 6)
            | (self.top_border.bits() << 9)
    }

    fn uniform(&self, cells: UVec2, has_source: bool) -> GpuSimulationUniform {
        let dx = self.cell_size;
        let friction_factor = (1.0 - self.friction.clamp(0.0, 1.0)).powf(self.dt);
        let flags = u32::from(self.overshoot_limiter)
            | (u32::from(self.ocean_forcing) << 1)
            | (u32::from(has_source) << 2);
        GpuSimulationUniform {
            size: cells,
            dt: self.dt,
            dx,
            gravity: self.gravity,
            friction_factor,
            timestamp: self.timestamp,
            border_mask: self.packed_border_mask(),
            pml_width: self.pml_width,
            flags,
            pml_eta_rest: self.pml_eta_rest,
            vel_clamp_alpha: 0.5,
            h_avgmax_beta: 2.0,
            eps_wet: 1e-4 * dx,
            pml_lambda_decay: 0.9,
            pml_lambda_update: 0.1,
            pml_sigma_max: 6.0,
            overshoot_alpha: 0.25,
            overshoot_lambda_edge: 2.0 * dx,
            pml_sigma_exponent: self.pml_sigma_exponent.clamp(1.0, 4.0),
            pml_cosine_blend: self.pml_cosine_blend.clamp(0.0, 1.0),
            ocean_band_cells: self.ocean_band_cells.max(0.0),
            manning_n2: self.manning_n * self.manning_n,
            evaporation: self.evaporation.max(0.0),
            wind_dir: self.wind_dir,
        }
    }
}

/// What [`ShallowWaterSolver::new`] builds on.
#[derive(Clone)]
pub struct ShallowWaterSolverDescriptor {
    /// Grid size in cells.
    pub cells: UVec2,
    pub textures: ShallowWaterTextures,
    pub inputs: ShallowWaterInputs,
    /// Passive tracer particles, rounded with [`round_particle_count`]. The buffer always holds at
    /// least 64.
    pub particle_count: u32,
}

/// Particle count rounded down to a multiple of the 64-wide particle workgroup, at least 64.
pub fn round_particle_count(n: u32) -> u32 {
    let n = n.max(64);
    (n / 64) * 64
}

const SIM_WORKGROUP: u32 = 16;

fn sim_wg2(size: UVec2) -> UVec2 {
    size.map(|n| n.div_ceil(SIM_WORKGROUP))
}

fn write_uniform<T: ShaderType + WriteInto>(queue: &RenderQueue, buffer: &Buffer, value: &T) {
    let mut dest = Vec::new();
    value.write_into(&mut Writer::new(value, &mut dest, 0).unwrap());
    queue.write_buffer(buffer, 0, &dest);
}

/// One shallow water grid on the GPU, ready to encode.
///
/// Each `encode_*` method sets the solver's pipelines and bind groups and dispatches into a pass
/// the caller owns. Dispatches in one pass run in order, so a caller can put its own kernels
/// between them as long as it binds its own groups. Settings come from the uniforms last written
/// with [`Self::write_settings`] and [`Self::write_interaction`], which take effect for the next
/// submitted command buffer.
pub struct ShallowWaterSolver {
    cells: UVec2,
    kernels: ShallowWaterKernels,
    textures: ShallowWaterTextures,
    inputs: ShallowWaterInputs,
    interaction: Buffer,
    simulation: Buffer,
    particles: Buffer,
    particle_count: u32,
    settings_layout: BindGroupLayout,
    buffers_group: BindGroup,
    settings_group: BindGroup,
    fallback_bed: TextureView,
    fallback_source: TextureView,
}

impl ShallowWaterSolver {
    /// Builds the bind groups and uniform buffers for a grid, or returns `None` while the
    /// pipelines are still compiling.
    pub fn new(
        device: &RenderDevice,
        pipelines: &ShallowWaterPipelines,
        cache: &PipelineCache,
        descriptor: ShallowWaterSolverDescriptor,
    ) -> Option<Self> {
        let kernels = pipelines.kernels(cache)?;
        let uniform = |label: &'static str, size: u64| {
            device.create_buffer(&BufferDescriptor {
                label: Some(label),
                size,
                usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let interaction = uniform(
            "shallow_water_interaction_uniform",
            GpuInteractionUniform::min_size().get(),
        );
        let simulation = uniform(
            "shallow_water_simulation_uniform",
            GpuSimulationUniform::min_size().get(),
        );
        let particle_count = round_particle_count(descriptor.particle_count);
        let particles = device.create_buffer(&BufferDescriptor {
            label: Some("shallow_water_particles"),
            size: (std::mem::size_of::<GpuParticle>() as u64) * (particle_count as u64),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let fallback = |label: &'static str, format: TextureFormat| {
            device
                .create_texture(&TextureDescriptor {
                    label: Some(label),
                    size: Extent3d::default(),
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: TextureDimension::D2,
                    format,
                    usage: TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                })
                .create_view(&TextureViewDescriptor::default())
        };

        let buffers_layout = cache.get_bind_group_layout(&pipelines.layout_buffers);
        let textures = &descriptor.textures;
        let buffers_group = device.create_bind_group(
            "shallow_water_buffers",
            &buffers_layout,
            &BindGroupEntries::sequential((
                &textures.bed_water,
                &textures.flow_x,
                &textures.flow_y,
                &textures.mac_u_temps,
                &textures.mac_w_temps,
                &textures.velocity,
                &textures.pml_state,
                particles.as_entire_buffer_binding(),
                &textures.ocean_eta,
            )),
        );

        let settings_layout = cache.get_bind_group_layout(&pipelines.layout_settings);
        let fallback_bed = fallback("shallow_water_no_bed", TextureFormat::R32Float);
        let fallback_source = fallback("shallow_water_no_source", TextureFormat::Rgba32Float);
        let settings_group = device.create_bind_group(
            "shallow_water_settings",
            &settings_layout,
            &BindGroupEntries::sequential((
                interaction.as_entire_buffer_binding(),
                simulation.as_entire_buffer_binding(),
                descriptor.inputs.bed.as_ref().unwrap_or(&fallback_bed),
                descriptor
                    .inputs
                    .source
                    .as_ref()
                    .unwrap_or(&fallback_source),
            )),
        );

        Some(Self {
            cells: descriptor.cells,
            kernels,
            textures: descriptor.textures,
            inputs: descriptor.inputs,
            interaction,
            simulation,
            particles,
            particle_count,
            settings_layout,
            buffers_group,
            settings_group,
            fallback_bed,
            fallback_source,
        })
    }

    /// Grid size in cells.
    pub fn cells(&self) -> UVec2 {
        self.cells
    }

    pub fn textures(&self) -> &ShallowWaterTextures {
        &self.textures
    }

    pub fn inputs(&self) -> &ShallowWaterInputs {
        &self.inputs
    }

    pub fn particles(&self) -> &Buffer {
        &self.particles
    }

    pub fn particle_count(&self) -> u32 {
        self.particle_count
    }

    /// Rebinds the bed and source textures.
    pub fn set_inputs(&mut self, device: &RenderDevice, inputs: ShallowWaterInputs) {
        self.settings_group = device.create_bind_group(
            "shallow_water_settings",
            &self.settings_layout,
            &BindGroupEntries::sequential((
                self.interaction.as_entire_buffer_binding(),
                self.simulation.as_entire_buffer_binding(),
                inputs.bed.as_ref().unwrap_or(&self.fallback_bed),
                inputs.source.as_ref().unwrap_or(&self.fallback_source),
            )),
        );
        self.inputs = inputs;
    }

    /// Writes the step parameters. Sources apply only when [`ShallowWaterInputs::source`] is set.
    pub fn write_settings(&self, queue: &RenderQueue, settings: &ShallowWaterSettings) {
        let uniform = settings.uniform(self.cells, self.inputs.source.is_some());
        write_uniform(queue, &self.simulation, &uniform);
    }

    /// Writes the brush and preset parameters for [`Self::encode_interact`] and
    /// [`Self::encode_load_preset`].
    pub fn write_interaction(&self, queue: &RenderQueue, interaction: &GpuInteractionUniform) {
        write_uniform(queue, &self.interaction, interaction);
    }

    fn dispatch(&self, pass: &mut ComputePass, pipeline: &ComputePipeline, groups: UVec2) {
        pass.set_pipeline(pipeline);
        pass.dispatch_workgroups(groups.x, groups.y, 1);
    }

    fn bind(&self, pass: &mut ComputePass) {
        pass.set_bind_group(0, &self.buffers_group, &[]);
        pass.set_bind_group(1, &self.settings_group, &[]);
    }

    fn cell_groups(&self) -> UVec2 {
        sim_wg2(self.cells)
    }

    fn u_face_groups(&self) -> UVec2 {
        sim_wg2(self.cells + UVec2::X)
    }

    fn w_face_groups(&self) -> UVec2 {
        sim_wg2(self.cells + UVec2::Y)
    }

    fn border_groups(&self) -> UVec2 {
        sim_wg2(self.cells + UVec2::ONE)
    }

    /// Zeroes every grid texture except `ocean_eta`.
    pub fn encode_clear(&self, pass: &mut ComputePass) {
        self.bind(pass);
        self.dispatch(pass, &self.kernels.clear, self.border_groups());
    }

    /// Writes the preset bed into `bed_water` and fills it to the interaction's fill level.
    pub fn encode_load_preset(&self, pass: &mut ComputePass) {
        self.bind(pass);
        self.dispatch(pass, &self.kernels.load_preset, self.cell_groups());
    }

    /// Copies the ocean target into wet cells with a wind-aligned column velocity.
    pub fn encode_seed_ocean(&self, pass: &mut ComputePass) {
        self.bind(pass);
        self.dispatch(pass, &self.kernels.seed_height, self.cell_groups());
        self.dispatch(pass, &self.kernels.seed_faces, self.border_groups());
    }

    /// Applies the brush from the interaction uniform.
    pub fn encode_interact(&self, pass: &mut ComputePass) {
        self.bind(pass);
        self.dispatch(pass, &self.kernels.interact, self.cell_groups());
    }

    /// Blends the depth toward the target surface along [`ShallowWaterBorder::Ocean`] edges.
    pub fn encode_ocean_band(&self, pass: &mut ComputePass) {
        self.bind(pass);
        self.dispatch(pass, &self.kernels.ocean_band, self.cell_groups());
    }

    /// Copies the bed input into `bed_water.x` and leaves the water depth as it is.
    pub fn encode_sync_bed(&self, pass: &mut ComputePass) {
        self.bind(pass);
        self.dispatch(pass, &self.kernels.sync_bed, self.cell_groups());
    }

    /// One time step: MacCormack advection of the face velocities, the depth update with sources,
    /// pressure, domain edges, wet and dry faces, friction, the PML, and the overshoot limiter.
    pub fn encode_step(&self, pass: &mut ComputePass) {
        let k = &self.kernels;
        let (cell, u, w, border) = (
            self.cell_groups(),
            self.u_face_groups(),
            self.w_face_groups(),
            self.border_groups(),
        );
        self.bind(pass);
        self.dispatch(pass, &k.mac_u_copy, u);
        self.dispatch(pass, &k.mac_u_sl_forward, u);
        self.dispatch(pass, &k.mac_u_sl_reverse, u);
        self.dispatch(pass, &k.mac_u_combine, u);
        self.dispatch(pass, &k.mac_w_copy, w);
        self.dispatch(pass, &k.mac_w_sl_forward, w);
        self.dispatch(pass, &k.mac_w_sl_reverse, w);
        self.dispatch(pass, &k.mac_w_combine, w);
        self.dispatch(pass, &k.integrate_height, cell);
        self.dispatch(pass, &k.integrate_velocity_u, u);
        self.dispatch(pass, &k.integrate_velocity_w, w);
        self.dispatch(pass, &k.apply_domain_boundaries, border);
        self.dispatch(pass, &k.wet_dry_u, u);
        self.dispatch(pass, &k.wet_dry_w, w);
        self.dispatch(pass, &k.friction_u, u);
        self.dispatch(pass, &k.friction_w, w);
        self.dispatch(pass, &k.pml_step, cell);
        self.dispatch(pass, &k.pml_damp_u, u);
        self.dispatch(pass, &k.pml_damp_w, w);
        self.dispatch(pass, &k.apply_domain_boundaries, border);
        self.dispatch(pass, &k.overshoot_reduce, cell);
    }

    /// Averages the face velocities into the cell-centered `velocity` texture.
    pub fn encode_reconstruct_velocity(&self, pass: &mut ComputePass) {
        self.bind(pass);
        self.dispatch(
            pass,
            &self.kernels.reconstruct_cell_velocity,
            self.cell_groups(),
        );
    }

    /// Advects the tracer particles with the cell-centered velocity.
    pub fn encode_particles(&self, pass: &mut ComputePass) {
        self.bind(pass);
        pass.set_pipeline(&self.kernels.update_particles);
        pass.dispatch_workgroups(self.particle_count / 64, 1, 1);
    }

    /// Whether the solver was built for these cells, textures, and particle count.
    pub(super) fn matches(
        &self,
        cells: UVec2,
        textures: &ShallowWaterTextures,
        particle_count: u32,
    ) -> bool {
        self.cells == cells
            && self.textures.ids() == textures.ids()
            && self.particle_count == round_particle_count(particle_count)
    }

    /// Whether the solver binds these inputs.
    pub(super) fn inputs_match(&self, inputs: &ShallowWaterInputs) -> bool {
        self.inputs.ids() == inputs.ids()
    }
}
