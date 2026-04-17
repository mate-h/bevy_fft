//! Compute pipelines, GPU buffers, bind groups, and the root render-graph node for shallow water.

use std::sync::atomic::{AtomicU32, Ordering};

use bevy::{
    app::SubApp,
    asset::AssetServer,
    ecs::world::{FromWorld, World},
    prelude::*,
    render::{
        Render, RenderSystems,
        graph::CameraDriverLabel,
        render_asset::RenderAssets,
        render_graph::{Node, NodeRunError, RenderGraph, RenderGraphContext, RenderLabel},
        render_resource::{
            binding_types::{storage_buffer, texture_storage_2d, uniform_buffer},
            encase::internal::{WriteInto, Writer},
            *,
        },
        renderer::{RenderContext, RenderDevice, RenderQueue},
        texture::GpuImage,
    },
};

use super::{GpuInteractionUniform, GpuParticle, GpuSimulationUniform, ShallowWaterController};

/// Monotonic step counter passed to the simulation shader (wave boundaries, particle RNG).
#[derive(Resource)]
pub struct ShallowWaterTimestamp(pub AtomicU32);

impl Default for ShallowWaterTimestamp {
    fn default() -> Self {
        Self(AtomicU32::new(0))
    }
}

#[derive(Resource)]
pub struct ShallowWaterPipelines {
    pub layout_buffers: BindGroupLayoutDescriptor,
    pub layout_settings: BindGroupLayoutDescriptor,
    pub clear: CachedComputePipelineId,
    pub load_preset: CachedComputePipelineId,
    pub interact: CachedComputePipelineId,
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
}

impl FromWorld for ShallowWaterPipelines {
    fn from_world(world: &mut World) -> Self {
        let pipeline_cache = world.resource::<PipelineCache>();
        let asset_server = world.resource::<AssetServer>();

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
                ),
            ),
        );

        let layout = vec![layout_buffers.clone(), layout_settings.clone()];
        let shader = asset_server.load("shallow_water/simulator.wgsl");

        let q = |label: &'static str, entry: &'static str| {
            pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some(label.into()),
                layout: layout.clone(),
                push_constant_ranges: vec![],
                shader: shader.clone(),
                shader_defs: vec![],
                entry_point: Some(entry.into()),
                zero_initialize_workgroup_memory: false,
            })
        };

        Self {
            layout_buffers,
            layout_settings,
            clear: q("shallow_water_clear", "clearBuffers"),
            load_preset: q("shallow_water_preset", "loadPreset"),
            interact: q("shallow_water_interact", "interact"),
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
            update_particles: pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some("shallow_water_particles".into()),
                layout,
                push_constant_ranges: vec![],
                shader,
                shader_defs: vec![],
                entry_point: Some("updateParticles".into()),
                zero_initialize_workgroup_memory: false,
            }),
        }
    }
}

#[derive(Resource)]
pub struct ShallowWaterGpuResources {
    pub interaction: Option<Buffer>,
    pub simulation: Option<Buffer>,
    pub particles: Option<Buffer>,
    pub particle_count: u32,
    pub buffer_bind_group: Option<BindGroup>,
    pub settings_bind_group: Option<BindGroup>,
    pub last_applied_serial: AtomicU32,
}

impl Default for ShallowWaterGpuResources {
    fn default() -> Self {
        Self {
            interaction: None,
            simulation: None,
            particles: None,
            particle_count: 0,
            buffer_bind_group: None,
            settings_bind_group: None,
            last_applied_serial: AtomicU32::new(0),
        }
    }
}

pub fn round_particle_count(n: u32) -> u32 {
    let n = n.max(64);
    (n / 64) * 64
}

fn write_uniform<T: ShaderType + WriteInto>(queue: &RenderQueue, buffer: &Buffer, value: &T) {
    let mut dest = Vec::new();
    value.write_into(&mut Writer::new(value, &mut dest, 0).unwrap());
    queue.write_buffer(buffer, 0, &dest);
}

const SIM_WORKGROUP: u32 = 16;

fn sim_wg2(w: u32, h: u32) -> (u32, u32) {
    (w.div_ceil(SIM_WORKGROUP), h.div_ceil(SIM_WORKGROUP))
}

struct SimWorkgroups {
    cell: (u32, u32),
    u_face: (u32, u32),
    w_face: (u32, u32),
    border: (u32, u32),
}

impl SimWorkgroups {
    fn new(nx: u32, ny: u32) -> Self {
        Self {
            cell: sim_wg2(nx, ny),
            u_face: sim_wg2(nx + 1, ny),
            w_face: sim_wg2(nx, ny + 1),
            border: sim_wg2(nx + 1, ny + 1),
        }
    }
}

fn build_simulation_uniform(
    controller: &ShallowWaterController,
    timestamp: u32,
) -> GpuSimulationUniform {
    let dx = 1.0_f32;
    let friction_factor = (1.0 - controller.friction.clamp(0.0, 1.0)).powf(controller.dt);
    GpuSimulationUniform {
        size: UVec2::new(controller.cells_x, controller.cells_y),
        dt: controller.dt,
        dx,
        gravity: controller.gravity,
        friction_factor,
        timestamp,
        border_mask: controller.packed_border_mask(),
        pml_width: controller.pml_width,
        flags: 1,
        pml_eta_rest: controller.pml_eta_rest,
        vel_clamp_alpha: 0.5,
        h_avgmax_beta: 2.0,
        eps_wet: 1e-4 * dx,
        pml_lambda_decay: 0.9,
        pml_lambda_update: 0.1,
        pml_sigma_max: 6.0,
        overshoot_alpha: 0.25,
        overshoot_lambda_edge: 2.0 * dx,
        pml_sigma_exponent: controller.pml_sigma_exponent.clamp(1.0, 4.0),
        pml_cosine_blend: controller.pml_cosine_blend.clamp(0.0, 1.0),
    }
}

#[allow(clippy::too_many_arguments)]
pub fn prepare_shallow_water_gpu(
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    gpu_images: Res<RenderAssets<GpuImage>>,
    controller: Res<ShallowWaterController>,
    timestamp: Res<ShallowWaterTimestamp>,
    pipelines: Res<ShallowWaterPipelines>,
    pipeline_cache: Res<PipelineCache>,
    mut gpu: ResMut<ShallowWaterGpuResources>,
) {
    let particles = round_particle_count(controller.particle_count);

    if gpu.interaction.is_none() {
        gpu.interaction = Some(render_device.create_buffer(&BufferDescriptor {
            label: Some("shallow_water_interaction_uniform"),
            size: GpuInteractionUniform::min_size().get(),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
    }
    if gpu.simulation.is_none() {
        gpu.simulation = Some(render_device.create_buffer(&BufferDescriptor {
            label: Some("shallow_water_simulation_uniform"),
            size: GpuSimulationUniform::min_size().get(),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
    }

    if gpu.particles.is_none() || gpu.particle_count != particles {
        gpu.particle_count = particles;
        let size = (std::mem::size_of::<GpuParticle>() as u64) * (particles as u64);
        gpu.particles = Some(render_device.create_buffer(&BufferDescriptor {
            label: Some("shallow_water_particles"),
            size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        gpu.buffer_bind_group = None;
    }

    let Some(interaction_buf) = gpu.interaction.as_ref() else {
        return;
    };
    let Some(simulation_buf) = gpu.simulation.as_ref() else {
        return;
    };
    let Some(particles_buf) = gpu.particles.as_ref() else {
        return;
    };

    let interaction = GpuInteractionUniform {
        mode: if controller.brush_input_active {
            controller.interaction_mode
        } else {
            0
        },
        radius: controller.brush_radius,
        force: controller.brush_force,
        dt: controller.dt,
        old_position: controller.pointer_prev_sim,
        position: controller.pointer_sim,
        preset: controller.preset_index,
    };
    write_uniform(&render_queue, interaction_buf, &interaction);

    let simulation =
        build_simulation_uniform(controller.as_ref(), timestamp.0.load(Ordering::Relaxed));
    write_uniform(&render_queue, simulation_buf, &simulation);

    let Some(bed) = gpu_images.get(&controller.bed_water) else {
        gpu.buffer_bind_group = None;
        gpu.settings_bind_group = None;
        return;
    };
    let Some(vel) = gpu_images.get(&controller.velocity) else {
        gpu.buffer_bind_group = None;
        gpu.settings_bind_group = None;
        return;
    };
    let Some(fx) = gpu_images.get(&controller.flow_x) else {
        gpu.buffer_bind_group = None;
        gpu.settings_bind_group = None;
        return;
    };
    let Some(fy) = gpu_images.get(&controller.flow_y) else {
        gpu.buffer_bind_group = None;
        gpu.settings_bind_group = None;
        return;
    };
    let Some(mac_u) = gpu_images.get(&controller.mac_u_temps) else {
        gpu.buffer_bind_group = None;
        gpu.settings_bind_group = None;
        return;
    };
    let Some(mac_w) = gpu_images.get(&controller.mac_w_temps) else {
        gpu.buffer_bind_group = None;
        gpu.settings_bind_group = None;
        return;
    };
    let Some(pml) = gpu_images.get(&controller.pml_state) else {
        gpu.buffer_bind_group = None;
        gpu.settings_bind_group = None;
        return;
    };

    let buffers_layout = pipeline_cache.get_bind_group_layout(&pipelines.layout_buffers);
    let settings_layout = pipeline_cache.get_bind_group_layout(&pipelines.layout_settings);

    let buffer_bind_group = render_device.create_bind_group(
        "shallow_water_buffers",
        &buffers_layout,
        &BindGroupEntries::sequential((
            &bed.texture_view,
            &fx.texture_view,
            &fy.texture_view,
            &mac_u.texture_view,
            &mac_w.texture_view,
            &vel.texture_view,
            &pml.texture_view,
            particles_buf.as_entire_buffer_binding(),
        )),
    );

    let settings_bind_group = render_device.create_bind_group(
        "shallow_water_settings",
        &settings_layout,
        &BindGroupEntries::sequential((
            interaction_buf.as_entire_buffer_binding(),
            simulation_buf.as_entire_buffer_binding(),
        )),
    );

    gpu.buffer_bind_group = Some(buffer_bind_group);
    gpu.settings_bind_group = Some(settings_bind_group);
}

#[derive(PartialEq, Eq, Debug, Copy, Clone, Hash, RenderLabel)]
pub struct ShallowWaterSimLabel;

pub struct ShallowWaterSimNode;

impl FromWorld for ShallowWaterSimNode {
    fn from_world(_world: &mut World) -> Self {
        Self
    }
}

impl Node for ShallowWaterSimNode {
    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), NodeRunError> {
        if world.get_resource::<ShallowWaterController>().is_none() {
            return Ok(());
        }

        let pl = world.resource::<ShallowWaterPipelines>();
        let cache = world.resource::<PipelineCache>();
        let controller = world.resource::<ShallowWaterController>();
        let gpu_res = world.resource::<ShallowWaterGpuResources>();
        let timestamp = world.resource::<ShallowWaterTimestamp>();

        let Some(bg0) = gpu_res.buffer_bind_group.as_ref() else {
            return Ok(());
        };
        let Some(bg1) = gpu_res.settings_bind_group.as_ref() else {
            return Ok(());
        };

        macro_rules! pipe {
            ($field:ident) => {
                cache.get_compute_pipeline(pl.$field)
            };
        }

        let Some(p_clear) = pipe!(clear) else {
            return Ok(());
        };
        let Some(p_preset) = pipe!(load_preset) else {
            return Ok(());
        };
        let Some(p_interact) = pipe!(interact) else {
            return Ok(());
        };

        macro_rules! require_pipe {
            ($name:ident) => {
                match pipe!($name) {
                    Some(p) => p,
                    None => return Ok(()),
                }
            };
        }

        let wg = SimWorkgroups::new(controller.cells_x, controller.cells_y);
        let pc = gpu_res.particle_count / 64;

        let enc = render_context.command_encoder();

        let apply_init =
            controller.sim_apply_serial != gpu_res.last_applied_serial.load(Ordering::Relaxed);

        {
            let mut pass = enc.begin_compute_pass(&ComputePassDescriptor {
                label: Some("shallow_water_pass"),
                timestamp_writes: None,
            });

            pass.set_bind_group(0, bg0, &[]);
            pass.set_bind_group(1, bg1, &[]);

            if apply_init {
                pass.set_pipeline(p_clear);
                pass.dispatch_workgroups(wg.border.0, wg.border.1, 1);
                pass.set_pipeline(p_preset);
                pass.dispatch_workgroups(wg.cell.0, wg.cell.1, 1);
            }

            pass.set_pipeline(p_interact);
            pass.dispatch_workgroups(wg.cell.0, wg.cell.1, 1);

            if !controller.paused {
                pass.set_pipeline(require_pipe!(mac_u_copy));
                pass.dispatch_workgroups(wg.u_face.0, wg.u_face.1, 1);
                pass.set_pipeline(require_pipe!(mac_u_sl_forward));
                pass.dispatch_workgroups(wg.u_face.0, wg.u_face.1, 1);
                pass.set_pipeline(require_pipe!(mac_u_sl_reverse));
                pass.dispatch_workgroups(wg.u_face.0, wg.u_face.1, 1);
                pass.set_pipeline(require_pipe!(mac_u_combine));
                pass.dispatch_workgroups(wg.u_face.0, wg.u_face.1, 1);

                pass.set_pipeline(require_pipe!(mac_w_copy));
                pass.dispatch_workgroups(wg.w_face.0, wg.w_face.1, 1);
                pass.set_pipeline(require_pipe!(mac_w_sl_forward));
                pass.dispatch_workgroups(wg.w_face.0, wg.w_face.1, 1);
                pass.set_pipeline(require_pipe!(mac_w_sl_reverse));
                pass.dispatch_workgroups(wg.w_face.0, wg.w_face.1, 1);
                pass.set_pipeline(require_pipe!(mac_w_combine));
                pass.dispatch_workgroups(wg.w_face.0, wg.w_face.1, 1);

                pass.set_pipeline(require_pipe!(integrate_height));
                pass.dispatch_workgroups(wg.cell.0, wg.cell.1, 1);
                pass.set_pipeline(require_pipe!(integrate_velocity_u));
                pass.dispatch_workgroups(wg.u_face.0, wg.u_face.1, 1);
                pass.set_pipeline(require_pipe!(integrate_velocity_w));
                pass.dispatch_workgroups(wg.w_face.0, wg.w_face.1, 1);
                pass.set_pipeline(require_pipe!(apply_domain_boundaries));
                pass.dispatch_workgroups(wg.border.0, wg.border.1, 1);
                pass.set_pipeline(require_pipe!(wet_dry_u));
                pass.dispatch_workgroups(wg.u_face.0, wg.u_face.1, 1);
                pass.set_pipeline(require_pipe!(wet_dry_w));
                pass.dispatch_workgroups(wg.w_face.0, wg.w_face.1, 1);
                pass.set_pipeline(require_pipe!(friction_u));
                pass.dispatch_workgroups(wg.u_face.0, wg.u_face.1, 1);
                pass.set_pipeline(require_pipe!(friction_w));
                pass.dispatch_workgroups(wg.w_face.0, wg.w_face.1, 1);
                pass.set_pipeline(require_pipe!(pml_step));
                pass.dispatch_workgroups(wg.cell.0, wg.cell.1, 1);
                pass.set_pipeline(require_pipe!(pml_damp_u));
                pass.dispatch_workgroups(wg.u_face.0, wg.u_face.1, 1);
                pass.set_pipeline(require_pipe!(pml_damp_w));
                pass.dispatch_workgroups(wg.w_face.0, wg.w_face.1, 1);
                pass.set_pipeline(require_pipe!(apply_domain_boundaries));
                pass.dispatch_workgroups(wg.border.0, wg.border.1, 1);
                pass.set_pipeline(require_pipe!(overshoot_reduce));
                pass.dispatch_workgroups(wg.cell.0, wg.cell.1, 1);
                pass.set_pipeline(require_pipe!(reconstruct_cell_velocity));
                pass.dispatch_workgroups(wg.cell.0, wg.cell.1, 1);
                pass.set_pipeline(require_pipe!(update_particles));
                pass.dispatch_workgroups(pc, 1, 1);
            }
        }

        if apply_init {
            gpu_res
                .last_applied_serial
                .store(controller.sim_apply_serial, Ordering::Relaxed);
        }

        if !controller.paused {
            timestamp.0.fetch_add(1, Ordering::Relaxed);
        }

        Ok(())
    }
}

/// Inserts [`ShallowWaterSimLabel`] before [`CameraDriverLabel`].
pub fn splice_shallow_water_before_camera(world: &mut World) {
    let Some(mut graph) = world.get_resource_mut::<RenderGraph>() else {
        return;
    };
    graph.add_node_edge(ShallowWaterSimLabel, CameraDriverLabel);
}

pub fn plug_shallow_water_render_app(render_app: &mut SubApp) {
    render_app
        .init_resource::<ShallowWaterTimestamp>()
        .init_resource::<ShallowWaterPipelines>()
        .init_resource::<ShallowWaterGpuResources>()
        .add_systems(
            Render,
            prepare_shallow_water_gpu.in_set(RenderSystems::PrepareBindGroups),
        );

    render_app
        .world_mut()
        .resource_scope(|world, mut graph: Mut<RenderGraph>| {
            graph.add_node(ShallowWaterSimLabel, ShallowWaterSimNode::from_world(world));
        });
    splice_shallow_water_before_camera(render_app.world_mut());
}
