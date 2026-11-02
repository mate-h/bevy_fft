//! The controller-driven [`ShallowWaterSolver`] that [`ShallowWaterPlugin`](super::ShallowWaterPlugin)
//! prepares and steps once per frame on the root render graph.

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

use bevy::{
    app::SubApp,
    prelude::*,
    render::{
        Render, RenderSystems,
        diagnostic::RecordDiagnostics,
        render_asset::RenderAssets,
        render_resource::*,
        renderer::{RenderContext, RenderDevice, RenderQueue},
        texture::GpuImage,
    },
};

use super::{
    GpuInteractionUniform, ShallowWaterController, ShallowWaterInputs, ShallowWaterPipelines,
    ShallowWaterSolver, ShallowWaterSolverDescriptor, ShallowWaterTextures,
};

/// Monotonic step counter passed to the simulation shader (wave boundaries, particle RNG).
#[derive(Resource)]
pub struct ShallowWaterTimestamp(pub AtomicU32);

impl Default for ShallowWaterTimestamp {
    fn default() -> Self {
        Self(AtomicU32::new(0))
    }
}

/// The plugin's solver and reset bookkeeping.
#[derive(Resource)]
pub struct ShallowWaterGpuResources {
    /// Built once the pipelines compile and the grid images reach the GPU, and rebuilt when
    /// they change.
    pub solver: Option<ShallowWaterSolver>,
    pub last_applied_serial: AtomicU32,
    /// Set for this frame when the ocean forcing pass wrote a target height for every cell.
    /// The sim consumes it to seed the wet interior after `loadPreset`.
    pub ocean_seed_written: AtomicBool,
}

impl Default for ShallowWaterGpuResources {
    fn default() -> Self {
        Self {
            solver: None,
            last_applied_serial: AtomicU32::new(0),
            ocean_seed_written: AtomicBool::new(false),
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn prepare_shallow_water_gpu(
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    gpu_images: Res<RenderAssets<GpuImage>>,
    controller: Res<ShallowWaterController>,
    lod: Option<Res<crate::water::WaterLodData>>,
    timestamp: Res<ShallowWaterTimestamp>,
    pipelines: Res<ShallowWaterPipelines>,
    pipeline_cache: Res<PipelineCache>,
    mut gpu: ResMut<ShallowWaterGpuResources>,
) {
    let cells = UVec2::new(controller.cells_x, controller.cells_y);
    let view = |handle: &Handle<Image>| gpu_images.get(handle).map(|g| g.texture_view.clone());
    let textures = ShallowWaterTextures::from_images(&controller.images(), &gpu_images);
    let bed = view(&controller.bed_source);
    let source = controller.source.as_ref().map(view);
    let (Some(textures), Some(bed), None | Some(Some(_))) = (textures, bed, &source) else {
        gpu.solver = None;
        return;
    };
    let inputs = ShallowWaterInputs {
        bed: Some(bed),
        source: source.flatten(),
    };

    if !gpu
        .solver
        .as_ref()
        .is_some_and(|s| s.matches(cells, &textures, controller.particle_count))
    {
        gpu.solver = ShallowWaterSolver::new(
            &render_device,
            &pipelines,
            &pipeline_cache,
            ShallowWaterSolverDescriptor {
                cells,
                textures,
                inputs: inputs.clone(),
                particle_count: controller.particle_count,
            },
        );
    }
    let Some(solver) = gpu.solver.as_mut() else {
        return;
    };
    if !solver.inputs_match(&inputs) {
        solver.set_inputs(&render_device, inputs);
    }

    solver.write_interaction(
        &render_queue,
        &GpuInteractionUniform {
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
            fill_level: controller.fill_level,
        },
    );

    let wind_dir = lod
        .as_ref()
        .map(|lod| Vec2::from_angle(lod.slices.wind_direction))
        .unwrap_or(Vec2::X);
    solver.write_settings(
        &render_queue,
        &controller.settings(timestamp.0.load(Ordering::Relaxed), wind_dir),
    );
}

#[derive(PartialEq, Eq, Debug, Copy, Clone, Hash)]
pub struct ShallowWaterSimLabel;

pub fn run_shallow_water_sim(
    mut ctx: RenderContext,
    controller: Res<ShallowWaterController>,
    gpu_res: Res<ShallowWaterGpuResources>,
    timestamp: Res<ShallowWaterTimestamp>,
) {
    let Some(solver) = gpu_res.solver.as_ref() else {
        return;
    };

    let diagnostics = ctx.diagnostic_recorder();
    let diagnostics = diagnostics.as_deref();
    let enc = ctx.command_encoder();
    let time_span = diagnostics.time_span(enc, "shallow_water");

    let apply_init =
        controller.sim_apply_serial != gpu_res.last_applied_serial.load(Ordering::Relaxed);
    // Wait for a full-grid ocean target before consuming the reset, so the seed is the FFT
    // field and not still water. Without a placement the forcing pass never writes one.
    let defer_init = controller.ocean_forcing
        && controller.placement.is_some()
        && apply_init
        && !gpu_res.ocean_seed_written.load(Ordering::Relaxed);

    {
        let mut pass = enc.begin_compute_pass(&ComputePassDescriptor {
            label: Some("shallow_water_pass"),
            timestamp_writes: None,
        });

        if apply_init {
            solver.encode_clear(&mut pass);
            solver.encode_load_preset(&mut pass);
            if controller.ocean_forcing && !defer_init {
                solver.encode_seed_ocean(&mut pass);
            }
        }

        solver.encode_interact(&mut pass);
        if !controller.paused && !defer_init {
            solver.encode_ocean_band(&mut pass);
            solver.encode_step(&mut pass);
            solver.encode_reconstruct_velocity(&mut pass);
            solver.encode_particles(&mut pass);
        }
    }
    time_span.end(enc);

    if apply_init && !defer_init {
        gpu_res
            .last_applied_serial
            .store(controller.sim_apply_serial, Ordering::Relaxed);
    }

    if !controller.paused && !defer_init {
        timestamp.0.fetch_add(1, Ordering::Relaxed);
    }
}

pub fn plug_shallow_water_render_app(render_app: &mut SubApp) {
    render_app
        .init_resource::<ShallowWaterTimestamp>()
        .init_resource::<ShallowWaterGpuResources>()
        .add_systems(
            Render,
            prepare_shallow_water_gpu.in_set(RenderSystems::PrepareBindGroups),
        );
    render_app.add_systems(
        bevy::render::renderer::RenderGraph,
        run_shallow_water_sim
            .before(bevy::core_pipeline::schedule::camera_driver)
            .in_set(bevy::render::renderer::RenderGraphSystems::Render),
    );
}
