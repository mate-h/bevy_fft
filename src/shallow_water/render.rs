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
    GpuInteractionUniform, NEST_REFINEMENT, ShallowWaterBorder, ShallowWaterController,
    ShallowWaterInputs, ShallowWaterNest, ShallowWaterNestDescriptor, ShallowWaterNestPipelines,
    ShallowWaterPipelines, ShallowWaterShift, ShallowWaterShiftParams, ShallowWaterShiftPipelines,
    ShallowWaterSolver, ShallowWaterSolverDescriptor, ShallowWaterTextures, ShallowWaterWorldBed,
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
    /// The controller's nested grid, built after the solver.
    pub nested: Option<ShallowWaterNestedGpu>,
    /// [`ShallowWaterController::settled`] on the GPU while it fits the grid.
    settled: Option<SettledGpu>,
}

/// An uploaded settled state and the grid textures it loads into.
struct SettledGpu {
    /// The state it was uploaded from.
    from: std::sync::Arc<super::ShallowWaterSettled>,
    sources: [Texture; 3],
    targets: [Texture; 3],
}

/// The render-world side of a [`ShallowWaterNestedGrid`](super::ShallowWaterNestedGrid).
pub struct ShallowWaterNestedGpu {
    pub child: ShallowWaterSolver,
    nest: ShallowWaterNest,
    shift: ShallowWaterShift,
    bed: ShallowWaterWorldBed,
    /// Parent cell under the child's first cell, once the child holds water.
    applied: Option<IVec2>,
    /// Whether the child shifts this frame, with the parameters already written.
    shifting: bool,
}

impl Default for ShallowWaterGpuResources {
    fn default() -> Self {
        Self {
            nested: None,
            settled: None,
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
    nest_pipelines: Res<ShallowWaterNestPipelines>,
    shift_pipelines: Res<ShallowWaterShiftPipelines>,
    pipeline_cache: Res<PipelineCache>,
    mut gpu: ResMut<ShallowWaterGpuResources>,
) {
    let cells = UVec2::new(controller.cells_x, controller.cells_y);
    let view = |handle: &Handle<Image>| gpu_images.get(handle).map(|g| g.texture_view.clone());
    let textures = ShallowWaterTextures::from_images(&controller.images(), &gpu_images);
    let bed = view(&controller.bed_source);
    let source = controller.source.as_ref().map(view);
    let pressure_head = controller.pressure_head.as_ref().map(view);
    let border = controller.border_profile.as_ref().map(view);
    let (
        Some(textures),
        Some(bed),
        None | Some(Some(_)),
        None | Some(Some(_)),
        None | Some(Some(_)),
    ) = (textures, bed, &source, &pressure_head, &border)
    else {
        gpu.solver = None;
        gpu.nested = None;
        return;
    };
    let inputs = ShallowWaterInputs {
        bed: Some(bed),
        source: source.flatten(),
        pressure_head: pressure_head.flatten(),
        border: border.flatten(),
    };

    if !gpu
        .solver
        .as_ref()
        .is_some_and(|s| s.matches(cells, &textures, controller.particle_count))
    {
        gpu.nested = None;
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
    if gpu.solver.is_none() {
        return;
    }
    let targets = [
        &controller.bed_water,
        &controller.flow_x,
        &controller.flow_y,
    ]
    .map(|handle| gpu_images.get(handle).map(|g| g.texture.clone()));
    gpu.settled = match (&controller.settled, targets) {
        (Some(settled), [Some(b), Some(u), Some(w)]) if settled.fits(&controller) => {
            let sources = match gpu.settled.take() {
                Some(old) if std::sync::Arc::ptr_eq(&old.from, settled) => old.sources,
                _ => settled.upload(&render_device, &render_queue),
            };
            Some(SettledGpu {
                from: settled.clone(),
                sources,
                targets: [b, u, w],
            })
        }
        _ => None,
    };
    let Some(solver) = gpu.solver.as_mut() else {
        return;
    };
    if !solver.inputs_match(&inputs) {
        solver.set_inputs(&render_device, inputs);
    }

    let (mover_count, movers) =
        super::pack_movers(&controller.movers, controller.placement.as_ref(), cells);
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
            mover_count,
            movers,
        },
    );

    let wind_dir = lod
        .as_ref()
        .map(|lod| Vec2::from_angle(lod.slices.wind_direction))
        .unwrap_or(Vec2::X);
    let settings = controller.settings(timestamp.0.load(Ordering::Relaxed), wind_dir);
    solver.write_settings(&render_queue, &settings);

    let apply_init = controller.sim_apply_serial != gpu.last_applied_serial.load(Ordering::Relaxed);
    let gpu = &mut *gpu;
    let (Some(nested), Some(offset), Some(placement)) = (
        controller.nested.as_ref(),
        controller.nest_offset(),
        controller.placement,
    ) else {
        gpu.nested = None;
        return;
    };
    let solver = gpu.solver.as_ref().unwrap();
    let child_cells = UVec2::splat(nested.cells);
    let child_textures = ShallowWaterTextures::from_images(&nested.images, &gpu_images);
    let bed = match &nested.world_bed {
        Some(bed) => view(&bed.image).map(|view| ShallowWaterWorldBed {
            view,
            origin: bed.origin,
            texel_size: bed.texel_size,
        }),
        None => view(&controller.bed_source).map(|view| ShallowWaterWorldBed {
            view,
            origin: placement.center - 0.5 * cells.as_vec2() * placement.cell_size,
            texel_size: placement.cell_size,
        }),
    };
    let (Some(child_textures), Some(bed)) = (child_textures, bed) else {
        gpu.nested = None;
        return;
    };
    let descriptor = |offset: IVec2| ShallowWaterNestDescriptor {
        offset,
        band: nested.band,
        parent_cell_size: placement.cell_size,
    };
    let stale = gpu.nested.as_ref().is_none_or(|n| {
        !n.child.matches(child_cells, &child_textures, 0) || n.bed.view.id() != bed.view.id()
    });
    if stale {
        gpu.nested = None;
        let Some(child) = ShallowWaterSolver::new(
            &render_device,
            &pipelines,
            &pipeline_cache,
            ShallowWaterSolverDescriptor {
                cells: child_cells,
                textures: child_textures,
                inputs: ShallowWaterInputs::default(),
                particle_count: 0,
            },
        ) else {
            return;
        };
        let nest = ShallowWaterNest::new(
            &render_device,
            &nest_pipelines,
            &pipeline_cache,
            solver,
            &child,
            descriptor(offset),
        );
        let shift = ShallowWaterShift::new(
            &render_device,
            &shift_pipelines,
            &pipeline_cache,
            &child,
            &bed,
            Some(solver),
        );
        let (Some(nest), Some(shift)) = (nest, shift) else {
            return;
        };
        gpu.nested = Some(ShallowWaterNestedGpu {
            child,
            nest,
            shift,
            bed,
            applied: None,
            shifting: false,
        });
    }
    let n = gpu.nested.as_mut().unwrap();
    let rho = NEST_REFINEMENT as f32;
    let child_placement = controller
        .nest_center()
        .map(|center| super::ShallowWaterPlacement {
            center,
            cell_size: placement.cell_size / rho,
            ..placement
        });
    let (mover_count, movers) =
        super::pack_movers(&controller.movers, child_placement.as_ref(), child_cells);
    n.child.write_interaction(
        &render_queue,
        &GpuInteractionUniform {
            dt: controller.dt,
            mover_count,
            movers,
            ..default()
        },
    );
    n.child.write_settings(
        &render_queue,
        &super::ShallowWaterSettings {
            dt: settings.dt / rho,
            cell_size: settings.cell_size / rho,
            left_border: ShallowWaterBorder::Wall,
            right_border: ShallowWaterBorder::Wall,
            bottom_border: ShallowWaterBorder::Wall,
            top_border: ShallowWaterBorder::Wall,
            pml_width: 0,
            ocean_band_cells: 0.0,
            ocean_forcing: false,
            ..settings
        },
    );
    // A reset or the first frame refills the whole child from the parent. A move shifts what it
    // holds by three child cells per parent cell.
    let covered = (nested.cells / NEST_REFINEMENT) as i32;
    let shift = match n.applied {
        Some(old) if !apply_init => {
            let delta = offset - old;
            (delta != IVec2::ZERO).then(|| {
                if delta.abs().max_element() >= covered {
                    IVec2::splat(nested.cells as i32)
                } else {
                    delta * NEST_REFINEMENT as i32
                }
            })
        }
        _ => Some(IVec2::splat(nested.cells as i32)),
    };
    n.shifting = shift.is_some();
    if let Some(shift) = shift {
        let corner = placement.center - 0.5 * cells.as_vec2() * placement.cell_size;
        n.shift.write_params(
            &render_queue,
            &n.bed,
            &ShallowWaterShiftParams {
                shift,
                origin: corner + offset.as_vec2() * placement.cell_size,
                cell_size: placement.cell_size / rho,
                subsamples: 1,
                fill_level: controller.fill_level,
                parent_offset: Some(offset),
            },
        );
        n.nest.set_offset(&render_queue, descriptor(offset));
        n.applied = Some(offset);
    }
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
        }
    }
    // A settled state replaces the preset and the seed, before the nested grid refills.
    if apply_init
        && !defer_init
        && let Some(settled) = &gpu_res.settled
    {
        for (source, target) in settled.sources.iter().zip(&settled.targets) {
            enc.copy_texture_to_texture(
                source.as_image_copy(),
                target.as_image_copy(),
                source.size(),
            );
        }
    }
    // The nested grid moves or refills after the parent's reset, so it reads the parent's
    // fresh surface.
    let nested = gpu_res.nested.as_ref();
    if let Some(n) = nested.filter(|n| n.shifting && !defer_init) {
        n.shift.encode(enc);
    }
    if !controller.paused && !defer_init {
        let mut pass = enc.begin_compute_pass(&ComputePassDescriptor {
            label: Some("shallow_water_step"),
            timestamp_writes: None,
        });
        match nested {
            Some(n) => {
                n.child.encode_interact(&mut pass);
                n.nest.encode_frame(&mut pass, solver, &n.child);
                n.child.encode_reconstruct_velocity(&mut pass);
            }
            None => solver.encode_step(&mut pass),
        }
        solver.encode_reconstruct_velocity(&mut pass);
        solver.encode_particles(&mut pass);
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
            .before(crate::water::WaterLodGraphSystems::Blend)
            .before(bevy::core_pipeline::schedule::camera_driver)
            .in_set(bevy::render::renderer::RenderGraphSystems::Render),
    );
}
