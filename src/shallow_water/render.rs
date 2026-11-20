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
    GpuInteractionUniform, ShallowWaterBorder, ShallowWaterController, ShallowWaterInputs,
    ShallowWaterNest, ShallowWaterNestDescriptor, ShallowWaterNestPipelines, ShallowWaterPipelines,
    ShallowWaterShift, ShallowWaterShiftParams, ShallowWaterShiftPipelines, ShallowWaterSolver,
    ShallowWaterSolverDescriptor, ShallowWaterTextures, ShallowWaterWorldBed,
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
    /// The controller's nested grids in order, each in the one before it, built after the
    /// solver.
    pub nested: Vec<ShallowWaterNestedGpu>,
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
    /// Couples the grid to its parent: the controller's grid or the nested grid before it.
    nest: ShallowWaterNest,
    shift: ShallowWaterShift,
    bed: ShallowWaterWorldBed,
    /// The steps per parent step the nest was built with.
    steps: u32,
    /// [`ShallowWaterNestLevel::corner_cells`](super::ShallowWaterNestLevel::corner_cells) once
    /// the grid holds water.
    corner: Option<IVec2>,
    /// The parent cell under the grid's first cell that the nest has.
    offset: IVec2,
    /// Whether the grid shifts this frame, with the parameters already written.
    shifting: bool,
}

impl Default for ShallowWaterGpuResources {
    fn default() -> Self {
        Self {
            nested: Vec::new(),
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
        gpu.nested.clear();
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
        gpu.nested.clear();
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
    let levels = controller.nest_levels();
    let Some(placement) = controller.placement.filter(|_| !levels.is_empty()) else {
        gpu.nested.clear();
        return;
    };
    let root_corner = placement.center - 0.5 * cells.as_vec2() * placement.cell_size;
    gpu.nested.truncate(levels.len());

    // Build the levels whose images, bed or step count changed, and every level inside one.
    for (k, (nested, level)) in controller.nested.iter().zip(&levels).enumerate() {
        let child_cells = UVec2::splat(nested.cells);
        let child_textures = ShallowWaterTextures::from_images(&nested.images, &gpu_images);
        // Without a world bed of its own, a grid interpolates its parent's bed.
        let bed = match (&nested.world_bed, k.checked_sub(1)) {
            (Some(bed), _) => view(&bed.image).map(|view| ShallowWaterWorldBed {
                view,
                origin: bed.origin,
                texel_size: bed.texel_size,
            }),
            (None, None) => view(&controller.bed_source).map(|view| ShallowWaterWorldBed {
                view,
                origin: root_corner,
                texel_size: placement.cell_size,
            }),
            (None, Some(p)) => gpu.nested.get(p).map(|parent| ShallowWaterWorldBed {
                view: parent.child.textures().bed_water.clone(),
                origin: levels[p].corner,
                texel_size: levels[p].cell_size,
            }),
        };
        let (Some(child_textures), Some(bed)) = (child_textures, bed) else {
            gpu.nested.truncate(k);
            break;
        };
        let stale = gpu.nested.get(k).is_none_or(|n| {
            !n.child.matches(child_cells, &child_textures, 0)
                || n.bed.view.id() != bed.view.id()
                || n.steps != level.steps
        });
        if !stale {
            gpu.nested[k].bed = bed;
            continue;
        }
        gpu.nested.truncate(k);
        let parent = match k.checked_sub(1) {
            Some(p) => &gpu.nested[p].child,
            None => gpu.solver.as_ref().unwrap(),
        };
        let parent_cell_size = k
            .checked_sub(1)
            .map_or(placement.cell_size, |p| levels[p].cell_size);
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
            break;
        };
        let nest = ShallowWaterNest::new(
            &render_device,
            &nest_pipelines,
            &pipeline_cache,
            parent,
            &child,
            ShallowWaterNestDescriptor {
                offset: level.offset,
                band: nested.band,
                parent_cell_size,
                child_steps: level.steps,
            },
        );
        let shift = ShallowWaterShift::new(
            &render_device,
            &shift_pipelines,
            &pipeline_cache,
            &child,
            &bed,
            Some(parent),
        );
        let (Some(nest), Some(shift)) = (nest, shift) else {
            break;
        };
        gpu.nested.push(ShallowWaterNestedGpu {
            child,
            nest,
            shift,
            bed,
            steps: level.steps,
            corner: None,
            offset: level.offset,
            shifting: false,
        });
    }

    // Write each level's inputs, settings and stirs, and move it to where it sits this frame.
    for (k, n) in gpu.nested.iter_mut().enumerate() {
        let (nested, level) = (&controller.nested[k], &levels[k]);
        let child_cells = UVec2::splat(nested.cells);
        let parent_cell_size = k
            .checked_sub(1)
            .map_or(placement.cell_size, |p| levels[p].cell_size);
        // The grid reads its own pressure head while the controller's grid has one.
        let child_inputs = ShallowWaterInputs {
            pressure_head: controller
                .pressure_head
                .as_ref()
                .and_then(|_| view(&nested.pressure_head)),
            ..default()
        };
        if !n.child.inputs_match(&child_inputs) {
            n.child.set_inputs(&render_device, child_inputs);
        }
        let level_placement = super::ShallowWaterPlacement {
            center: level.center,
            cell_size: level.cell_size,
            ..placement
        };
        let (mover_count, movers) =
            super::pack_movers(&controller.movers, Some(&level_placement), child_cells);
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
                dt: level.dt,
                cell_size: level.cell_size,
                left_border: ShallowWaterBorder::Wall,
                right_border: ShallowWaterBorder::Wall,
                bottom_border: ShallowWaterBorder::Wall,
                top_border: ShallowWaterBorder::Wall,
                pml_width: 0,
                ocean_band_cells: 0.0,
                ocean_forcing: false,
                momentum_conserving: level.momentum_conserving,
                ..settings
            },
        );
        // A reset or the first frame refills the whole grid from its parent. A move over the
        // world shifts what it holds by three cells per parent cell, after its parent moved.
        let full = IVec2::splat(nested.cells as i32);
        let shift = match n.corner {
            Some(old) if !apply_init => {
                let delta = level.corner_cells - old;
                (delta != IVec2::ZERO).then(|| {
                    if delta.abs().max_element() >= nested.cells as i32 {
                        full
                    } else {
                        delta
                    }
                })
            }
            _ => Some(full),
        };
        n.shifting = shift.is_some();
        if let Some(shift) = shift {
            n.shift.write_params(
                &render_queue,
                &n.bed,
                &ShallowWaterShiftParams {
                    shift,
                    origin: level.corner,
                    cell_size: level.cell_size,
                    subsamples: 1,
                    fill_level: controller.fill_level,
                    parent_offset: Some(level.offset),
                },
            );
            n.corner = Some(level.corner_cells);
        }
        // A grid in a nested parent sits at a new offset whenever its parent moves, even when it
        // stays put over the world.
        if n.shifting || n.offset != level.offset {
            n.nest.set_offset(
                &render_queue,
                ShallowWaterNestDescriptor {
                    offset: level.offset,
                    band: nested.band,
                    parent_cell_size,
                    child_steps: level.steps,
                },
            );
            n.offset = level.offset;
        }
    }
}

/// Encodes one frame of `parent` and the chain of nested grids in it: each grid steps through
/// its own nest's frame with the next grid in, the last with plain steps.
fn encode_nested_frame(
    pass: &mut ComputePass,
    parent: &ShallowWaterSolver,
    levels: &[ShallowWaterNestedGpu],
) {
    let Some((n, inner)) = levels.split_first() else {
        parent.encode_step(pass);
        return;
    };
    n.nest.encode_frame_with(pass, parent, &n.child, |pass| {
        if inner.is_empty() {
            n.child.encode_step_with_face_volumes(pass);
        } else {
            encode_nested_frame(pass, &n.child, inner);
        }
    });
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
    // The nested grids move or refill after the parent's reset, each after its own parent, so
    // they read their parent's fresh surface.
    let nested = &gpu_res.nested;
    if !defer_init {
        for n in nested.iter().filter(|n| n.shifting) {
            n.shift.encode(enc);
        }
    }
    if !controller.paused && !defer_init {
        let mut pass = enc.begin_compute_pass(&ComputePassDescriptor {
            label: Some("shallow_water_step"),
            timestamp_writes: None,
        });
        for n in nested {
            n.child.encode_interact(&mut pass);
        }
        encode_nested_frame(&mut pass, solver, nested);
        for n in nested {
            n.child.encode_reconstruct_velocity(&mut pass);
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
        .init_resource::<super::waterfalls::ShallowWaterWaterfallPipeline>()
        .init_resource::<super::waterfalls::ShallowWaterWaterfallGpu>()
        .add_systems(
            Render,
            (
                prepare_shallow_water_gpu,
                super::waterfalls::prepare_shallow_water_waterfalls,
            )
                .in_set(RenderSystems::PrepareBindGroups),
        );
    render_app.add_systems(
        bevy::render::renderer::RenderGraph,
        super::waterfalls::run_shallow_water_waterfalls
            .after(run_shallow_water_sim)
            .before(crate::water::WaterLodGraphSystems::Blend)
            .in_set(bevy::render::renderer::RenderGraphSystems::Render),
    );
    render_app.add_systems(
        bevy::render::renderer::RenderGraph,
        run_shallow_water_sim
            .before(crate::water::WaterLodGraphSystems::Blend)
            .before(bevy::core_pipeline::schedule::camera_driver)
            .in_set(bevy::render::renderer::RenderGraphSystems::Render),
    );
}
