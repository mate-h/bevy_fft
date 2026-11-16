//! A nested shallow water grid moving over world bathymetry. Run explicitly with a graphics
//! adapter.
//!
//! The parent is 96 cells of 1 m with its corner at the world origin, over a bumpy world bed. The
//! child, 96 cells of 1/3 m, moves one parent cell along +x every few frames, and the cells it
//! uncovers take the world bed and the parent's surface. Still water must stay still, and the
//! parent must keep its water while the child moves through a wave.
use bevy::{
    app::PluginsState,
    prelude::*,
    render::{
        Render, RenderApp, RenderSystems,
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        gpu_readback::{Readback, ReadbackComplete},
        render_asset::RenderAssets,
        render_resource::*,
        renderer::{RenderContext, RenderDevice, RenderGraph, RenderGraphSystems, RenderQueue},
        texture::GpuImage,
    },
    window::ExitCondition,
    winit::WinitPlugin,
};
use bevy_fft::shallow_water::*;
use std::{
    sync::{
        Arc,
        atomic::{AtomicU32, Ordering},
    },
    thread,
    time::{Duration, Instant},
};

const N: u32 = 96;
const G: f32 = 9.81;
const DT: f32 = 0.08;
const LEVEL: f32 = 1.2;
/// World bed texels per meter, and the texture's size.
const BED_RES: u32 = 2;
const BED_SIZE: u32 = N * BED_RES;
/// The child's first offset in the parent, along x and y.
const START: IVec2 = IVec2::new(16, 32);
/// Frames between the child's moves.
const MOVE_EVERY: u32 = 6;

#[derive(Clone, Copy)]
struct Scenario {
    /// Fill both grids from the world with the shift kernel instead of from the CPU.
    fill_on_gpu: bool,
    hump: bool,
    frames: u32,
}

#[derive(Resource, Clone, ExtractResource)]
#[extract_app(RenderApp)]
struct Grids {
    parent: ShallowWaterImages,
    child: ShallowWaterImages,
    scenario_fill_on_gpu: bool,
    frames: u32,
    frames_done: Arc<AtomicU32>,
}

#[derive(Resource, Default)]
struct Driver {
    solvers: Option<(ShallowWaterSolver, ShallowWaterSolver)>,
    bed: Option<ShallowWaterWorldBed>,
    shifts: Option<(ShallowWaterShift, ShallowWaterShift)>,
    nest: Option<ShallowWaterNest>,
    frame: u32,
    offset: IVec2,
    /// Which grids shift this frame.
    shifting: (bool, bool),
}

#[derive(Resource, Default)]
struct Captures {
    parent: Vec<u8>,
    child: Vec<u8>,
    serial: u32,
}

/// World bed texel `(i, j)`: a gentle slope with bumps.
fn bed_texel(i: u32, j: u32) -> f32 {
    let (x, y) = (
        (i as f32 + 0.5) / BED_RES as f32,
        (j as f32 + 0.5) / BED_RES as f32,
    );
    0.2 + 0.004 * x + 0.25 * (x / 6.0).sin() * (y / 8.0).cos()
}

/// The world bed at `p` in meters, bilinear between texels as the kernel samples it.
fn world_bed(p: Vec2) -> f32 {
    let f = p * BED_RES as f32 - 0.5;
    let i = f.floor();
    let t = f - i;
    let at = |di: f32, dj: f32| {
        let c = (i + Vec2::new(di, dj)).clamp(Vec2::ZERO, Vec2::splat((BED_SIZE - 1) as f32));
        bed_texel(c.x as u32, c.y as u32)
    };
    let a = at(0.0, 0.0) + (at(1.0, 0.0) - at(0.0, 0.0)) * t.x;
    let b = at(0.0, 1.0) + (at(1.0, 1.0) - at(0.0, 1.0)) * t.x;
    a + (b - a) * t.y
}

fn hump(p: Vec2) -> f32 {
    0.05 * (-(p - Vec2::new(30.0, 48.0)).length_squared() / 18.0).exp()
}

fn settings(cell_size: f32, dt: f32) -> ShallowWaterSettings {
    ShallowWaterSettings {
        dt,
        cell_size,
        gravity: G,
        overshoot_limiter: false,
        ..default()
    }
}

fn nest_descriptor(offset: IVec2) -> ShallowWaterNestDescriptor {
    ShallowWaterNestDescriptor {
        offset,
        band: 1,
        parent_cell_size: 1.0,
    }
}

#[allow(clippy::too_many_arguments)]
fn prepare(
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    images: Res<RenderAssets<GpuImage>>,
    pipelines: Res<ShallowWaterPipelines>,
    nest_pipelines: Res<ShallowWaterNestPipelines>,
    shift_pipelines: Res<ShallowWaterShiftPipelines>,
    cache: Res<PipelineCache>,
    grids: Option<Res<Grids>>,
    mut driver: ResMut<Driver>,
) {
    let Some(grids) = grids else {
        return;
    };
    let driver = &mut *driver;
    if driver.solvers.is_none() {
        let build = |images_: &ShallowWaterImages| {
            ShallowWaterSolver::new(
                &device,
                &pipelines,
                &cache,
                ShallowWaterSolverDescriptor {
                    cells: UVec2::splat(N),
                    textures: ShallowWaterTextures::from_images(images_, &images)?,
                    inputs: ShallowWaterInputs::default(),
                    particle_count: 0,
                },
            )
        };
        let (Some(parent), Some(child)) = (build(&grids.parent), build(&grids.child)) else {
            return;
        };
        parent.write_settings(&queue, &settings(1.0, DT));
        child.write_settings(&queue, &settings(1.0 / 3.0, DT / 3.0));
        let texels: Vec<f32> = (0..BED_SIZE * BED_SIZE)
            .map(|k| bed_texel(k % BED_SIZE, k / BED_SIZE))
            .collect();
        let texture = device.create_texture_with_data(
            &queue,
            &TextureDescriptor {
                label: Some("test_world_bed"),
                size: Extent3d {
                    width: BED_SIZE,
                    height: BED_SIZE,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::R32Float,
                usage: TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
            Default::default(),
            bytemuck::cast_slice(&texels),
        );
        driver.bed = Some(ShallowWaterWorldBed {
            view: texture.create_view(&TextureViewDescriptor::default()),
            origin: Vec2::ZERO,
            texel_size: 1.0 / BED_RES as f32,
        });
        driver.solvers = Some((parent, child));
        driver.offset = START;
    }
    let (parent, child) = driver.solvers.as_ref().unwrap();
    let bed = driver.bed.as_ref().unwrap();
    if driver.nest.is_none() {
        let nest = ShallowWaterNest::new(
            &device,
            &nest_pipelines,
            &cache,
            parent,
            child,
            nest_descriptor(START),
        );
        let shifts =
            ShallowWaterShift::new(&device, &shift_pipelines, &cache, parent, bed, None).zip(
                ShallowWaterShift::new(&device, &shift_pipelines, &cache, child, bed, Some(parent)),
            );
        let (Some(nest), Some(shifts)) = (nest, shifts) else {
            return;
        };
        driver.nest = Some(nest);
        driver.shifts = Some(shifts);
    }

    // The first frame fills both grids from the world when the scenario asks. Later frames move
    // the child every `MOVE_EVERY` frames.
    let frame = driver.frame;
    let full = IVec2::splat(N as i32);
    let child_origin = |offset: IVec2| offset.as_vec2();
    let (parent_shift, child_shift) = driver.shifts.as_ref().unwrap();
    let mut shifting = (false, false);
    let mut offset = driver.offset;
    if frame == 0 && grids.scenario_fill_on_gpu {
        parent_shift.write_params(
            &queue,
            bed,
            &ShallowWaterShiftParams {
                shift: full,
                origin: Vec2::ZERO,
                cell_size: 1.0,
                subsamples: 3,
                fill_level: LEVEL,
                parent_offset: None,
            },
        );
        child_shift.write_params(
            &queue,
            bed,
            &ShallowWaterShiftParams {
                shift: full,
                origin: child_origin(offset),
                cell_size: 1.0 / 3.0,
                subsamples: 1,
                fill_level: LEVEL,
                parent_offset: Some(offset),
            },
        );
        shifting = (true, true);
    } else if frame > 0 && frame.is_multiple_of(MOVE_EVERY) {
        offset += IVec2::X;
        child_shift.write_params(
            &queue,
            bed,
            &ShallowWaterShiftParams {
                shift: IVec2::new(NEST_REFINEMENT as i32, 0),
                origin: child_origin(offset),
                cell_size: 1.0 / 3.0,
                subsamples: 1,
                fill_level: LEVEL,
                parent_offset: Some(offset),
            },
        );
        driver
            .nest
            .as_ref()
            .unwrap()
            .set_offset(&queue, nest_descriptor(offset));
        shifting = (false, true);
    }
    driver.offset = offset;
    driver.shifting = shifting;
}

fn run(mut ctx: RenderContext, mut driver: ResMut<Driver>, grids: Option<Res<Grids>>) {
    let Some(grids) = grids else {
        return;
    };
    if driver.frame >= grids.frames {
        return;
    }
    let (Some((parent, child)), Some(nest), Some((parent_shift, child_shift))) =
        (&driver.solvers, &driver.nest, &driver.shifts)
    else {
        return;
    };
    let encoder = ctx.command_encoder();
    // The parent fills first, so the child's new cells read its surface.
    if driver.shifting.0 {
        parent_shift.encode(encoder);
    }
    if driver.shifting.1 {
        child_shift.encode(encoder);
    }
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("shallow_water_nest_moving_test"),
            timestamp_writes: None,
        });
        nest.encode_frame(&mut pass, parent, child);
    }
    driver.frame += 1;
    grids.frames_done.store(driver.frame, Ordering::Relaxed);
}

fn step(app: &mut App) {
    app.update();
    assert!(app.should_exit().is_none(), "renderer must remain healthy");
    thread::sleep(Duration::from_millis(1));
}

/// Fills a `bed_water` image the way the shift kernel would: the mean world bed over
/// `subsamples`² points per cell, under `LEVEL` plus the hump.
fn fill(image: &mut Image, origin: Vec2, cell: f32, subsamples: u32, with_hump: bool) {
    let data = image.data.as_mut().unwrap();
    for j in 0..N {
        for i in 0..N {
            let corner = origin + Vec2::new(i as f32, j as f32) * cell;
            let mut bed = 0.0;
            for k in 0..subsamples * subsamples {
                let sub = (Vec2::new((k % subsamples) as f32, (k / subsamples) as f32) + 0.5)
                    / subsamples as f32;
                bed += world_bed(corner + sub * cell);
            }
            bed /= (subsamples * subsamples) as f32;
            let centre = corner + 0.5 * cell;
            let surface = LEVEL + if with_hump { hump(centre) } else { 0.0 };
            let o = ((j * N + i) * 16) as usize;
            data[o..o + 4].copy_from_slice(&bed.to_le_bytes());
            data[o + 4..o + 8].copy_from_slice(&(surface - bed).max(0.0).to_le_bytes());
        }
    }
}

/// Bed and depth per cell of the parent and the child after the scenario's frames.
fn simulate(scenario: Scenario) -> (Vec<Vec2>, Vec<Vec2>) {
    let mut app = App::new();
    app.add_plugins(
        DefaultPlugins
            .build()
            .disable::<WinitPlugin>()
            .disable::<bevy::render::pipelined_rendering::PipelinedRenderingPlugin>()
            .set(WindowPlugin {
                primary_window: None,
                exit_condition: ExitCondition::DontExit,
                ..default()
            }),
    )
    .add_plugins(ShallowWaterSolverPlugin)
    .add_plugins(ExtractResourcePlugin::<Grids>::default())
    .insert_resource(bevy::render::error_handler::RenderErrorHandler(
        |error, _, _| {
            panic!("GPU validation failed: {error:?}");
        },
    ))
    .init_resource::<Captures>();
    while app.plugins_state() == PluginsState::Adding {
        bevy::tasks::tick_global_task_pools_on_main_thread();
        thread::sleep(Duration::from_millis(1));
    }
    app.finish();
    app.cleanup();
    app.sub_app_mut(RenderApp)
        .init_resource::<Driver>()
        .add_systems(Render, prepare.in_set(RenderSystems::PrepareBindGroups))
        .add_systems(RenderGraph, run.in_set(RenderGraphSystems::Render));

    let frames_done = Arc::new(AtomicU32::new(0));
    let (parent, child) = {
        let mut assets = app.world_mut().resource_mut::<Assets<Image>>();
        let parent = ShallowWaterImages::new(&mut assets, N, N);
        let child = ShallowWaterImages::new(&mut assets, N, N);
        if !scenario.fill_on_gpu {
            fill(
                &mut assets.get_mut(&parent.bed_water).unwrap(),
                Vec2::ZERO,
                1.0,
                3,
                scenario.hump,
            );
            fill(
                &mut assets.get_mut(&child.bed_water).unwrap(),
                START.as_vec2(),
                1.0 / 3.0,
                1,
                scenario.hump,
            );
        }
        (parent, child)
    };
    app.insert_resource(Grids {
        parent: parent.clone(),
        child: child.clone(),
        scenario_fill_on_gpu: scenario.fill_on_gpu,
        frames: scenario.frames,
        frames_done: frames_done.clone(),
    });
    let deadline = Instant::now() + Duration::from_secs(120);
    while frames_done.load(Ordering::Relaxed) < scenario.frames {
        step(&mut app);
        assert!(
            Instant::now() < deadline,
            "the nest never finished its frames"
        );
    }
    app.world_mut()
        .spawn(Readback::texture(parent.bed_water.clone()))
        .observe(
            |event: On<ReadbackComplete>, mut captures: ResMut<Captures>| {
                captures.parent = event.data.clone();
                captures.serial |= 1;
            },
        );
    app.world_mut()
        .spawn(Readback::texture(child.bed_water.clone()))
        .observe(
            |event: On<ReadbackComplete>, mut captures: ResMut<Captures>| {
                captures.child = event.data.clone();
                captures.serial |= 2;
            },
        );
    while app.world().resource::<Captures>().serial != 3 {
        step(&mut app);
        assert!(Instant::now() < deadline, "readback timed out");
    }
    let decode = |bytes: &[u8]| -> Vec<Vec2> {
        let row = bytes.len() / N as usize;
        let lane = |o: usize| f32::from_le_bytes(bytes[o..o + 4].try_into().unwrap());
        (0..(N * N) as usize)
            .map(|i| {
                let o = (i / N as usize) * row + (i % N as usize) * 16;
                Vec2::new(lane(o), lane(o + 4))
            })
            .collect()
    };
    let captures = app.world().resource::<Captures>();
    (decode(&captures.parent), decode(&captures.child))
}

fn eta(c: Vec2) -> f32 {
    c.x + c.y
}

/// The child's offset after `frames` frames.
fn offset_after(frames: u32) -> IVec2 {
    START + IVec2::X * ((frames - 1) / MOVE_EVERY) as i32
}

#[test]
#[ignore = "requires a graphics adapter"]
fn still_water_stays_still_while_a_nest_moves_over_a_bumpy_bed() {
    let frames = 121;
    let (parent, child) = simulate(Scenario {
        fill_on_gpu: true,
        hump: false,
        frames,
    });
    let worst = parent
        .iter()
        .chain(&child)
        .map(|c| (eta(*c) - LEVEL).abs())
        .fold(0.0f32, f32::max);
    assert!(worst < 1e-5, "the surface moved {worst} m");
    // The child's bed is the world's under its cells where it ended up.
    let offset = offset_after(frames).as_vec2();
    for (k, c) in child.iter().enumerate() {
        let p = offset + (Vec2::new((k as u32 % N) as f32, (k as u32 / N) as f32) + 0.5) / 3.0;
        assert!(
            (c.x - world_bed(p)).abs() < 1e-4,
            "child cell {k} has bed {} for {}",
            c.x,
            world_bed(p)
        );
    }
}

#[test]
#[ignore = "requires a graphics adapter"]
fn a_moving_nest_keeps_the_parents_water() {
    let frames = 121;
    let (parent, child) = simulate(Scenario {
        fill_on_gpu: false,
        hump: true,
        frames,
    });
    for c in parent.iter().chain(&child) {
        assert!(c.is_finite() && c.y >= 0.0, "a cell holds {c}");
    }
    let mut start = 0.0f64;
    for j in 0..N {
        for i in 0..N {
            let corner = Vec2::new(i as f32, j as f32);
            let mut bed = 0.0;
            for k in 0..9 {
                bed += world_bed(corner + (Vec2::new((k % 3) as f32, (k / 3) as f32) + 0.5) / 3.0);
            }
            start += (LEVEL + hump(corner + 0.5) - bed / 9.0).max(0.0) as f64;
        }
    }
    let volume: f64 = parent.iter().map(|c| c.y as f64).sum();
    let drift = (volume - start) / start;
    assert!(drift.abs() < 1e-7, "the parent's volume drifted by {drift}");
    let crest = child
        .iter()
        .map(|c| (eta(*c) - LEVEL).abs())
        .fold(0.0f32, f32::max);
    assert!(crest > 0.002, "the wave never reached the child: {crest} m");
}
