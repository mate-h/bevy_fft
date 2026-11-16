//! A fine shallow water grid nested two ways in a coarse one. Run explicitly with a graphics
//! adapter.
//!
//! The parent is 96 cells of 1 m and the child 96 cells of 1/3 m over the parent's middle third.
//! A hump of water starts beside the child, runs through it and out the far side. The pair must
//! keep the parent's volume, the child must agree with the parent over its interior, and the
//! parent outside the child must look like a parent run alone. Still water over a sloped bed
//! must stay still.
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
const OFFSET: i32 = 32;
const G: f32 = 9.81;
const DT: f32 = 0.08;

#[derive(Clone, Copy)]
struct Scenario {
    nested: bool,
    /// Bed and depth at a point in parent cell units.
    state: fn(Vec2) -> Vec2,
    frames: u32,
}

#[derive(Resource, Clone, ExtractResource)]
#[extract_app(RenderApp)]
struct Grids {
    parent: ShallowWaterImages,
    child: ShallowWaterImages,
    nested: bool,
    frames: u32,
    frames_done: Arc<AtomicU32>,
}

#[derive(Resource, Default)]
struct Driver {
    solvers: Option<(ShallowWaterSolver, ShallowWaterSolver)>,
    nest: Option<ShallowWaterNest>,
    frame: u32,
}

#[derive(Resource, Default)]
struct Captures {
    parent: Vec<u8>,
    child: Vec<u8>,
    serial: u32,
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

#[allow(clippy::too_many_arguments)]
fn prepare(
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    images: Res<RenderAssets<GpuImage>>,
    pipelines: Res<ShallowWaterPipelines>,
    nest_pipelines: Res<ShallowWaterNestPipelines>,
    cache: Res<PipelineCache>,
    grids: Option<Res<Grids>>,
    mut driver: ResMut<Driver>,
) {
    let Some(grids) = grids else {
        return;
    };
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
        driver.solvers = Some((parent, child));
    }
    if grids.nested && driver.nest.is_none() {
        let (parent, child) = driver.solvers.as_ref().unwrap();
        driver.nest = ShallowWaterNest::new(
            &device,
            &nest_pipelines,
            &cache,
            parent,
            child,
            ShallowWaterNestDescriptor {
                offset: IVec2::splat(OFFSET),
                band: 1,
                parent_cell_size: 1.0,
            },
        );
    }
}

fn run(mut ctx: RenderContext, mut driver: ResMut<Driver>, grids: Option<Res<Grids>>) {
    let Some(grids) = grids else {
        return;
    };
    if driver.frame >= grids.frames {
        return;
    }
    let Some((parent, child)) = driver.solvers.as_ref() else {
        return;
    };
    if grids.nested && driver.nest.is_none() {
        return;
    }
    {
        let mut pass = ctx
            .command_encoder()
            .begin_compute_pass(&ComputePassDescriptor {
                label: Some("shallow_water_nest_test"),
                timestamp_writes: None,
            });
        match &driver.nest {
            Some(nest) => nest.encode_frame(&mut pass, parent, child),
            None => parent.encode_step(&mut pass),
        }
    }
    driver.frame += 1;
    grids.frames_done.store(driver.frame, Ordering::Relaxed);
}

fn step(app: &mut App) {
    app.update();
    assert!(app.should_exit().is_none(), "renderer must remain healthy");
    thread::sleep(Duration::from_millis(1));
}

/// A grid's `bed_water` image filled from `state`, sampled at cell centres. `scale` is the
/// grid's cell width in parent cells and `origin` its corner in parent cells.
fn fill(image: &mut Image, state: fn(Vec2) -> Vec2, origin: Vec2, scale: f32) {
    let data = image.data.as_mut().unwrap();
    for j in 0..N {
        for i in 0..N {
            let p = origin + (Vec2::new(i as f32, j as f32) + 0.5) * scale;
            let s = state(p);
            let o = ((j * N + i) * 16) as usize;
            data[o..o + 4].copy_from_slice(&s.x.to_le_bytes());
            data[o + 4..o + 8].copy_from_slice(&s.y.to_le_bytes());
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
        fill(
            &mut assets.get_mut(&parent.bed_water).unwrap(),
            scenario.state,
            Vec2::ZERO,
            1.0,
        );
        fill(
            &mut assets.get_mut(&child.bed_water).unwrap(),
            scenario.state,
            Vec2::splat(OFFSET as f32),
            1.0 / 3.0,
        );
        (parent, child)
    };
    app.insert_resource(Grids {
        parent: parent.clone(),
        child: child.clone(),
        nested: scenario.nested,
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

/// A flat bed under 1 m of water with a 5 cm hump centred at x = 20, y = 48.
fn hump(p: Vec2) -> Vec2 {
    let r2 = (p - Vec2::new(20.0, 48.0)).length_squared();
    Vec2::new(0.0, 1.0 + 0.05 * (-r2 / 18.0).exp())
}

/// A bed rising along x and y, under still water at 1.2 m.
fn slope(p: Vec2) -> Vec2 {
    let bed = 0.008 * p.x + 0.002 * p.y;
    Vec2::new(bed, 1.2 - bed)
}

fn eta(c: Vec2) -> f32 {
    c.x + c.y
}

#[test]
#[ignore = "requires a graphics adapter"]
fn a_wave_crosses_a_nested_grid_and_the_pair_keeps_its_water() {
    let frames = 90;
    let (alone, _) = simulate(Scenario {
        nested: false,
        state: hump,
        frames,
    });
    let (parent, child) = simulate(Scenario {
        nested: true,
        state: hump,
        frames,
    });
    for c in parent.iter().chain(&child) {
        assert!(c.is_finite() && c.y >= 0.0, "a cell holds {c}");
    }
    let volume = |cells: &[Vec2]| cells.iter().map(|c| c.y as f64).sum::<f64>();
    let start: f64 = (0..N * N)
        .map(|i| hump(Vec2::new((i % N) as f32 + 0.5, (i / N) as f32 + 0.5)).y as f64)
        .sum();
    // Without the reflux the parent loses about 1e-6 of its water over this run.
    let drift = (volume(&parent) - start) / start;
    let crest = child
        .iter()
        .map(|c| (eta(*c) - 1.0).abs())
        .fold(0.0f32, f32::max);
    assert!(crest > 0.005, "the wave never reached the child: {crest} m");
    assert!(drift.abs() < 1e-7, "the parent's volume drifted by {drift}");

    // The child agrees with the parent over its interior.
    let mut worst = 0.0f32;
    for pj in OFFSET + 1..OFFSET + 31 {
        for pi in OFFSET + 1..OFFSET + 31 {
            let mut mean = 0.0;
            for k in 0..9 {
                let (ci, cj) = (3 * (pi - OFFSET) + k % 3, 3 * (pj - OFFSET) + k / 3);
                mean += eta(child[(cj * N as i32 + ci) as usize]) / 9.0;
            }
            worst = worst.max((mean - eta(parent[(pj * N as i32 + pi) as usize])).abs());
        }
    }
    assert!(worst < 1e-5, "child and parent differ by {worst} m");

    // Outside the child, the nested parent looks like the parent alone.
    let (mut diff, mut wave) = (0.0f32, 0.0f32);
    for (i, (a, b)) in alone.iter().zip(&parent).enumerate() {
        let (x, y) = ((i as u32 % N) as i32, (i as u32 / N) as i32);
        let inside =
            (OFFSET - 2..OFFSET + 34).contains(&x) && (OFFSET - 2..OFFSET + 34).contains(&y);
        if !inside {
            diff += (eta(*a) - eta(*b)).powi(2);
            wave += (eta(*a) - 1.0).powi(2);
        }
    }
    let ratio = (diff / wave).sqrt();
    assert!(
        ratio < 0.05,
        "the nest changed the wave outside it by {ratio}"
    );
}

#[test]
#[ignore = "requires a graphics adapter"]
fn still_water_over_a_slope_stays_still_in_a_nest() {
    let (parent, child) = simulate(Scenario {
        nested: true,
        state: slope,
        frames: 60,
    });
    let worst = parent
        .iter()
        .chain(&child)
        .map(|c| (eta(*c) - 1.2).abs())
        .fold(0.0f32, f32::max);
    assert!(worst < 1e-4, "the surface moved {worst} m");
}
