//! Dam breaks on a flat channel, against their exact solutions: Stoker's bore over a wet bed and
//! Ritter's front over a dry one, the moving shocks a river makes when a jam gives way or a flood
//! runs down a dry bed. Run explicitly with a graphics adapter.
//!
//! The grid is a 120 m channel of 0.25 m cells, four cells wide, with walls on every edge and a
//! dam at 30 m. The water starts at rest, 1 m deep upstream. The test steps 6 s, reads the depth
//! back and finds where the bore or the front has reached.
//!
//! The solver advects velocity in non-conservative form (MacCormack, CMF10), which moves a bore
//! at the wrong speed (docs/river_solver_spike.md). On 2026-11-20 the GPU ran Stoker's bore 24%
//! short of its exact distance, with the plateau behind it 0.331 m deep against 0.396 m, and the
//! dry front to 35% of Ritter's frictionless distance, near the 41% the same formulation reaches
//! without friction. These tests record that error and fail if it grows. Bead bevy_fft-zog.1.3,
//! momentum-conserving advection, should bring the bore within a cell.
use bevy::{
    app::PluginsState,
    prelude::*,
    render::{
        Render, RenderApp, RenderSystems,
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        gpu_readback::{Readback, ReadbackComplete},
        render_resource::*,
        renderer::{RenderContext, RenderDevice, RenderGraph, RenderGraphSystems, RenderQueue},
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

const NX: u32 = 480;
const NY: u32 = 4;
const DX: f32 = 0.25;
const G: f32 = 9.81;
const DT: f32 = 0.005;
const STEPS_PER_FRAME: u32 = 40;
const FRAMES: u32 = 30;
/// The dam, in meters from the left wall.
const DAM: f32 = 30.0;
const H_LEFT: f32 = 1.0;

#[derive(Clone, Copy)]
struct Case {
    /// Depth downstream of the dam, zero for a dry bed.
    h_right: f32,
    manning_n: f32,
}

#[derive(Resource, Clone, ExtractResource)]
#[extract_app(RenderApp)]
struct Grid {
    images: ShallowWaterImages,
    manning_n: f32,
    frames_done: Arc<AtomicU32>,
}

#[derive(Resource, Default)]
struct Capture {
    bytes: Vec<u8>,
    serial: u64,
}

#[derive(Resource, Default)]
struct Driver {
    solver: Option<ShallowWaterSolver>,
    frame: u32,
}

fn prepare_driver(
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    images: Res<bevy::render::render_asset::RenderAssets<bevy::render::texture::GpuImage>>,
    pipelines: Res<ShallowWaterPipelines>,
    cache: Res<PipelineCache>,
    grid: Option<Res<Grid>>,
    mut driver: ResMut<Driver>,
) {
    let Some(grid) = grid else {
        return;
    };
    if driver.solver.is_some() {
        return;
    }
    let Some(textures) = ShallowWaterTextures::from_images(&grid.images, &images) else {
        return;
    };
    let Some(solver) = ShallowWaterSolver::new(
        &device,
        &pipelines,
        &cache,
        ShallowWaterSolverDescriptor {
            cells: UVec2::new(NX, NY),
            textures,
            inputs: ShallowWaterInputs::default(),
            particle_count: 0,
        },
    ) else {
        return;
    };
    solver.write_settings(
        &queue,
        &ShallowWaterSettings {
            dt: DT,
            cell_size: DX,
            gravity: G,
            manning_n: grid.manning_n,
            overshoot_limiter: false,
            ..default()
        },
    );
    driver.solver = Some(solver);
}

fn run_driver(mut ctx: RenderContext, mut driver: ResMut<Driver>, grid: Option<Res<Grid>>) {
    let Some(grid) = grid else {
        return;
    };
    if driver.frame >= FRAMES {
        return;
    }
    let Some(solver) = driver.solver.as_ref() else {
        return;
    };
    {
        let mut pass = ctx
            .command_encoder()
            .begin_compute_pass(&ComputePassDescriptor {
                label: Some("shallow_water_dam_break_test"),
                timestamp_writes: None,
            });
        for _ in 0..STEPS_PER_FRAME {
            solver.encode_step(&mut pass);
        }
    }
    driver.frame += 1;
    grid.frames_done.store(driver.frame, Ordering::Relaxed);
}

fn step(app: &mut App) {
    app.update();
    assert!(app.should_exit().is_none(), "renderer must remain healthy");
    thread::sleep(Duration::from_millis(2));
}

/// Depth along the channel's centre after `FRAMES × STEPS_PER_FRAME` steps, and the volume of
/// the whole grid.
fn run(case: Case) -> (Vec<f32>, f32) {
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
    .add_plugins(ExtractResourcePlugin::<Grid>::default())
    .insert_resource(bevy::render::error_handler::RenderErrorHandler(
        |error, _, _| {
            panic!("GPU validation failed: {error:?}");
        },
    ))
    .init_resource::<Capture>();
    while app.plugins_state() == PluginsState::Adding {
        bevy::tasks::tick_global_task_pools_on_main_thread();
        thread::sleep(Duration::from_millis(1));
    }
    app.finish();
    app.cleanup();
    let render_app = app.sub_app_mut(RenderApp);
    render_app
        .init_resource::<Driver>()
        .add_systems(
            Render,
            prepare_driver.in_set(RenderSystems::PrepareBindGroups),
        )
        .add_systems(RenderGraph, run_driver.in_set(RenderGraphSystems::Render));

    let frames_done = Arc::new(AtomicU32::new(0));
    let images = {
        let mut assets = app.world_mut().resource_mut::<Assets<Image>>();
        let images = ShallowWaterImages::new(&mut assets, NX, NY);
        // A flat bed at zero with the dam's two depths, uploaded with the image.
        let mut bed_water = assets.get_mut(&images.bed_water).unwrap();
        let texels: Vec<[f32; 4]> = (0..NX * NY)
            .map(|i| {
                let x = ((i % NX) as f32 + 0.5) * DX;
                let h = if x < DAM { H_LEFT } else { case.h_right };
                [0.0, h, 0.0, 0.0]
            })
            .collect();
        bed_water.data = Some(bytemuck::cast_slice(&texels).to_vec());
        images
    };
    app.insert_resource(Grid {
        images,
        manning_n: case.manning_n,
        frames_done: frames_done.clone(),
    });

    let deadline = Instant::now() + Duration::from_secs(60);
    while frames_done.load(Ordering::Relaxed) < FRAMES {
        step(&mut app);
        assert!(
            Instant::now() < deadline,
            "solver never finished its frames"
        );
    }
    let bed_water = app.world().resource::<Grid>().images.bed_water.clone();
    app.world_mut().spawn(Readback::texture(bed_water)).observe(
        |event: On<ReadbackComplete>, mut capture: ResMut<Capture>| {
            capture.bytes = event.data.clone();
            capture.serial += 1;
        },
    );
    while app.world().resource::<Capture>().serial == 0 {
        step(&mut app);
        assert!(Instant::now() < deadline, "bed_water readback timed out");
    }
    let bytes = &app.world().resource::<Capture>().bytes;
    let row = bytes.len() / NY as usize;
    let depth = |x: usize, y: usize| {
        let o = y * row + x * 16 + 4;
        f32::from_le_bytes(bytes[o..o + 4].try_into().unwrap())
    };
    let centre: Vec<f32> = (0..NX as usize)
        .map(|x| depth(x, NY as usize / 2))
        .collect();
    let volume = (0..NY as usize)
        .flat_map(|y| (0..NX as usize).map(move |x| (x, y)))
        .map(|(x, y)| depth(x, y))
        .sum::<f32>()
        * DX
        * DX;
    (centre, volume)
}

fn elapsed() -> f32 {
    DT * (FRAMES * STEPS_PER_FRAME) as f32
}

fn initial_volume(h_right: f32) -> f32 {
    let length = NX as f32 * DX;
    (DAM * H_LEFT + (length - DAM) * h_right) * NY as f32 * DX
}

/// Stoker's solution for a dam break onto still water `h_right` deep: the depth and speed between
/// the rarefaction and the bore, and the bore's speed.
fn stoker(h_left: f32, h_right: f32) -> (f32, f32, f32) {
    let (g, hl, hr) = (G as f64, h_left as f64, h_right as f64);
    // The rarefaction's speed behind the bore must match the bore's speed of the water it moves.
    let mismatch = |hm: f64| {
        let rarefaction = 2.0 * ((g * hl).sqrt() - (g * hm).sqrt());
        let bore = (hm - hr) * (g * (hm + hr) / (2.0 * hm * hr)).sqrt();
        rarefaction - bore
    };
    let (mut lo, mut hi) = (hr, hl);
    for _ in 0..100 {
        let mid = 0.5 * (lo + hi);
        if mismatch(mid) > 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let hm = 0.5 * (lo + hi);
    let um = 2.0 * ((g * hl).sqrt() - (g * hm).sqrt());
    let speed = hm * um / (hm - hr);
    (hm as f32, um as f32, speed as f32)
}

/// The furthest cell centre downstream of the dam where the depth exceeds `level`.
fn reach(depth: &[f32], level: f32) -> f32 {
    depth
        .iter()
        .rposition(|&h| h > level)
        .map(|i| (i as f32 + 0.5) * DX)
        .unwrap_or(0.0)
}

/// Stoker: 1 m upstream, 0.1 m downstream, no friction. The bore's exact position after 6 s is
/// about 63 m from the left wall.
#[test]
#[ignore = "requires a graphics adapter"]
fn a_dam_break_bore_runs_close_to_stokers_speed() {
    let h_right = 0.1;
    let (depth, volume) = run(Case {
        h_right,
        manning_n: 0.0,
    });
    assert!(
        depth.iter().all(|h| h.is_finite() && *h >= 0.0),
        "depth went negative or non-finite"
    );
    let start = initial_volume(h_right);
    assert!(
        (volume - start).abs() < 1e-4 * start,
        "volume {volume} m³ against {start} m³ at the start"
    );
    let (hm, _, speed) = stoker(H_LEFT, h_right);
    let exact = DAM + speed * elapsed();
    let found = reach(&depth, 0.5 * (hm + h_right));
    let error = (found - DAM) / (exact - DAM) - 1.0;
    // The plateau between the rarefaction and the bore, over its middle half.
    let plateau_from = DAM + 0.6 * (exact - DAM);
    let plateau = &depth[(plateau_from / DX) as usize..((exact - 2.0) / DX) as usize];
    let plateau_depth = plateau.iter().sum::<f32>() / plateau.len().max(1) as f32;
    println!(
        "Stoker bore at {found:.2} m against {exact:.2} m ({:+.1}% of its run), plateau \
         {plateau_depth:.3} m against {hm:.3} m",
        100.0 * error
    );
    // Recorded 2026-11-20 with MacCormack advection. Tighten once momentum-conserving
    // advection lands.
    assert!(
        error.abs() < 0.3,
        "the bore ran {:+.1}% of its exact distance",
        100.0 * error
    );
}

/// Ritter: 1 m upstream onto a dry bed, here with Manning friction n = 0.03 as on a river bed.
/// Friction holds the front back from Ritter's frictionless 2√(g h) t, and the non-conservative
/// advection holds it back further, so this checks the front lands between 30% of that and all of
/// it, and that no water appears or vanishes.
#[test]
#[ignore = "requires a graphics adapter"]
fn a_dam_break_onto_a_dry_bed_advances_its_front() {
    let (depth, volume) = run(Case {
        h_right: 0.0,
        manning_n: 0.03,
    });
    assert!(
        depth.iter().all(|h| h.is_finite() && *h >= 0.0),
        "depth went negative or non-finite"
    );
    let start = initial_volume(0.0);
    assert!(
        (volume - start).abs() < 1e-4 * start,
        "volume {volume} m³ against {start} m³ at the start"
    );
    let ritter = DAM + 2.0 * (G * H_LEFT).sqrt() * elapsed();
    let found = reach(&depth, 0.005);
    let share = (found - DAM) / (ritter - DAM);
    println!(
        "dry front at {found:.2} m, {:.0}% of Ritter's frictionless run to {ritter:.2} m",
        100.0 * share
    );
    assert!(
        (0.3..1.0).contains(&share),
        "the front ran {:.0}% of Ritter's distance",
        100.0 * share
    );
}
