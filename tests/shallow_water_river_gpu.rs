//! A river through a shallow water grid: inflow and outflow faces from a border profile, and
//! Manning friction on a sloped bed. Run explicitly with a graphics adapter.
use bevy::{
    app::PluginsState,
    prelude::*,
    render::gpu_readback::{Readback, ReadbackComplete},
    time::TimeUpdateStrategy,
    window::ExitCondition,
    winit::WinitPlugin,
};
use bevy_fft::shallow_water::*;
use std::{
    thread,
    time::{Duration, Instant},
};

const NX: u32 = 96;
const NY: u32 = 16;
/// Bed slope along +x.
const SLOPE: f32 = 0.002;
const MANNING_N: f32 = 0.03;
/// Discharge per meter of channel width, m²/s.
const Q: f32 = 0.25;
const G: f32 = 9.81;
const DT: f32 = 0.08;

#[derive(Resource, Default)]
struct Capture {
    bytes: Vec<u8>,
    serial: u64,
}

fn step(app: &mut App) {
    app.update();
    assert!(app.should_exit().is_none(), "renderer must remain healthy");
    thread::sleep(Duration::from_millis(1));
}

/// A dry channel `NX` cells long falling along +x, walled along its sides, fed over its whole
/// left edge and draining over its right edge at normal depth.
fn channel_app() -> App {
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
    .add_plugins(ShallowWaterPlugin)
    .insert_resource(TimeUpdateStrategy::ManualDuration(Duration::from_secs_f32(
        1.0 / 60.0,
    )))
    .insert_resource(bevy::render::error_handler::RenderErrorHandler(
        |error, _, _| {
            panic!("GPU validation failed: {error:?}");
        },
    ))
    .init_resource::<Capture>();
    let mut images = app.world_mut().resource_mut::<Assets<Image>>();
    let mut controller = ShallowWaterController::new(&mut images, NX, NY);
    let bed: Vec<f32> = (0..NX * NY)
        .map(|i| SLOPE * (NX as f32 - (i % NX) as f32 - 0.5))
        .collect();
    controller.bed_source = images.add(bed_source_image(NX, NY, &bed));
    controller.preset_index = SHALLOW_WATER_PRESET_BED_SOURCE;
    controller.fill_level = 0.0;
    controller.gravity = G;
    controller.dt = DT;
    controller.manning_n = MANNING_N;
    let mut profile = ShallowWaterBorderProfile::new(UVec2::new(NX, NY));
    for j in 0..NY {
        profile.set(
            GridEdge::Left,
            j,
            ShallowWaterBorderFace::Inflow {
                discharge_per_width: Q,
            },
        );
        profile.set(
            GridEdge::Right,
            j,
            ShallowWaterBorderFace::Outflow { slope: SLOPE },
        );
    }
    controller.border_profile = Some(images.add(profile.image()));
    app.insert_resource(controller);
    while app.plugins_state() == PluginsState::Adding {
        bevy::tasks::tick_global_task_pools_on_main_thread();
        thread::sleep(Duration::from_millis(1));
    }
    app.finish();
    app.cleanup();
    step(&mut app);
    let bed_water = app
        .world()
        .resource::<ShallowWaterController>()
        .bed_water
        .clone();
    app.world_mut().spawn(Readback::texture(bed_water)).observe(
        |event: On<ReadbackComplete>, mut capture: ResMut<Capture>| {
            capture.bytes = event.data.clone();
            capture.serial += 1;
        },
    );
    app
}

/// Steps `frames` frames, then pauses until a fresh readback lands, and returns the depth per
/// cell, row by row.
fn run_for(app: &mut App, frames: u32, deadline: Instant) -> Vec<f32> {
    for _ in 0..frames {
        step(app);
        assert!(Instant::now() < deadline, "river run timed out");
    }
    app.world_mut()
        .resource_mut::<ShallowWaterController>()
        .paused = true;
    let serial = app.world().resource::<Capture>().serial;
    while app.world().resource::<Capture>().serial <= serial + 1 {
        step(app);
        assert!(Instant::now() < deadline, "river readback timed out");
    }
    app.world_mut()
        .resource_mut::<ShallowWaterController>()
        .paused = false;
    let bytes = &app.world().resource::<Capture>().bytes;
    let row = bytes.len() / NY as usize;
    (0..(NX * NY) as usize)
        .map(|i| {
            let o = (i / NX as usize) * row + (i % NX as usize) * 16 + 4;
            f32::from_le_bytes(bytes[o..o + 4].try_into().unwrap())
        })
        .collect()
}

#[test]
#[ignore = "requires a graphics adapter"]
fn a_river_settles_at_normal_depth_and_passes_its_discharge() {
    // Manning's uniform flow: q = h^(5/3) √S / n.
    let normal_depth = (Q * MANNING_N / SLOPE.sqrt()).powf(0.6);
    let mut app = channel_app();
    let deadline = Instant::now() + Duration::from_secs(240);
    // The flow fills the dry channel within about 80 s, the time a kinematic wave takes to cross
    // it, and the depth then closes on normal depth with about that time constant.
    let settled = run_for(&mut app, 5000, deadline);
    let later = run_for(&mut app, 500, deadline);
    for (i, h) in later.iter().enumerate() {
        assert!(h.is_finite() && *h >= 0.0, "cell {i} holds {h}");
    }
    let mean = |cells: &[f32], columns: std::ops::Range<u32>| {
        let picked: Vec<f32> = (0..NY)
            .flat_map(|j| columns.clone().map(move |i| (i, j)))
            .map(|(i, j)| cells[(j * NX + i) as usize])
            .collect();
        picked.iter().sum::<f32>() / picked.len() as f32
    };
    for columns in [16..32, 40..56, 64..80] {
        let h = mean(&later, columns.clone());
        assert!(
            (h - normal_depth).abs() < 0.05 * normal_depth,
            "depth {h} m over columns {columns:?}, normal depth {normal_depth} m"
        );
    }
    // In a steady river what flows in flows out, so the volume holds still.
    let volume = |cells: &[f32]| cells.iter().sum::<f32>();
    let inflow = Q * NY as f32 * 500.0 * DT;
    let change = volume(&later) - volume(&settled);
    assert!(
        change.abs() < 0.01 * inflow,
        "the volume moved {change} m³ while {inflow} m³ flowed in"
    );
}

const VX: u32 = 96;
const VY: u32 = 48;

/// Bed of a valley falling along +x: an 8 m channel 1 m below the valley floor, banks 0.5 m above
/// it out to 7 m from the centreline, and a floodplain 0.7 m below the floor behind them, walled
/// at the grid's sides. `notch` lowers the left bank (+z) to 0.8 m below the floor over a stretch.
fn valley_bed(x: f32, z: f32, notch: bool) -> f32 {
    let floor = SLOPE * (48.0 - x);
    if z.abs() < 4.0 {
        floor - 1.0
    } else if z.abs() < 7.0 {
        if notch && z > 0.0 && (-8.0..0.0).contains(&x) {
            floor - 0.8
        } else {
            floor + 0.5
        }
    } else {
        floor - 0.7
    }
}

fn valley_river() -> RiverSpline {
    RiverSpline {
        points: vec![Vec2::new(-60.0, 0.0), Vec2::new(60.0, 0.0)],
        width: 8.0,
        discharge: 5.0,
        outflow_slope: SLOPE,
    }
}

/// Runs the valley until it settles and returns its bed and depth per cell.
fn settled_valley(notch: bool) -> (Vec<f32>, Vec<f32>) {
    settled_valley_with(notch, false)
}

/// The same, with ocean borders and a PML on every edge, around a sea far below the valley, the
/// way the beach's river runs in over a land edge.
fn settled_valley_with(notch: bool, ocean_edges: bool) -> (Vec<f32>, Vec<f32>) {
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
    .add_plugins(ShallowWaterPlugin)
    .insert_resource(TimeUpdateStrategy::ManualDuration(Duration::from_secs_f32(
        1.0 / 60.0,
    )))
    .insert_resource(bevy::render::error_handler::RenderErrorHandler(
        |error, _, _| {
            panic!("GPU validation failed: {error:?}");
        },
    ))
    .init_resource::<Capture>();
    let mut images = app.world_mut().resource_mut::<Assets<Image>>();
    let mut controller = ShallowWaterController::new(&mut images, VX, VY);
    let placement = ShallowWaterPlacement {
        center: Vec2::ZERO,
        cell_size: 1.0,
        datum: 0.0,
        blend_cells: 0.0,
    };
    // Solver heights stay positive.
    let bed: Vec<f32> = (0..VX * VY)
        .map(|i| {
            let x = (i % VX) as f32 + 0.5 - 0.5 * VX as f32;
            let z = (i / VX) as f32 + 0.5 - 0.5 * VY as f32;
            2.0 + valley_bed(x, z, notch)
        })
        .collect();
    controller.bed_source = images.add(bed_source_image(VX, VY, &bed));
    controller.preset_index = SHALLOW_WATER_PRESET_BED_SOURCE;
    controller.fill_level = 0.0;
    controller.gravity = G;
    controller.dt = DT;
    controller.manning_n = MANNING_N;
    controller.placement = Some(placement);
    if ocean_edges {
        controller.left_border = ShallowWaterBorder::Ocean;
        controller.right_border = ShallowWaterBorder::Ocean;
        controller.bottom_border = ShallowWaterBorder::Ocean;
        controller.top_border = ShallowWaterBorder::Ocean;
        controller.pml_width = 8;
        controller.pml_eta_rest = 0.0;
        controller.ocean_band_cells = 5.0;
    }
    let mut profile = ShallowWaterBorderProfile::new(UVec2::new(VX, VY));
    profile.add_river(&valley_river(), &placement);
    controller.border_profile = Some(images.add(profile.image()));
    app.insert_resource(controller);
    while app.plugins_state() == PluginsState::Adding {
        bevy::tasks::tick_global_task_pools_on_main_thread();
        thread::sleep(Duration::from_millis(1));
    }
    app.finish();
    app.cleanup();
    step(&mut app);
    let bed_water = app
        .world()
        .resource::<ShallowWaterController>()
        .bed_water
        .clone();
    app.world_mut().spawn(Readback::texture(bed_water)).observe(
        |event: On<ReadbackComplete>, mut capture: ResMut<Capture>| {
            capture.bytes = event.data.clone();
            capture.serial += 1;
        },
    );
    let deadline = Instant::now() + Duration::from_secs(240);
    for _ in 0..3000 {
        step(&mut app);
        assert!(Instant::now() < deadline, "valley run timed out");
    }
    app.world_mut()
        .resource_mut::<ShallowWaterController>()
        .paused = true;
    let serial = app.world().resource::<Capture>().serial;
    while app.world().resource::<Capture>().serial <= serial + 1 {
        step(&mut app);
        assert!(Instant::now() < deadline, "valley readback timed out");
    }
    let bytes = &app.world().resource::<Capture>().bytes;
    let row = bytes.len() / VY as usize;
    let lane = |o: usize| f32::from_le_bytes(bytes[o..o + 4].try_into().unwrap());
    (0..(VX * VY) as usize)
        .map(|i| {
            let o = (i / VX as usize) * row + (i % VX as usize) * 16;
            (lane(o), lane(o + 4))
        })
        .unzip()
}

fn bank_report(notch: bool) -> RiverBankReport {
    let (bed, depth) = settled_valley(notch);
    let grid = SettledGrid {
        cells: UVec2::new(VX, VY),
        cell_size: 1.0,
        corner: -0.5 * Vec2::new(VX as f32, VY as f32),
        bed: &bed,
        depth: &depth,
    };
    check_river_banks(
        &valley_river(),
        &grid,
        &RiverBankCheck {
            margin: 3.0,
            // The ends of the grid, where the inflow and outflow sit.
            along: (20.0, 100.0),
            ..default()
        },
    )
}

#[test]
#[ignore = "requires a graphics adapter"]
fn the_bank_checker_finds_a_notch_in_a_settled_river() {
    let sound = bank_report(false);
    assert!(sound.is_clean(), "sound banks flagged: {sound:?}");
    let notched = bank_report(true);
    // The notch spans world x -8..0, 52 to 60 m along the river.
    assert_eq!(notched.leaks.len(), 1, "{notched:?}");
    let leak = &notched.leaks[0];
    assert!(leak.along.0 >= 50.0 && leak.along.1 <= 62.0, "{leak:?}");
    assert!(
        leak.deepest.y > 7.0,
        "the leak is behind the left bank: {leak:?}"
    );
    assert_eq!(notched.low_banks.len(), 1, "{notched:?}");
    let low = &notched.low_banks[0];
    assert_eq!(low.side, 1.0);
    assert!(
        low.along.0 >= 50.0 && low.along.1 <= 62.0,
        "low bank at {:?}",
        low.along
    );
}

#[test]
#[ignore = "requires a graphics adapter"]
fn a_river_runs_in_over_an_ocean_border_on_land() {
    // The ocean band and the PML pull the sea toward its level. On land, where the sea is far
    // below, they leave the water alone, so a river running in over that edge keeps it.
    let (_, depth) = settled_valley_with(false, true);
    let mut sum = 0.0;
    let mut count = 0;
    for j in 0..VY {
        let z = j as f32 + 0.5 - 0.5 * VY as f32;
        if z.abs() < 3.0 {
            for i in 30..66 {
                sum += depth[(j * VX + i) as usize];
                count += 1;
            }
        }
    }
    let mean = sum / count as f32;
    assert!(mean > 0.4, "the channel holds {mean} m of water");
}
