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
