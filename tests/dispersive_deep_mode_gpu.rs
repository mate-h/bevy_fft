//! Dispersion of the Airy-only deep mode against the full hybrid. Run explicitly with a graphics
//! adapter.
use bevy::{
    app::PluginsState,
    prelude::*,
    render::gpu_readback::{Readback, ReadbackComplete},
    time::TimeUpdateStrategy,
    window::ExitCondition,
    winit::WinitPlugin,
};
use bevy_fft::{dispersive::*, fft::FftPlugin};
use std::{
    f32::consts::TAU,
    thread,
    time::{Duration, Instant},
};

const N: u32 = 128;
/// `init_basin` fills the patch to this depth over a flat bed.
const REST_DEPTH: f32 = 4.0;
const AMPLITUDE: f32 = 0.05;
const WAVELENGTH: f32 = 32.0;
const G: f32 = 9.81;
const DT: f32 = 1.0 / 60.0;

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

/// A standing wave `AMPLITUDE cos(k x)` released from rest, stepped `steps` times. Returns the
/// share of the initial amplitude left in the same mode, cos(ω t) for a linear wave.
fn standing_wave(deep_water_depth: Option<f32>, steps: u32) -> f32 {
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
    .add_plugins((FftPlugin, DispersivePlugin))
    .insert_resource(TimeUpdateStrategy::ManualDuration(Duration::from_secs_f32(
        DT,
    )))
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
    let world = app.world_mut();
    let mut controller = world.resource_scope(|world, mut images: Mut<Assets<Image>>| {
        DispersiveController::spawn(&mut world.commands(), &mut images, N)
    });
    controller.tile_world = N as f32;
    controller.dt = DT;
    controller.edge_trim_cells = 0.0;
    controller.deep_water_depth = deep_water_depth;
    controller.paused = true;
    let state = controller.state.clone();
    world.insert_resource(controller);
    world.spawn(Readback::texture(state.clone())).observe(
        |event: On<ReadbackComplete>, mut capture: ResMut<Capture>| {
            capture.bytes = event.data.clone();
            capture.serial += 1;
        },
    );
    world.flush();
    for _ in 0..30 {
        step(&mut app);
    }

    // Replace the basin with the standing wave, at rest.
    let k = TAU / WAVELENGTH;
    let column = |x: u32| AMPLITUDE * (k * (x as f32 + 0.5)).cos();
    {
        let mut images = app.world_mut().resource_mut::<Assets<Image>>();
        let mut image = images.get_mut(&state).unwrap();
        let data = image.data.as_mut().unwrap();
        for y in 0..N {
            for x in 0..N {
                let o = ((y * N + x) * 16) as usize;
                data[o..o + 4].copy_from_slice(&(REST_DEPTH + column(x)).to_le_bytes());
                data[o + 4..o + 16].fill(0);
            }
        }
    }
    for _ in 0..5 {
        step(&mut app);
    }

    let deadline = Instant::now() + Duration::from_secs(120);
    for _ in 0..steps {
        app.world_mut()
            .resource_mut::<DispersiveController>()
            .step_serial += 1;
        step(&mut app);
        assert!(Instant::now() < deadline, "dispersive run timed out");
    }
    let serial = app.world().resource::<Capture>().serial;
    while app.world().resource::<Capture>().serial <= serial + 1 {
        step(&mut app);
        assert!(Instant::now() < deadline, "dispersive readback timed out");
    }
    let bytes = &app.world().resource::<Capture>().bytes;
    let texel = bytes.len() / (N * N) as usize;
    let (mut projection, mut norm) = (0.0f32, 0.0f32);
    for y in 0..N {
        for x in 0..N {
            let i = (y * N + x) as usize;
            let h = f32::from_le_bytes(bytes[i * texel..i * texel + 4].try_into().unwrap());
            projection += (h - REST_DEPTH) * column(x);
            norm += column(x) * column(x);
        }
    }
    projection / norm
}

/// ω for wavenumber `k` at `depth`, with the solver's grid correction (Jeschke and Wojtan eq. 27).
fn omega(k: f32, depth: f32) -> f32 {
    let dx = 1.0;
    let beta = (2.0 * k / dx * (k * dx * 0.5).sin()).sqrt();
    (G * k * (k * depth).tanh()).sqrt() * k / beta
}

#[test]
#[ignore = "requires a graphics adapter"]
fn deep_mode_waves_follow_deep_water_dispersion() {
    let steps = 45;
    let t = steps as f32 * DT;
    let k = TAU / WAVELENGTH;
    // 4 m of water under a 32 m wave is shallow; 200 m is deep.
    let deep = standing_wave(Some(200.0), steps);
    let expected_deep = (omega(k, 200.0) * t).cos();
    assert!(
        (deep - expected_deep).abs() < 0.03,
        "deep mode at {deep}, cos(ω t) = {expected_deep}"
    );
    let hybrid = standing_wave(None, steps);
    let expected_shallow = (omega(k, REST_DEPTH) * t).cos();
    assert!(
        (hybrid - expected_shallow).abs() < 0.06,
        "hybrid at {hybrid}, cos(ω t) = {expected_shallow}"
    );
}

#[test]
#[ignore = "requires a graphics adapter"]
fn deep_mode_stays_stable_over_many_periods() {
    // 20 s is about four and a half periods. A flux carried half a cell off the merge's faces
    // grows such a wave fivefold over this run.
    let steps = 1200;
    let k = TAU / WAVELENGTH;
    let left = standing_wave(Some(200.0), steps);
    let expected = (omega(k, 200.0) * steps as f32 * DT).cos();
    assert!(left.abs() <= 1.0, "the wave grew to {left}");
    assert!(
        (left - expected).abs() < 0.2,
        "after {steps} steps at {left}, cos(ω t) = {expected}"
    );
}

#[test]
#[ignore = "requires a graphics adapter"]
fn a_circling_boat_in_deep_mode_keeps_a_bounded_wake() {
    const SIDE: u32 = 256;
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
    .add_plugins((FftPlugin, DispersivePlugin))
    .insert_resource(TimeUpdateStrategy::ManualDuration(Duration::from_secs_f32(
        DT,
    )))
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
    let world = app.world_mut();
    let mut controller = world.resource_scope(|world, mut images: Mut<Assets<Image>>| {
        DispersiveController::spawn(&mut world.commands(), &mut images, SIDE)
    });
    // The wake example's boat: 6 m/s on a 55 m circle, a 6 m brush pressing 0.6 m down.
    controller.choose_mode(100.0, kelvin_wavelength(6.0, G));
    controller.brush_radius = 6.0;
    controller.brush_strength = 1.2;
    let state = controller.state.clone();
    world.insert_resource(controller);
    world.spawn(Readback::texture(state)).observe(
        |event: On<ReadbackComplete>, mut capture: ResMut<Capture>| {
            capture.bytes = event.data.clone();
            capture.serial += 1;
        },
    );
    world.flush();
    for _ in 0..30 {
        step(&mut app);
    }
    let deadline = Instant::now() + Duration::from_secs(300);
    let mut angle = 0.0f32;
    for frame in 0..1200 {
        angle += 6.0 / 55.0 * DT;
        let mut controller = app.world_mut().resource_mut::<DispersiveController>();
        let at = controller.world_to_cells(Vec2::from_angle(angle) * 55.0);
        controller.pointer_prev = if frame == 0 { at } else { controller.pointer };
        controller.pointer = at;
        controller.brush_active = true;
        step(&mut app);
        assert!(Instant::now() < deadline, "dispersive run timed out");
    }
    let serial = app.world().resource::<Capture>().serial;
    while app.world().resource::<Capture>().serial <= serial + 1 {
        step(&mut app);
        assert!(Instant::now() < deadline, "dispersive readback timed out");
    }
    let bytes = &app.world().resource::<Capture>().bytes;
    let texel = bytes.len() / (SIDE * SIDE) as usize;
    let (mut highest, mut total) = (0.0f32, 0.0f32);
    for i in 0..(SIDE * SIDE) as usize {
        let eta =
            f32::from_le_bytes(bytes[i * texel..i * texel + 4].try_into().unwrap()) - REST_DEPTH;
        highest = highest.max(eta.abs());
        total += eta;
    }
    let mean = total / (SIDE * SIDE) as f32;
    // The deepest point is the 0.6 m dent under the boat; the wake stays below it.
    assert!(highest < 1.0, "waves reached {highest} m");
    assert!(highest > 0.3, "no wake: {highest} m");
    // The brush presses the surface instead of adding water.
    assert!(mean.abs() < 1e-3, "mean level moved {mean} m");
}
