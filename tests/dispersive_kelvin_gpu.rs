//! The Kelvin wake angle of a towed pressure in deep mode. Run explicitly with a graphics adapter.
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
    thread,
    time::{Duration, Instant},
};

const N: u32 = 256;
/// `init_basin` fills the patch to this depth over a flat bed.
const REST_DEPTH: f32 = 4.0;
const DT: f32 = 1.0 / 60.0;
const G: f32 = 9.81;

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

/// Tows the brush along +X at `speed` from the origin for `seconds`, on a 256 m patch of 1 m
/// cells in deep mode that follows it. Returns the surface deviation per cell, row by row, and
/// where the brush ended in cells.
fn tow(speed: f32, radius: f32, seconds: f32) -> (Vec<f32>, Vec2) {
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
    controller.choose_mode(100.0, kelvin_wavelength(speed, G));
    assert!(controller.deep_water_depth.is_some());
    // The patch follows the brush, so the ring the start sends out leaves through the trimmed
    // edges and only the steady wake stays.
    controller.placement = Some(DispersivePlacement::default());
    controller.brush_radius = radius;
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
    let frames = (seconds / DT) as u32;
    let mut at = Vec2::ZERO;
    for frame in 0..frames {
        let x = speed * frame as f32 * DT;
        let mut controller = app.world_mut().resource_mut::<DispersiveController>();
        controller.follow(Vec2::new(x, 0.0), 1.0);
        at = controller.world_to_cells(Vec2::new(x, 0.0));
        controller.pointer_prev = if frame == 0 { at } else { controller.pointer };
        controller.pointer = at;
        controller.brush_active = true;
        // Ramp the pressure in over 5 s, so the start sends out less of a ring.
        controller.brush_strength = 1.2 * (frame as f32 * DT / 5.0).min(1.0);
        step(&mut app);
        assert!(Instant::now() < deadline, "dispersive run timed out");
    }
    app.world_mut()
        .resource_mut::<DispersiveController>()
        .paused = true;
    let serial = app.world().resource::<Capture>().serial;
    while app.world().resource::<Capture>().serial <= serial + 1 {
        step(&mut app);
        assert!(Instant::now() < deadline, "dispersive readback timed out");
    }
    let bytes = &app.world().resource::<Capture>().bytes;
    let texel = bytes.len() / (N * N) as usize;
    let eta = (0..(N * N) as usize)
        .map(|i| {
            f32::from_le_bytes(bytes[i * texel..i * texel + 4].try_into().unwrap()) - REST_DEPTH
        })
        .collect();
    (eta, at)
}

/// Half-angle of the wake in degrees, as Rabaud and Moisy (2013) and Huang et al. (2021, Fig. 13)
/// measure it: at each distance from 40 to 116 m behind the brush, the angle of the largest wave
/// between 8 and 32 degrees off the track on either side, which off the transverse waves is the
/// cusp line. Median over distances.
fn wake_angle(eta: &[f32], at: Vec2) -> (f32, Vec<f32>) {
    let row = at.y.floor() as i32;
    let mut angles = Vec::new();
    for d in (40..116).step_by(4) {
        let x = (at.x - d as f32) as u32;
        let near = (d as f32 * 8f32.to_radians().tan()).ceil() as i32;
        let far = (d as f32 * 32f32.to_radians().tan()) as i32;
        let side = |sign: i32| {
            let z = (near..=far)
                .max_by(|&a, &b| {
                    let v = |z: i32| eta[((row + sign * z) as u32 * N + x) as usize].abs();
                    v(a).total_cmp(&v(b))
                })
                .unwrap();
            ((z as f32 + 0.5) / d as f32).atan().to_degrees()
        };
        angles.push(0.5 * (side(1) + side(-1)));
    }
    let mut sorted = angles.clone();
    sorted.sort_by(f32::total_cmp);
    (sorted[sorted.len() / 2], angles)
}

fn rms(eta: &[f32], xs: std::ops::Range<u32>, ys: std::ops::Range<u32>) -> f32 {
    let cells = (xs.len() * ys.len()) as f32;
    let sum: f32 = ys
        .flat_map(|y| xs.clone().map(move |x| (x, y)))
        .map(|(x, y)| eta[(y * N + x) as usize].powi(2))
        .sum();
    (sum / cells).sqrt()
}

#[test]
#[ignore = "requires a graphics adapter"]
fn a_towed_pressure_trails_a_kelvin_wake() {
    // A 5 m brush at 5 m/s is a hull Froude number of about 0.5, where the wake still opens at
    // the Kelvin angle. Its transverse waves are 16 m long and its cusp waves 11 m. The linear
    // steady solution for the same pressure reads 17.8 degrees with this measure, 40 to 116 m
    // behind. Before tanh(k h) was clamped, every wave shorter than 14 m came out NaN and was
    // zeroed, the cusps vanished, and this read 8 degrees.
    let (eta, at) = tow(5.0, 5.0, 60.0);
    let (angle, angles) = wake_angle(&eta, at);
    assert!(
        (angle - 19.47).abs() < 1.5,
        "wake half-angle {angle} degrees, per distance {angles:?}"
    );
    // A deep-water wake trails the brush; nothing runs ahead of it.
    let x = at.x as u32;
    let ahead = rms(&eta, x + 20..x + 100, 64..192);
    let behind = rms(&eta, x - 100..x - 20, 64..192);
    assert!(
        ahead < 0.05 * behind,
        "ahead {ahead} m against {behind} m behind"
    );
}
