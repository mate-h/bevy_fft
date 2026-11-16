//! A dispersive patch that follows a moving point by whole cells. Run explicitly with a graphics
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
    thread,
    time::{Duration, Instant},
};

const N: u32 = 128;
/// `init_basin` fills the patch to this depth over a flat bed.
const REST_DEPTH: f32 = 4.0;
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

/// Cell `(x, y)` of a 0.1 m Gaussian hump, 2 m wide, on cell (64, 64).
fn hump(x: u32, y: u32) -> f32 {
    let d = Vec2::new(x as f32, y as f32) - 64.0;
    0.1 * (-d.length_squared() / 4.0).exp()
}

/// A 128 m patch of 4 m water at the origin holding the hump, at rest, paused.
fn patch_app(deep_water_depth: Option<f32>) -> App {
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
    controller.deep_water_depth = deep_water_depth;
    controller.placement = Some(DispersivePlacement::default());
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
    {
        let mut images = app.world_mut().resource_mut::<Assets<Image>>();
        let mut image = images.get_mut(&state).unwrap();
        let data = image.data.as_mut().unwrap();
        for y in 0..N {
            for x in 0..N {
                let o = ((y * N + x) * 16) as usize;
                data[o..o + 4].copy_from_slice(&(REST_DEPTH + hump(x, y)).to_le_bytes());
                data[o + 4..o + 16].fill(0);
            }
        }
    }
    for _ in 0..5 {
        step(&mut app);
    }
    app
}

/// Runs `steps` steps of the paused patch.
fn run(app: &mut App, steps: u32) {
    for _ in 0..steps {
        app.world_mut()
            .resource_mut::<DispersiveController>()
            .step_serial += 1;
        step(app);
    }
}

/// The surface deviation per cell, row by row, once the readback has caught up with the paused
/// patch. Readbacks arrive a few frames after they're issued, so it lets four go by.
fn eta(app: &mut App) -> Vec<f32> {
    let deadline = Instant::now() + Duration::from_secs(60);
    let serial = app.world().resource::<Capture>().serial;
    while app.world().resource::<Capture>().serial <= serial + 3 {
        step(app);
        assert!(Instant::now() < deadline, "readback timed out");
    }
    let bytes = &app.world().resource::<Capture>().bytes;
    let texel = bytes.len() / (N * N) as usize;
    (0..(N * N) as usize)
        .map(|i| {
            f32::from_le_bytes(bytes[i * texel..i * texel + 4].try_into().unwrap()) - REST_DEPTH
        })
        .collect()
}

#[test]
#[ignore = "requires a graphics adapter"]
fn following_moves_the_water_by_whole_cells() {
    let mut app = patch_app(Some(100.0));
    {
        let mut controller = app.world_mut().resource_mut::<DispersiveController>();
        // 8.4 m along X and -3.6 m along Z round to 8 and -4 cells.
        controller.follow(Vec2::new(8.4, -3.6), 2.0);
        assert_eq!(controller.moved, IVec2::new(8, -4));
        assert_eq!(controller.placement.unwrap().center, Vec2::new(8.0, -4.0));
        // Closer than the margin, it stays.
        controller.follow(Vec2::new(9.0, -4.0), 2.0);
        assert_eq!(controller.moved, IVec2::new(8, -4));
    }
    let e = eta(&mut app);
    let mut worst = 0.0f32;
    for y in 0..N {
        for x in 0..N {
            // Cell (x, y) now holds what cell (x + 8, y - 4) held, and rest where that was off
            // the patch.
            let (sx, sy) = (x as i32 + 8, y as i32 - 4);
            let expected = if (0..N as i32).contains(&sx) && (0..N as i32).contains(&sy) {
                hump(sx as u32, sy as u32)
            } else {
                0.0
            };
            worst = worst.max((e[(y * N + x) as usize] - expected).abs());
        }
    }
    assert!(worst < 1e-6, "shifted surface off by {worst} m");
}

/// The surface after `before` steps, a move of `shift` cells, and `after` more steps.
fn moved_run(deep_water_depth: Option<f32>, shift: IVec2, before: u32, after: u32) -> Vec<f32> {
    let mut app = patch_app(deep_water_depth);
    run(&mut app, before);
    {
        let mut controller = app.world_mut().resource_mut::<DispersiveController>();
        controller.follow(shift.as_vec2(), 1.0);
        assert_eq!(controller.moved, shift);
    }
    run(&mut app, after);
    eta(&mut app)
}

#[test]
#[ignore = "requires a graphics adapter"]
fn a_moved_patch_evolves_like_a_still_one() {
    // The step is the same everywhere on the patch except for the edge trim, so away from the
    // edges a patch that moved half way through matches one that stayed, cell for world cell.
    let shift = IVec2::new(6, -3);
    for (name, depth) in [("deep", Some(100.0)), ("hybrid", None)] {
        let still = moved_run(depth, IVec2::ZERO, 60, 60);
        let moved = moved_run(depth, shift, 60, 60);
        let (mut diff, mut norm) = (0.0f32, 0.0f32);
        for y in 24..104u32 {
            for x in 24..104u32 {
                let a = moved[(y * N + x) as usize];
                let b = still
                    [(((y as i32 + shift.y) as u32) * N + (x as i32 + shift.x) as u32) as usize];
                diff += (a - b).powi(2);
                norm += b * b;
            }
        }
        let error = (diff / norm).sqrt();
        eprintln!("{name}: relative error {error}");
        assert!(
            error < 0.01,
            "{name}: moved patch off by {error} of its waves"
        );
    }
}
