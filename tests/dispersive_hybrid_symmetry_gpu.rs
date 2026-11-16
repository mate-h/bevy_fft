//! Mirror symmetry of the hybrid's surface waves. A flux carried half a cell off the faces the
//! merge reads shifts every Airy step by a cell and pushes the waves one way. Run explicitly with
//! a graphics adapter.
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
const CENTER: u32 = 64;

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

/// A 0.1 m Gaussian hump, 2 m wide, on the centre cell, released from rest and stepped `steps`
/// times. Returns the surface deviation per cell, row by row.
fn hump(deep_water_depth: Option<f32>, steps: u32) -> Vec<f32> {
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
    {
        let mut images = app.world_mut().resource_mut::<Assets<Image>>();
        let mut image = images.get_mut(&state).unwrap();
        let data = image.data.as_mut().unwrap();
        for y in 0..N {
            for x in 0..N {
                let d = Vec2::new(x as f32, y as f32) - CENTER as f32;
                let eta = 0.1 * (-d.length_squared() / 4.0).exp();
                let o = ((y * N + x) * 16) as usize;
                data[o..o + 4].copy_from_slice(&(REST_DEPTH + eta).to_le_bytes());
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
    (0..(N * N) as usize)
        .map(|i| {
            f32::from_le_bytes(bytes[i * texel..i * texel + 4].try_into().unwrap()) - REST_DEPTH
        })
        .collect()
}

/// Energy of the part of `eta` that is odd under x → 2·CENTER − x, against its total energy.
fn odd_share_x(eta: &[f32]) -> f32 {
    let (mut odd, mut total) = (0.0f32, 0.0f32);
    for y in 0..N {
        for x in 1..N {
            let m = 2 * CENTER - x;
            if m >= N {
                continue;
            }
            let (a, b) = (eta[(y * N + x) as usize], eta[(y * N + m) as usize]);
            odd += (0.5 * (a - b)).powi(2);
            total += a * a;
        }
    }
    odd / total
}

/// Centre of mass of the surface deviation's energy along x, in cells from the centre.
fn drift_x(eta: &[f32]) -> f32 {
    let (mut moment, mut total) = (0.0f32, 0.0f32);
    for y in 0..N {
        for x in 0..N {
            let e = eta[(y * N + x) as usize].powi(2);
            moment += e * (x as f32 - CENTER as f32);
            total += e;
        }
    }
    moment / total
}

#[test]
#[ignore = "requires a graphics adapter"]
fn hybrid_waves_spread_symmetrically() {
    // 3 s: the ring has spread about 18 m but not wrapped around.
    for (name, depth) in [("hybrid", None), ("deep", Some(100.0))] {
        let eta = hump(depth, 180);
        let odd = odd_share_x(&eta);
        let drift = drift_x(&eta);
        // Before the fix the hybrid had 6% of its energy odd and drifted 1.25 cells.
        assert!(odd < 1e-4, "{name}: odd share {odd}, drift {drift} cells");
        assert!(drift.abs() < 0.05, "{name}: drift {drift} cells");
        // The same along y, by transposing the patch.
        let transposed: Vec<f32> = (0..N * N)
            .map(|i| eta[((i % N) * N + i / N) as usize])
            .collect();
        let odd = odd_share_x(&transposed);
        let drift = drift_x(&transposed);
        assert!(odd < 1e-4, "{name} along y: odd share {odd}");
        assert!(drift.abs() < 0.05, "{name} along y: drift {drift} cells");
    }
}
