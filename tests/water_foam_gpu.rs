//! Foam sim transport and decay on the GPU. Run explicitly with a graphics adapter.
use bevy::{
    app::PluginsState,
    prelude::*,
    render::gpu_readback::{Readback, ReadbackComplete},
    time::TimeUpdateStrategy,
    window::ExitCondition,
    winit::WinitPlugin,
};
use bevy_fft::water::*;
use std::{
    thread,
    time::{Duration, Instant},
};

const RES: usize = 64;

#[derive(Resource, Default)]
struct Capture {
    bytes: Vec<u8>,
    serial: u64,
}

fn step(app: &mut App) {
    app.update();
    assert!(app.should_exit().is_none(), "renderer must remain healthy");
    thread::sleep(Duration::from_millis(2));
}

/// Steps until a readback newer than the current one arrives.
fn next_capture(app: &mut App) -> Vec<u8> {
    let first = app.world().resource::<Capture>().serial;
    let deadline = Instant::now() + Duration::from_secs(30);
    while app.world().resource::<Capture>().serial <= first {
        step(app);
        assert!(Instant::now() < deadline, "foam readback timed out");
    }
    app.world().resource::<Capture>().bytes.clone()
}

/// Active and residual foam per texel, row by row.
fn foam_texels(bytes: &[u8]) -> Vec<Vec2> {
    let row = bytes.len() / RES;
    (0..RES)
        .flat_map(|y| {
            (0..RES).map(move |x| {
                let o = y * row + x * 8;
                let lane =
                    |i: usize| f32::from_le_bytes(bytes[o + i..o + i + 4].try_into().unwrap());
                Vec2::new(lane(0), lane(4))
            })
        })
        .collect()
}

/// World X and Z of texel `(x, y)` with one 1 m slice centred on the origin.
fn texel_world(x: usize, y: usize) -> Vec2 {
    Vec2::new(x as f32 + 0.5, y as f32 + 0.5) - RES as f32 * 0.5
}

fn foam_app(dt: f32, settings: WaterFoamSettings) -> App {
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
    .add_plugins(WaterPlugin)
    .insert_resource(WaterLodSettings {
        resolution: RES as u32,
        slice_count: 1,
        base_texel_size: 1.0,
    })
    .insert_resource(settings)
    .insert_resource(TimeUpdateStrategy::ManualDuration(Duration::from_secs_f32(
        dt,
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
    step(&mut app);
    let state = app.world().resource::<WaterLodData>().foam_state.clone();
    app.world_mut().spawn(Readback::texture(state)).observe(
        |event: On<ReadbackComplete>, mut capture: ResMut<Capture>| {
            capture.bytes = event.data.clone();
            capture.serial += 1;
        },
    );
    app
}

/// Writes `source(xz)` into the red lane of the LOD foam array, the coverage writers produce.
fn set_source(app: &mut App, source: impl Fn(Vec2) -> f32) {
    let handle = app.world().resource::<WaterLodData>().foam.clone();
    let mut images = app.world_mut().resource_mut::<Assets<Image>>();
    let mut image = images.get_mut(&handle).unwrap();
    let data = image.data.as_mut().unwrap();
    for y in 0..RES {
        for x in 0..RES {
            let o = (y * RES + x) * 8;
            let bits = half_bits(source(texel_world(x, y)));
            data[o..o + 2].copy_from_slice(&bits.to_le_bytes());
        }
    }
}

fn set_drift(app: &mut App, drift: Vec2) {
    let handle = app.world().resource::<WaterLodData>().drift.clone();
    let mut images = app.world_mut().resource_mut::<Assets<Image>>();
    let mut image = images.get_mut(&handle).unwrap();
    for texel in image.data.as_mut().unwrap().chunks_exact_mut(8) {
        texel[0..4].copy_from_slice(&drift.x.to_le_bytes());
        texel[4..8].copy_from_slice(&drift.y.to_le_bytes());
    }
}

/// Nearest f16 for values in [0, 1], enough for test sources.
fn half_bits(v: f32) -> u16 {
    if v <= 6.0e-5 {
        return 0;
    }
    let exp = v.log2().floor() as i32;
    let frac = ((v / 2f32.powi(exp) - 1.0) * 1024.0).round() as u16;
    (((exp + 15) as u16) << 10) + frac
}

#[test]
#[ignore = "requires a graphics adapter"]
fn drifting_foam_keeps_its_total_and_moves_with_the_drift() {
    let dt = 1.0 / 60.0;
    let drift = Vec2::new(2.0, 0.0);
    let mut app = foam_app(
        dt,
        WaterFoamSettings {
            active_half_life: f32::INFINITY,
            residual_half_life: f32::INFINITY,
            residual_yield: 0.0,
        },
    );
    set_drift(&mut app, drift);
    let center = Vec2::new(-8.0, 2.0);
    set_source(&mut app, |xz| {
        0.6 * (-(xz - center).length_squared() / 18.0).exp()
    });
    for _ in 0..10 {
        step(&mut app);
    }
    set_source(&mut app, |_| 0.0);
    for _ in 0..5 {
        step(&mut app);
    }
    let measure = |bytes: &[u8]| {
        let texels = foam_texels(bytes);
        let mut total = 0.0;
        let mut moment = Vec2::ZERO;
        for y in 0..RES {
            for x in 0..RES {
                let a = texels[y * RES + x].x;
                total += a;
                moment += texel_world(x, y) * a;
            }
        }
        (total, moment / total)
    };
    let (total_a, centroid_a) = measure(&next_capture(&mut app));
    let steps = 60;
    for _ in 0..steps {
        step(&mut app);
    }
    let (total_b, centroid_b) = measure(&app.world().resource::<Capture>().bytes.clone());
    assert!(
        total_a > 10.0,
        "source must reach the foam sim, total {total_a}"
    );
    let moved = centroid_b - centroid_a;
    let expected = drift * dt * steps as f32;
    assert!(
        (total_b / total_a - 1.0).abs() < 0.01,
        "total foam changed from {total_a} to {total_b}"
    );
    assert!(
        (moved - expected).length() < 0.1,
        "centroid moved {moved:?}, expected {expected:?}"
    );
}

fn check_decay(dt: f32) {
    let settings = WaterFoamSettings::default();
    let mut app = foam_app(dt, settings);
    // Weak enough that residual foam stays below its cap of full coverage.
    set_source(&mut app, |xz| {
        if xz.abs().max_element() < 8.0 {
            0.1
        } else {
            0.0
        }
    });
    for _ in 0..((0.1 / dt).round() as u32) {
        step(&mut app);
    }
    set_source(&mut app, |_| 0.0);
    for _ in 0..3 {
        step(&mut app);
    }
    let sums = |bytes: &[u8]| foam_texels(bytes).into_iter().sum::<Vec2>();
    let a = sums(&next_capture(&mut app));
    let steps = (1.0 / dt).round() as u32;
    for _ in 0..steps {
        step(&mut app);
    }
    let b = sums(&app.world().resource::<Capture>().bytes.clone());
    assert!(
        a.x > 1.0 && a.y > 0.1,
        "source must reach the foam sim, {a:?}"
    );
    // Readbacks can land a frame apart from the updates that requested them, so count the steps
    // the active foam went through. Exact per-step decay makes that a whole number.
    let per_step = WaterFoamParams::step(&settings, WaterLodSlices::default(), dt);
    let counted = (b.x / a.x).ln() / per_step.active_decay.ln();
    assert!(
        (counted - counted.round()).abs() < 0.02 && (counted.round() - steps as f32).abs() <= 1.0,
        "dt {dt}: active foam went through {counted} steps, expected {steps}"
    );
    let exact = WaterFoamParams::step(&settings, WaterLodSlices::default(), counted.round() * dt);
    let residual = a.y * exact.residual_decay + a.x * exact.residual_gain;
    assert!(
        (b.y / residual - 1.0).abs() < 0.02,
        "dt {dt}: residual {} against exact {residual}",
        b.y
    );
}

#[test]
#[ignore = "requires a graphics adapter"]
fn foam_decay_matches_the_exact_solution_at_30_and_120_steps_per_second() {
    check_decay(1.0 / 30.0);
    check_decay(1.0 / 120.0);
}
