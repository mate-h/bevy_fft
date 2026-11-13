//! Advected foam texture coordinates on the GPU. Run explicitly with a graphics adapter.
//!
//! A steady shear drift `u = (s z, 0)` carries a texture coordinate that starts over at `age` zero
//! to `x - s z age`, so each phase's offset falls along Z with slope `-s age`. The strain-rate norm
//! of that shear is `s / sqrt 2`, so the phase's accumulated deformation is `s age / sqrt 2`. Both
//! follow from the same age, which the test reads back without knowing the exact frame.
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
const LAYERS: usize = WATER_FOAM_FLOW_LAYERS_PER_SLICE as usize;
/// Seconds between restarts of the short and the long layer. Matches `FOAM_FLOW_LATENCY`.
const LATENCY: [f32; 2] = [1.0, 4.0];

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

fn next_capture(app: &mut App) -> Vec<u8> {
    let first = app.world().resource::<Capture>().serial;
    let deadline = Instant::now() + Duration::from_secs(30);
    while app.world().resource::<Capture>().serial <= first {
        step(app);
        assert!(Instant::now() < deadline, "flow readback timed out");
    }
    app.world().resource::<Capture>().bytes.clone()
}

/// The four lanes of flow layer `layer` at texel `(x, y)`.
fn texel(bytes: &[u8], layer: usize, x: usize, y: usize) -> Vec4 {
    let row = bytes.len() / (RES * LAYERS);
    let o = (layer * RES + y) * row + x * 16;
    let lane = |i: usize| f32::from_le_bytes(bytes[o + i..o + i + 4].try_into().unwrap());
    Vec4::new(lane(0), lane(4), lane(8), lane(12))
}

/// World Z of texel row `y` with one 1 m slice centred on the origin.
fn texel_z(y: usize) -> f32 {
    y as f32 + 0.5 - RES as f32 * 0.5
}

fn flow_app(dt: f32, shear: f32) -> App {
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

    let drift = app.world().resource::<WaterLodData>().drift.clone();
    {
        let mut images = app.world_mut().resource_mut::<Assets<Image>>();
        let mut image = images.get_mut(&drift).unwrap();
        let data = image.data.as_mut().unwrap();
        for y in 0..RES {
            let u = shear * texel_z(y);
            for x in 0..RES {
                let o = (y * RES + x) * 8;
                data[o..o + 4].copy_from_slice(&u.to_le_bytes());
                data[o + 4..o + 8].copy_from_slice(&0f32.to_le_bytes());
            }
        }
    }

    let flow = app.world().resource::<WaterLodData>().foam_flow.clone();
    app.world_mut().spawn(Readback::texture(flow)).observe(
        |event: On<ReadbackComplete>, mut capture: ResMut<Capture>| {
            capture.bytes = event.data.clone();
            capture.serial += 1;
        },
    );
    app
}

/// Mean over the interior of the offset's slope along Z, its slope along X, and the deformation,
/// for flow layer `layer` and phase `phase`.
struct Phase {
    slope_z: f32,
    slope_x: f32,
    deformation: f32,
    deformation_spread: f32,
    z_lane_spread: f32,
}

fn measure(bytes: &[u8], layer: usize, phase: usize) -> Phase {
    let offset = |x: usize, y: usize| {
        let t = texel(bytes, layer, x, y);
        if phase == 0 { t.xy() } else { t.zw() }
    };
    let deformation = |x: usize, y: usize| texel(bytes, 2, x, y)[2 * layer + phase];
    let (lo, hi) = (RES / 2 - 12, RES / 2 + 12);
    let mut n = 0.0;
    let (mut slope_z, mut slope_x, mut d) = (0.0, 0.0, 0.0);
    let (mut d_min, mut d_max) = (f32::MAX, f32::MIN);
    let (mut zl_min, mut zl_max) = (f32::MAX, f32::MIN);
    for y in lo..hi {
        for x in lo..hi {
            slope_z += 0.5 * (offset(x, y + 1).x - offset(x, y - 1).x);
            slope_x += 0.5 * (offset(x + 1, y).x - offset(x - 1, y).x);
            let here = deformation(x, y);
            d += here;
            d_min = d_min.min(here);
            d_max = d_max.max(here);
            zl_min = zl_min.min(offset(x, y).y);
            zl_max = zl_max.max(offset(x, y).y);
            n += 1.0;
        }
    }
    Phase {
        slope_z: slope_z / n,
        slope_x: slope_x / n,
        deformation: d / n,
        deformation_spread: d_max - d_min,
        z_lane_spread: zl_max - zl_min,
    }
}

/// Distance between two ages on a cycle of length `period`.
fn cyclic_gap(a: f32, b: f32, period: f32) -> f32 {
    let d = (a - b).rem_euclid(period);
    d.min(period - d)
}

#[test]
#[ignore = "requires a graphics adapter"]
fn sheared_texture_coordinates_and_deformation_follow_the_phase_ages() {
    let dt = 1.0 / 60.0;
    let shear = 0.05;
    let mut app = flow_app(dt, shear);
    // Long enough for every phase to start over at least once.
    for _ in 0..320 {
        step(&mut app);
    }
    let bytes = next_capture(&mut app);
    let mut ages = [[0.0f32; 2]; 2];
    for (layer, layer_ages) in ages.iter_mut().enumerate() {
        for (phase, slot) in layer_ages.iter_mut().enumerate() {
            let p = measure(&bytes, layer, phase);
            let age = -p.slope_z / shear;
            *slot = age;
            let label = format!("layer {layer} phase {phase}");
            assert!(
                age > -0.02 && age < LATENCY[layer] + 0.02,
                "{label}: offset slope gives age {age}"
            );
            assert!(p.slope_x.abs() < 1e-4, "{label}: X slope {}", p.slope_x);
            assert!(
                p.z_lane_spread < 1e-4,
                "{label}: Z lane spread {}",
                p.z_lane_spread
            );
            let expected = shear * age / std::f32::consts::SQRT_2;
            assert!(
                (p.deformation - expected).abs() < 0.02 * expected.max(0.01),
                "{label}: deformation {} against {expected} at age {age}",
                p.deformation
            );
            assert!(
                p.deformation_spread < 1e-3,
                "{label}: deformation spread {}",
                p.deformation_spread
            );
        }
    }
    // The phases of a layer run half a latency apart, and both layers start over together.
    for layer in 0..2 {
        let gap = cyclic_gap(ages[layer][0], ages[layer][1], LATENCY[layer]);
        assert!(
            (gap - 0.5 * LATENCY[layer]).abs() < 0.02,
            "layer {layer}: phases {:?} are {gap} s apart",
            ages[layer]
        );
    }
    let gap = cyclic_gap(ages[0][0], ages[1][0], LATENCY[0]);
    assert!(gap < 0.02, "layer ages {ages:?} disagree by {gap} s");
}
