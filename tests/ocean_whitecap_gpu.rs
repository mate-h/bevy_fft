//! Ocean whitecap coverage against Monahan and Woolf (1989). Run explicitly with a graphics adapter.
//!
//! The ocean cascades of the `ocean` example run headless at a fixed step, and the foam sim's
//! active and residual lanes are read back from [`WaterLodData::foam_state`]. Active foam renders
//! at about 0.55 albedo and residual foam at about 0.1 (Koepke 1984), so mean active coverage is
//! the Stage A coverage, which counts only foam with albedo of 0.5 or more. Their Stage B coverage
//! comes from photographs that see every whitecap, so it is compared with active and residual foam
//! together.
//!
//! A second test compares the breaking the LOD writer reports where the far range takes over,
//! with the far range off and then forced on everywhere.
use bevy::{
    app::PluginsState,
    prelude::*,
    render::gpu_readback::{Readback, ReadbackComplete},
    time::TimeUpdateStrategy,
    window::ExitCondition,
    winit::WinitPlugin,
};
use bevy_fft::prelude::*;
use std::{
    thread,
    time::{Duration, Instant},
};

/// Cascade tiles of the `ocean` example, largest first.
const CASCADE_TILES: [f32; 3] = [256.0, 48.0, 9.0];
const CASCADE_HANDOFF_WAVES: f32 = 6.0;
const FFT_SIZE: u32 = 256;

const RES: usize = 256;
const SLICES: usize = 7;
/// Slice whose texels resolve the smallest cascade's breakers and which is wide enough to hold
/// several of the largest cascade's crests at once.
const MEASURED_SLICE: usize = 3;
/// Slices with 0.5 m and 1 m texels, where the far range blends in by default.
const BLEND_SLICES: [usize; 2] = [4, 5];
const DT: f32 = 1.0 / 30.0;

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

fn cascade_band(index: usize) -> (f32, f32) {
    let handoff = |i: usize| CASCADE_HANDOFF_WAVES * std::f32::consts::TAU / CASCADE_TILES[i];
    let k_min = if index == 0 { 0.0 } else { handoff(index) };
    let k_max = if index + 1 < CASCADE_TILES.len() {
        handoff(index + 1)
    } else {
        f32::INFINITY
    };
    (k_min, k_max)
}

fn ocean_app(wind_speed: f32, surface: OceanSurfaceSettings) -> App {
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
    .add_plugins((FftPlugin, OceanPlugin))
    .insert_resource(WaterLodSettings {
        resolution: RES as u32,
        slice_count: SLICES as u32,
        ..default()
    })
    .insert_resource(TimeUpdateStrategy::ManualDuration(Duration::from_secs_f32(
        DT,
    )))
    .insert_resource(bevy::render::error_handler::RenderErrorHandler(
        |error, _, _| {
            panic!("GPU validation failed: {error:?}");
        },
    ))
    .insert_resource(surface)
    .init_resource::<Capture>();
    while app.plugins_state() == PluginsState::Adding {
        bevy::tasks::tick_global_task_pools_on_main_thread();
        thread::sleep(Duration::from_millis(1));
    }
    app.finish();
    app.cleanup();
    for (index, &tile) in CASCADE_TILES.iter().enumerate() {
        let (band_k_min, band_k_max) = cascade_band(index);
        app.world_mut().spawn((
            FftSource::square_inverse_only(FFT_SIZE),
            OceanSimSettings {
                texture_size: FFT_SIZE,
                tile_size: tile,
                band_k_min,
                band_k_max,
                wind_speed,
                ..default()
            },
            OceanCascade {
                index: index as u32,
            },
            OceanH0Uniform::default(),
            OceanDynamicUniform::default(),
            OceanFoamUniform::default(),
            OceanFoamPhase(0),
        ));
    }
    step(&mut app);
    app
}

/// Mean active foam and mean total foam over slice `slice` of a foam state readback.
fn slice_mean(bytes: &[u8], slice: usize) -> Vec2 {
    let layer = bytes.len() / SLICES;
    let texels = &bytes[slice * layer..(slice + 1) * layer];
    let lane = |o: usize| f32::from_le_bytes(texels[o..o + 4].try_into().unwrap());
    let sum: Vec2 = (0..RES * RES)
        .map(|i| {
            let active = lane(i * 8);
            Vec2::new(active, (active + lane(i * 8 + 4)).min(1.0))
        })
        .sum();
    sum / (RES * RES) as f32
}

/// Mean of `reduce` over `samples` readbacks of `image` one second apart, after a warm-up long
/// enough for residual foam to reach steady state.
fn measure(
    wind_speed: f32,
    surface: OceanSurfaceSettings,
    image: impl Fn(&WaterLodData) -> Handle<Image>,
    reduce: impl Fn(&[u8]) -> Vec2,
    samples: u32,
) -> Vec2 {
    let mut app = ocean_app(wind_speed, surface);
    for _ in 0..(10.0 / DT) as u32 {
        step(&mut app);
    }
    let handle = image(app.world().resource::<WaterLodData>());
    app.world_mut().spawn(Readback::texture(handle)).observe(
        |event: On<ReadbackComplete>, mut capture: ResMut<Capture>| {
            capture.bytes = event.data.clone();
            capture.serial += 1;
        },
    );
    let mut total = Vec2::ZERO;
    for _ in 0..samples {
        let first = app.world().resource::<Capture>().serial;
        let deadline = Instant::now() + Duration::from_secs(30);
        for _ in 0..(1.0 / DT) as u32 {
            step(&mut app);
        }
        while app.world().resource::<Capture>().serial <= first {
            step(&mut app);
            assert!(Instant::now() < deadline, "readback timed out");
        }
        total += reduce(&app.world().resource::<Capture>().bytes);
    }
    total / samples as f32
}

/// Value of a little-endian half float.
fn half(bits: u16) -> f32 {
    let exponent = ((bits >> 10) & 31) as i32;
    let mantissa = (bits & 1023) as f32;
    let sign = if bits >> 15 == 1 { -1.0 } else { 1.0 };
    if exponent == 0 {
        sign * mantissa * 2f32.powi(-24)
    } else {
        sign * (1.0 + mantissa / 1024.0) * 2f32.powi(exponent - 15)
    }
}

/// Mean breaking intensity, the red lane of the LOD foam array, over slice `slice`.
fn breaking_mean(bytes: &[u8], slice: usize) -> f32 {
    let layer = bytes.len() / SLICES;
    let texels = &bytes[slice * layer..(slice + 1) * layer];
    let sum: f32 = (0..RES * RES)
        .map(|i| half(u16::from_le_bytes([texels[i * 8], texels[i * 8 + 1]])))
        .sum();
    sum / (RES * RES) as f32
}

/// Monahan and Woolf (1989) Stage A and Stage B whitecap coverage at wind speed `u` in m/s, as
/// fractions at zero air-sea temperature difference.
fn monahan_woolf(u: f32) -> Vec2 {
    Vec2::new(2.92e-7 * u.powf(3.204), 1.95e-5 * u.powf(2.55))
}

#[test]
#[ignore = "requires a graphics adapter"]
fn whitecap_coverage_follows_monahan_and_woolf_from_5_to_15_m_per_s() {
    let mut failures = Vec::new();
    for u in [5.0, 10.0, 15.0] {
        let measured = measure(
            u,
            OceanSurfaceSettings::default(),
            |lod| lod.foam_state.clone(),
            |bytes| slice_mean(bytes, MEASURED_SLICE),
            30,
        );
        let fit = monahan_woolf(u);
        let ratio = measured / fit;
        println!(
            "U = {u} m/s: active {:.3e} against W_A {:.3e} (x{:.2}), total {:.3e} against W_B {:.3e} (x{:.2})",
            measured.x, fit.x, ratio.x, measured.y, fit.y, ratio.y
        );
        if !(0.5..=2.0).contains(&ratio.x) || !(0.5..=2.0).contains(&ratio.y) {
            failures.push((u, ratio));
        }
    }
    assert!(
        failures.is_empty(),
        "coverage outside a factor of two of the fits: {failures:?}"
    );
}

#[test]
#[ignore = "requires a graphics adapter"]
fn far_whitecaps_break_like_near_whitecaps_where_they_meet() {
    let blend = |bytes: &[u8]| {
        Vec2::new(
            breaking_mean(bytes, BLEND_SLICES[0]),
            breaking_mean(bytes, BLEND_SLICES[1]),
        )
    };
    let mut failures = Vec::new();
    for u in [5.0, 10.0, 15.0] {
        let near = measure(
            u,
            OceanSurfaceSettings {
                far_foam_texel: f32::MAX,
                ..default()
            },
            |lod| lod.foam.clone(),
            blend,
            30,
        );
        let far = measure(
            u,
            OceanSurfaceSettings {
                far_foam_texel: 1e-4,
                ..default()
            },
            |lod| lod.foam.clone(),
            blend,
            30,
        );
        let ratio = far / near;
        println!(
            "U = {u} m/s: breaking near {:.3e}, {:.3e} and far {:.3e}, {:.3e} on 0.5 m and 1 m texels (x{:.2}, x{:.2})",
            near.x, near.y, far.x, far.y, ratio.x, ratio.y
        );
        if !(0.4..=3.0).contains(&ratio.x) || !(0.4..=3.0).contains(&ratio.y) {
            failures.push((u, ratio));
        }
    }
    assert!(
        failures.is_empty(),
        "far range breaking strays from the near range: {failures:?}"
    );
}
