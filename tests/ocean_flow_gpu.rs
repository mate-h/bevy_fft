//! The ocean's flow lane against the change of the displacement it writes. Run explicitly with a
//! graphics adapter.
//!
//! In linear wave theory a surface particle moves with the time derivative of its displacement, so
//! the LOD writer's flow times the step must equal the change of the LOD displacement over that step:
//! height in the displacement array's B lane, horizontal X in its A lane and Z in the foam array's B
//! lane, the last two scaled by the choppiness. The ocean of the `ocean` example runs headless at a
//! fixed step, and the three arrays are read back on consecutive frames.
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
    collections::BTreeMap,
    thread,
    time::{Duration, Instant},
};

/// Cascade tiles of the `ocean` example, largest first.
const CASCADE_TILES: [f32; 3] = [256.0, 48.0, 9.0];
const CASCADE_HANDOFF_WAVES: f32 = 6.0;
const FFT_SIZE: u32 = 256;

const RES: usize = 128;
const SLICES: usize = 7;
const DT: f32 = 1.0 / 30.0;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
enum Lane {
    Displacement,
    Foam,
    Flow,
}

/// Readbacks by the update in which they arrived.
#[derive(Resource, Default)]
struct Captures {
    update: u64,
    by_update: BTreeMap<u64, BTreeMap<Lane, Vec<u8>>>,
}

fn step(app: &mut App) {
    app.world_mut().resource_mut::<Captures>().update += 1;
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

fn ocean_app(wind_speed: f32) -> App {
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
    .init_resource::<Captures>();
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

/// Lane `lane` of texel `i` of an `Rgba16Float` readback.
fn half_lane(bytes: &[u8], i: usize, lane: usize) -> f32 {
    let o = i * 8 + lane * 2;
    half(u16::from_le_bytes([bytes[o], bytes[o + 1]]))
}

/// Lane `lane` of texel `i` of an `Rgba32Float` readback.
fn float_lane(bytes: &[u8], i: usize, lane: usize) -> f32 {
    let o = i * 16 + lane * 4;
    f32::from_le_bytes(bytes[o..o + 4].try_into().unwrap())
}

/// Readbacks of the displacement, foam and flow arrays on two consecutive frames.
fn two_frames(wind_speed: f32) -> [BTreeMap<Lane, Vec<u8>>; 2] {
    let mut app = ocean_app(wind_speed);
    for _ in 0..(3.0 / DT) as u32 {
        step(&mut app);
    }
    let lod = app.world().resource::<WaterLodData>();
    let images = [
        (Lane::Displacement, lod.displacement.clone()),
        (Lane::Foam, lod.foam.clone()),
        (Lane::Flow, lod.flow.clone()),
    ];
    for (lane, handle) in images {
        app.world_mut().spawn(Readback::texture(handle)).observe(
            move |event: On<ReadbackComplete>, mut captures: ResMut<Captures>| {
                let update = captures.update;
                captures
                    .by_update
                    .entry(update)
                    .or_default()
                    .insert(lane, event.data.clone());
            },
        );
    }
    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        step(&mut app);
        assert!(Instant::now() < deadline, "readbacks timed out");
        let captures = app.world().resource::<Captures>();
        let complete: Vec<_> = captures
            .by_update
            .iter()
            .filter(|(_, lanes)| lanes.len() == 3)
            .collect();
        // Settle for a few frames, then take two consecutive complete updates.
        if complete.len() < 6 {
            continue;
        }
        // A readback can repeat the previous frame; the pair must hold two different frames.
        for pair in complete.windows(2).rev() {
            let differ = pair[0].1[&Lane::Displacement] != pair[1].1[&Lane::Displacement];
            if *pair[1].0 == *pair[0].0 + 1 && differ {
                return [pair[0].1.clone(), pair[1].1.clone()];
            }
        }
    }
}

/// Correlation and relative RMS error of `measured` against `expected`.
fn agreement(measured: &[f32], expected: &[f32]) -> (f32, f32) {
    let n = measured.len() as f32;
    let mean = |v: &[f32]| v.iter().sum::<f32>() / n;
    let (mm, me) = (mean(measured), mean(expected));
    let (mut cov, mut vm, mut ve, mut err) = (0.0, 0.0, 0.0, 0.0);
    for (m, e) in measured.iter().zip(expected) {
        cov += (m - mm) * (e - me);
        vm += (m - mm) * (m - mm);
        ve += (e - me) * (e - me);
        err += (m - e) * (m - e);
    }
    (cov / (vm * ve).sqrt(), (err / ve).sqrt())
}

#[test]
#[ignore = "requires a graphics adapter"]
fn ocean_flow_is_the_change_of_its_displacement() {
    let surface = OceanSurfaceSettings::default();
    let [before, after] = two_frames(10.0);
    let layer = RES * RES;
    let mut failures = Vec::new();
    // Fine slices, where every cascade shows and the half-float displacement still resolves a
    // frame's change.
    for slice in 0..3 {
        let texels = slice * layer..(slice + 1) * layer;
        let mut flow = [Vec::new(), Vec::new(), Vec::new()];
        let mut change = [Vec::new(), Vec::new(), Vec::new()];
        for i in texels {
            let d = |frame: &BTreeMap<Lane, Vec<u8>>, lane| {
                half_lane(&frame[&Lane::Displacement], i, lane)
            };
            let z = |frame: &BTreeMap<Lane, Vec<u8>>| half_lane(&frame[&Lane::Foam], i, 2);
            change[0].push((d(&after, 3) - d(&before, 3)) / surface.choppiness);
            change[1].push((z(&after) - z(&before)) / surface.choppiness);
            change[2].push(d(&after, 2) - d(&before, 2));
            for (lane, f) in flow.iter_mut().enumerate() {
                f.push(float_lane(&after[&Lane::Flow], i, lane) * DT);
            }
        }
        for (lane, name) in ["X", "Z", "height"].iter().enumerate() {
            let (r, rms) = agreement(&flow[lane], &change[lane]);
            let rms_of = |v: &[f32]| (v.iter().map(|x| x * x).sum::<f32>() / v.len() as f32).sqrt();
            println!(
                "slice {slice} {name}: correlation {r:.4}, relative RMS error {rms:.3}, RMS flow step {:.2e}, RMS change {:.2e}",
                rms_of(&flow[lane]),
                rms_of(&change[lane]),
            );
            if !(r >= 0.98 && rms <= 0.2) {
                failures.push((slice, *name, r, rms));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "flow strays from the change of displacement: {failures:?}"
    );
}
