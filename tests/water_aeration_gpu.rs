//! Air fraction of white water on the GPU. Run explicitly with a graphics adapter.
//!
//! A heightfield holds a stationary hydraulic jump across the grid: water 0.2 m deep runs along
//! +X at Froude number 2 or 3 and jumps to Bélanger's conjugate depth over a 1.5 m wide face.
//! The discharge per unit width is the same everywhere, so nothing but the jump loses energy. The
//! air fraction the foam sim keeps in `foam_state.b` should peak at the jump and fall off
//! downstream at the rate bubbles leave the deeper water. Still water, gently varying subcritical
//! flow, and uniform supercritical flow with no jump should hold none. The air that leaves the
//! column downstream of a jump leaves residual foam behind.
use bevy::{
    app::PluginsState,
    asset::RenderAssetUsages,
    prelude::*,
    render::{
        gpu_readback::{Readback, ReadbackComplete},
        render_resource::{Extent3d, TextureDimension, TextureFormat, TextureUsages},
    },
    time::TimeUpdateStrategy,
    window::ExitCondition,
    winit::WinitPlugin,
};
use bevy_fft::water::*;
use std::{
    thread,
    time::{Duration, Instant},
};

const CELLS: u32 = 128;
const GRAVITY: f32 = 9.81;
/// World X of the jump's face.
const JUMP_X: f32 = -20.0;
/// Width of the jump's face in meters.
const FACE: f32 = 1.5;
const INFLOW_DEPTH: f32 = 0.2;

#[derive(Resource, Default)]
struct Capture {
    bytes: Vec<u8>,
    serial: u64,
}

fn float_image(data: impl Iterator<Item = Vec4>) -> Image {
    let bytes: Vec<u8> = data
        .flat_map(|v| v.to_array())
        .flat_map(f32::to_le_bytes)
        .collect();
    let mut image = Image::new(
        Extent3d {
            width: CELLS,
            height: CELLS,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        bytes,
        TextureFormat::Rgba32Float,
        RenderAssetUsages::RENDER_WORLD,
    );
    image.texture_descriptor.usage |= TextureUsages::TEXTURE_BINDING;
    image
}

/// World X of the centre of grid column `i`.
fn cell_x(i: u32) -> f32 {
    (i % CELLS) as f32 + 0.5 - CELLS as f32 / 2.0
}

/// Bed, depth and velocity along X for a flow over a flat bed whose depth at world X is `depth`,
/// with discharge `q` per unit width.
fn flow(depth: impl Fn(f32) -> f32, q: f32) -> (Image, Image) {
    stepped_flow(|_| 0.0, depth, q)
}

/// Bed, depth and velocity along X for a flow over a bed of height `bed` at world X.
fn stepped_flow(bed: impl Fn(f32) -> f32, depth: impl Fn(f32) -> f32, q: f32) -> (Image, Image) {
    let n = CELLS * CELLS;
    let bed_water = float_image((0..n).map(|i| {
        let x = cell_x(i);
        Vec4::new(bed(x), depth(x), 0.0, 0.0)
    }));
    let velocity = float_image((0..n).map(|i| Vec4::new(q / depth(cell_x(i)), 0.0, 0.0, 0.0)));
    (bed_water, velocity)
}

/// Bélanger's conjugate depth (Chanson 2009, eq. 1) for inflow Froude number `froude`.
fn conjugate_depth(froude: f32) -> f32 {
    0.5 * INFLOW_DEPTH * ((1.0 + 8.0 * froude * froude).sqrt() - 1.0)
}

fn inflow_discharge(froude: f32) -> f32 {
    froude * (GRAVITY * INFLOW_DEPTH).sqrt() * INFLOW_DEPTH
}

fn jump(froude: f32) -> (Image, Image) {
    let d2 = conjugate_depth(froude);
    let depth = move |x: f32| {
        let t = 0.5 * (1.0 + ((x - JUMP_X) / (0.5 * FACE)).tanh());
        INFLOW_DEPTH + (d2 - INFLOW_DEPTH) * t
    };
    flow(depth, inflow_discharge(froude))
}

fn step(app: &mut App) {
    app.update();
    assert!(app.should_exit().is_none(), "renderer must remain healthy");
    thread::sleep(Duration::from_millis(2));
}

fn aeration_app((bed_water, velocity): (Image, Image)) -> App {
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
    .insert_resource(WaterLodClear)
    .insert_resource(WaterLodSettings {
        resolution: CELLS,
        slice_count: 1,
        base_texel_size: 1.0,
    })
    .insert_resource(TimeUpdateStrategy::ManualDuration(Duration::from_secs_f32(
        1.0 / 60.0,
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
    let mut images = app.world_mut().resource_mut::<Assets<Image>>();
    let bed_water = images.add(bed_water);
    let velocity = images.add(velocity);
    app.world_mut().spawn(WaterHeightfield {
        dry: WaterHeightfieldDry::Level(0.0),
        ..WaterHeightfield::new(bed_water, velocity, UVec2::splat(CELLS))
    });
    let state = app.world().resource::<WaterLodData>().foam_state.clone();
    app.world_mut().spawn(Readback::texture(state)).observe(
        |event: On<ReadbackComplete>, mut capture: ResMut<Capture>| {
            capture.bytes = event.data.clone();
            capture.serial += 1;
        },
    );
    app
}

/// Air fraction along the grid's middle row, one value per texel from west to east, after
/// `seconds` of simulated time.
fn air_profile(app: &mut App, seconds: f32) -> Vec<f32> {
    state_profile(app, seconds).iter().map(|t| t.z).collect()
}

/// Foam state along the grid's middle row after `seconds` of simulated time: active foam,
/// residual foam and air fraction in `x`, `y` and `z`.
fn state_profile(app: &mut App, seconds: f32) -> Vec<Vec3> {
    for _ in 0..(seconds * 60.0) as u32 {
        step(app);
    }
    let first = app.world().resource::<Capture>().serial;
    let deadline = Instant::now() + Duration::from_secs(30);
    while app.world().resource::<Capture>().serial <= first {
        step(app);
        assert!(Instant::now() < deadline, "foam state readback timed out");
    }
    let bytes = newest_half(&app.world().resource::<Capture>().bytes);
    let res = CELLS as usize;
    let row = bytes.len() / res;
    let y = res / 2;
    (0..res)
        .map(|x| {
            let lane = |i: usize| {
                let o = y * row + x * 16 + 4 * i;
                f32::from_le_bytes(bytes[o..o + 4].try_into().unwrap())
            };
            Vec3::new(lane(0), lane(1), lane(2))
        })
        .collect()
}

/// Texel index of world X on the one 1 m slice centred on the origin.
fn texel(x: f32) -> usize {
    (x + CELLS as f32 / 2.0).floor() as usize
}

/// The newer of the two frames the foam state array holds, by the sim time stamped in alpha.
fn newest_half(bytes: &[u8]) -> &[u8] {
    let (a, b) = bytes.split_at(bytes.len() / 2);
    let stamp = |half: &[u8]| f32::from_le_bytes(half[12..16].try_into().unwrap());
    if stamp(b) > stamp(a) { b } else { a }
}

/// Checks a jump at inflow Froude number `froude` peaks at its face with an air fraction in
/// `expected`, and that the air falls off downstream at the detrainment rate.
fn check_jump(froude: f32, expected: std::ops::RangeInclusive<f32>) {
    let mut app = aeration_app(jump(froude));
    let air = air_profile(&mut app, 4.0);
    let (peak_at, peak) = air
        .iter()
        .copied()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .unwrap();
    let peak_x = peak_at as f32 + 0.5 - CELLS as f32 / 2.0;
    let at = |x: f32| air[texel(x)];
    println!(
        "Fr1 {froude}: peak {peak:.3} at x = {peak_x}; upstream {:.4}; downstream +2 m {:.3}, +4 m {:.3}, +6 m {:.4}, +10 m {:.5}",
        at(JUMP_X - 6.0),
        at(JUMP_X + 2.0),
        at(JUMP_X + 4.0),
        at(JUMP_X + 6.0),
        at(JUMP_X + 10.0),
    );
    assert!(
        (peak_x - JUMP_X).abs() <= 3.0,
        "air peaks at x = {peak_x}, not at the jump"
    );
    assert!(expected.contains(&peak), "peak air fraction {peak}");
    assert!(at(JUMP_X - 6.0) < 0.01, "air upstream {}", at(JUMP_X - 6.0));
    // Past the jump the air leaves at the detrainment speed over the conjugate depth, carried at
    // the tailwater's speed, so it falls by e every u2 d2 / U metres.
    let settings = WaterAerationSettings::default();
    let d2 = conjugate_depth(froude);
    let u2 = inflow_discharge(froude) / d2;
    let e_fold = u2 * d2 / settings.detrainment_speed;
    let (a, b) = (at(JUMP_X + 3.0), at(JUMP_X + 5.0));
    assert!(
        a > b && b > at(JUMP_X + 8.0),
        "air must fall off downstream"
    );
    let measured = 2.0 / (a / b).ln();
    println!("e-folding length {measured:.2} m, expected {e_fold:.2} m");
    assert!(
        (0.5 * e_fold..=2.0 * e_fold).contains(&measured),
        "air falls by e every {measured} m downstream, expected about {e_fold} m"
    );
    assert!(at(JUMP_X + 12.0) < 0.01 * peak, "{}", at(JUMP_X + 12.0));
}

#[test]
#[ignore = "requires a graphics adapter"]
fn a_hydraulic_jump_aerates_the_water_at_its_face_and_the_air_leaves_downstream() {
    // Vallé and Pasternack (2006) measured a median of 33 percent, from 26 to 48, over the roller of
    // a jump at Fr1 = 2, and peaks of 40 to 50 percent.
    check_jump(2.0, 0.26..=0.48);
    check_jump(3.0, 0.4..=0.5);
}

#[test]
#[ignore = "requires a graphics adapter"]
fn air_leaving_the_water_trails_residual_foam_downstream() {
    // The same jump with and without foam from surfacing air, so the foam the jump makes by
    // breaking cancels out.
    let run = |foam_yield: f32| {
        let mut app = aeration_app(jump(2.0));
        app.insert_resource(WaterAerationSettings {
            foam_yield,
            ..default()
        });
        // Long enough for foam to drift 10 m at the tailwater's 1.2 m/s.
        state_profile(&mut app, 10.0)
    };
    let with = run(WaterAerationSettings::default().foam_yield);
    let without = run(0.0);
    let gained = |x: f32| with[texel(x)].y - without[texel(x)].y;
    let peak = with.iter().map(|t| t.z).fold(0.0, f32::max);
    println!(
        "residual foam gained: upstream {:.3}, +2 m {:.3}, +6 m {:.3}, +10 m {:.3}; air at +6 m {:.4} of peak {peak:.3}",
        gained(JUMP_X - 6.0),
        gained(JUMP_X + 2.0),
        gained(JUMP_X + 6.0),
        gained(JUMP_X + 10.0),
        with[texel(JUMP_X + 6.0)].z,
    );
    assert!(
        gained(JUMP_X - 6.0).abs() < 0.01,
        "foam upstream of the jump"
    );
    // Six metres on, the air is nearly gone but the foam it left is still there.
    assert!(with[texel(JUMP_X + 6.0)].z < 0.1 * peak);
    assert!(gained(JUMP_X + 6.0) > 0.2, "{}", gained(JUMP_X + 6.0));
    // Past the air the foam only drifts and fades, so over the next 4 m it halves once per
    // residual half-life at the tailwater's speed.
    let u2 = inflow_discharge(2.0) / conjugate_depth(2.0);
    let half_life = WaterFoamSettings::default().residual_half_life;
    let expected = 0.5f32.powf(4.0 / u2 / half_life);
    let fade = gained(JUMP_X + 10.0) / gained(JUMP_X + 6.0);
    println!("foam fades to {fade:.3} over 4 m, expected {expected:.3}");
    assert!((fade / expected - 1.0).abs() < 0.25, "{fade} vs {expected}");
    // The air is the same either way.
    for (a, b) in with.iter().zip(&without) {
        assert!((a.z - b.z).abs() < 1e-3, "air changed with the foam yield");
    }
}

#[test]
#[ignore = "requires a graphics adapter"]
fn still_and_smoothly_flowing_water_holds_no_air() {
    let cases = [
        ("still", flow(|_| 1.0, 0.0)),
        (
            "subcritical",
            flow(
                |x| 1.0 + 0.2 * (std::f32::consts::TAU * x / 40.0).sin(),
                1.5,
            ),
        ),
        (
            "supercritical",
            flow(|_| INFLOW_DEPTH, inflow_discharge(3.0)),
        ),
    ];
    for (name, images) in cases {
        let mut app = aeration_app(images);
        let air = air_profile(&mut app, 2.0);
        let most = air.iter().copied().fold(0.0f32, f32::max);
        println!("{name}: most air {most:.5}");
        assert!(most < 0.005, "{name} water holds air {most}");
    }
}

#[test]
#[ignore = "requires a graphics adapter"]
fn a_waterfall_aerates_its_lip_and_landing() {
    // 0.3 m of water at 1 m/s pours over a 3 m drop at x = 0 into a pool 0.5 m deep.
    let fall = stepped_flow(
        |x| if x < 0.0 { 3.0 } else { 0.0 },
        |x| if x < 0.0 { 0.3 } else { 0.5 },
        0.3,
    );
    let mut app = aeration_app(fall);
    let air = air_profile(&mut app, 2.0);
    let near = air[texel(-1.0)..=texel(1.0)]
        .iter()
        .copied()
        .fold(0.0f32, f32::max);
    let upstream = air[texel(-10.0)];
    println!("waterfall: lip and landing {near:.3}, 10 m upstream {upstream:.4}");
    assert!((0.15..=0.5).contains(&near), "air at the fall {near}");
    assert!(upstream < 0.01, "air upstream {upstream}");
}
