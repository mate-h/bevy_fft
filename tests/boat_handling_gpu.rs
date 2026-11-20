//! The boat example's handling on the FFT ocean, with its wake patch: speed, steadiness and
//! turning. Run explicitly with a graphics adapter.
#[path = "../examples/common/boat.rs"]
mod boat;
#[path = "../examples/common/boat_drive.rs"]
mod boat_drive;
#[path = "../examples/common/boat_hull.rs"]
mod boat_hull;

use bevy::{
    app::PluginsState,
    prelude::*,
    render::{
        gpu_readback::{Readback, ReadbackComplete},
        renderer::RenderDevice,
    },
    time::TimeUpdateStrategy,
    window::ExitCondition,
    winit::WinitPlugin,
};
use bevy_fft::prelude::*;
use bevy_rapier3d::prelude::*;
use std::time::Duration;

/// GPU tests in this file take turns.
static GPU: std::sync::Mutex<()> = std::sync::Mutex::new(());

const DT: f32 = 1.0 / 60.0;
const CASCADE_TILES: [f32; 3] = [256.0, 48.0, 9.0];
const CASCADE_HANDOFF_WAVES: f32 = 6.0;
/// A moderate sea, wind at 5 m/s.
const WIND: f32 = 5.0;

/// One frame, then a wait for the GPU, as a window's vsync would, so the water forces come back
/// two or three frames late rather than as late as the CPU can run ahead.
fn step(app: &mut App) {
    app.update();
    assert!(app.should_exit().is_none(), "renderer must remain healthy");
    let _ = app.world().resource::<RenderDevice>().poll(
        bevy::render::render_resource::PollType::Wait {
            submission_index: None,
            timeout: Some(Duration::from_secs(1)),
        },
    );
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

/// The ocean, the boat at rest at the origin and its wake patch. Returns the app and the boat.
fn boat_app() -> (App, Entity) {
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
    .add_plugins((
        FftPlugin,
        OceanPlugin,
        FloatingBodyPlugin,
        DispersivePlugin,
        RapierPhysicsPlugin::<NoUserData>::default(),
        boat::BoatPhysicsPlugin,
    ))
    .insert_resource(TimeUpdateStrategy::ManualDuration(Duration::from_secs_f32(
        DT,
    )))
    .insert_resource(bevy::render::error_handler::RenderErrorHandler(
        |error, _, _| {
            panic!("GPU validation failed: {error:?}");
        },
    ));
    while app.plugins_state() == PluginsState::Adding {
        bevy::tasks::tick_global_task_pools_on_main_thread();
        std::thread::sleep(Duration::from_millis(1));
    }
    app.finish();
    app.cleanup();
    let world = app.world_mut();
    for (index, &tile) in CASCADE_TILES.iter().enumerate() {
        let (band_k_min, band_k_max) = cascade_band(index);
        world.spawn((
            FftSource::square_inverse_only(256),
            OceanSimSettings {
                texture_size: 256,
                tile_size: tile,
                band_k_min,
                band_k_max,
                wind_speed: WIND,
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
    let hull = boat_hull::build_hull(
        boat_hull::HullShape::default(),
        boat::VOXEL,
        boat::SEA_WATER,
    );
    // The water LOD data centres on the boat, as it does on the chase camera in the example.
    let entity = world
        .spawn((boat::boat_physics(&hull), Transform::default(), WaterViewer))
        .id();
    let wake = world.resource_scope(|world, mut images: Mut<Assets<Image>>| {
        boat::wake_patch(&mut world.commands(), &mut images, boat::WAKE_CELL)
    });
    world.insert_resource(wake);
    world.flush();
    // Until the boat is let go, after its forces have come back a few dozen times.
    for _ in 0..600 {
        step(&mut app);
        if *app.world().get::<RigidBody>(entity).unwrap() == RigidBody::Dynamic {
            break;
        }
    }
    assert_eq!(
        *app.world().get::<RigidBody>(entity).unwrap(),
        RigidBody::Dynamic,
        "the boat's forces never came back"
    );
    (app, entity)
}

/// What a run records each frame.
#[derive(Default)]
struct Track {
    /// Speed ahead in m/s, heading in degrees (unwrapped, positive to starboard), roll and pitch
    /// in degrees.
    speed: Vec<f32>,
    heading: Vec<f32>,
    roll: Vec<f32>,
    pitch: Vec<f32>,
}

/// Runs `seconds` with the lever and rudder at `lever` and `rudder_deg`, recording the boat.
fn run(
    app: &mut App,
    entity: Entity,
    seconds: f32,
    lever: f32,
    rudder_deg: f32,
    track: &mut Track,
) {
    {
        let mut drive = app.world_mut().get_mut::<boat::BoatDrive>(entity).unwrap();
        drive.lever = lever;
        drive.rudder = rudder_deg.to_radians();
    }
    for _ in 0..(seconds / DT) as usize {
        step(app);
        let world = app.world();
        let transform = world.get::<Transform>(entity).unwrap();
        let velocity = world.get::<Velocity>(entity).unwrap();
        let forward = (transform.rotation * Vec3::NEG_Z)
            .with_y(0.0)
            .normalize_or(Vec3::NEG_Z);
        let heading = forward.x.atan2(-forward.z).to_degrees();
        let heading = match track.heading.last() {
            Some(&last) => last + (heading - last + 180.0).rem_euclid(360.0) - 180.0,
            None => heading,
        };
        let (_, pitch, roll) = transform.rotation.to_euler(EulerRot::YXZ);
        track.speed.push(velocity.linear.dot(forward));
        track.heading.push(heading);
        track.roll.push(roll.to_degrees());
        track.pitch.push(pitch.to_degrees());
        assert!(
            transform.translation.y.abs() < 3.0,
            "the boat left the water: {}",
            transform.translation
        );
    }
}

fn mean(v: &[f32]) -> f32 {
    v.iter().sum::<f32>() / v.len().max(1) as f32
}

fn largest(v: &[f32]) -> f32 {
    v.iter().fold(0.0, |m, x| m.max(x.abs()))
}

#[test]
#[ignore = "requires a graphics adapter"]
fn full_throttle_holds_a_steady_speed_and_course() {
    let _gpu = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let (mut app, boat) = boat_app();
    let mut track = Track::default();
    run(&mut app, boat, 30.0, 1.0, 0.0, &mut track);
    let last = |v: &Vec<f32>| v[v.len() - (10.0 / DT) as usize..].to_vec();
    let speed = mean(&last(&track.speed));
    // She rises onto the plane, where full thrust balances the drag at about 7.5 m/s.
    assert!((6.5..8.5).contains(&speed), "speed {speed} m/s");
    assert!(
        largest(&track.roll) < 20.0,
        "roll {}°",
        largest(&track.roll)
    );
    assert!(
        largest(&track.pitch) < 12.0,
        "pitch {}°",
        largest(&track.pitch)
    );
    // Hands off, the skeg holds the course against the waves.
    let course = last(&track.heading);
    let drift = course[course.len() - 1] - course[0];
    assert!(drift.abs() < 30.0, "drifted {drift}° in 10 s");
}

#[test]
#[ignore = "requires a graphics adapter"]
fn hard_over_turns_her_around() {
    let _gpu = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let (mut app, boat) = boat_app();
    let mut track = Track::default();
    run(&mut app, boat, 10.0, 1.0, 0.0, &mut track);
    let start = *track.heading.last().unwrap();
    let before = track.heading.len();
    run(&mut app, boat, 15.0, 1.0, 35.0, &mut track);
    // Starboard rudder turns to starboard, through 180° in well under 15 s.
    let turned = track.heading[before..]
        .iter()
        .position(|&h| h - start >= 180.0)
        .map(|frames| frames as f32 * DT);
    assert!(
        turned.is_some_and(|t| t < 15.0),
        "turned {}° in 15 s",
        track.heading.last().unwrap() - start
    );
    let turning = &track.roll[before..];
    assert!(
        largest(turning) < 25.0,
        "heeled {}° in the turn",
        largest(turning)
    );
}

/// What the wake test reads back from the patch.
#[derive(Resource, Default)]
struct WakeCapture {
    state: Vec<u8>,
    head: Vec<u8>,
    serial: u64,
}

/// Bilinear sample of the per-cell values `v` of an `n` by `n` grid at grid coordinate `g`, cell
/// `i` spanning `i..i + 1`.
fn sample(v: &[f32], n: u32, g: Vec2) -> f32 {
    let x = g - 0.5;
    let i0 = x.floor();
    let f = x - i0;
    let at = |dx: i32, dy: i32| {
        let c = (i0.as_ivec2() + IVec2::new(dx, dy)).clamp(IVec2::ZERO, IVec2::splat(n as i32 - 1));
        v[(c.y as u32 * n + c.x as u32) as usize]
    };
    let a = at(0, 0) + (at(1, 0) - at(0, 0)) * f.x;
    let b = at(0, 1) + (at(1, 1) - at(0, 1)) * f.x;
    a + (b - a) * f.y
}

fn floats(bytes: &[u8], n: u32) -> Vec<f32> {
    let texel = bytes.len() / (n * n) as usize;
    (0..(n * n) as usize)
        .map(|i| f32::from_le_bytes(bytes[i * texel..i * texel + 4].try_into().unwrap()))
        .collect()
}

/// Crest-to-trough heights of the wake on lines abeam of the boat, 5, 7.5, 10, 12.5, 15 and 20 m
/// out, each the mean of port and starboard, from 5 m ahead back to where the patch's edge trim
/// starts. Holds the boat and the patch still to read the patch back, and prints the heights and
/// the pressure head's volume against her displacement. `speed` is only printed.
fn abeam_heights(app: &mut App, boat: Entity, speed: f32) -> Vec<f32> {
    let (state, head) = {
        let mut wake = app.world_mut().resource_mut::<DispersiveController>();
        wake.paused = true;
        let maps = wake.body_maps.as_ref().unwrap();
        (wake.state.clone(), maps.pressure_head.clone())
    };
    *app.world_mut().get_mut::<RigidBody>(boat).unwrap() = RigidBody::Fixed;
    app.init_resource::<WakeCapture>();
    app.world_mut().spawn(Readback::texture(state)).observe(
        |event: On<ReadbackComplete>, mut c: ResMut<WakeCapture>| {
            c.state = event.data.clone();
            c.serial += 1;
        },
    );
    app.world_mut().spawn(Readback::texture(head)).observe(
        |event: On<ReadbackComplete>, mut c: ResMut<WakeCapture>| {
            c.head = event.data.clone();
        },
    );
    for _ in 0..600 {
        step(app);
        if app.world().resource::<WakeCapture>().serial > 3 {
            break;
        }
    }
    let capture = app.world().resource::<WakeCapture>();
    assert!(capture.serial > 3, "the patch never came back");
    let wake = app.world().resource::<DispersiveController>();
    let n = wake.n;
    let cell = wake.cell_size();
    let eta: Vec<f32> = floats(&capture.state, n).iter().map(|h| h - 4.0).collect();
    let head = floats(&capture.head, n);
    let body = app.world().get::<FloatingBody>(boat).unwrap();
    let transform = *app.world().get::<Transform>(boat).unwrap();
    let head_volume = head.iter().sum::<f32>() * cell * cell;
    println!(
        "speed {speed:.2} m/s, mass {:.0} kg, submerged {:.2} m³ ({:.0}%), footprint scale {:.2}, \
         head volume {head_volume:.2} m³ against displacement {:.2} m³, deepest head {:.2} m",
        body.mass,
        body.submerged_volume,
        100.0 * body.submerged_volume * boat::SEA_WATER / body.mass,
        body.footprint_scale,
        body.mass / boat::SEA_WATER,
        head.iter().fold(0.0f32, |m, &h| m.max(h)),
    );
    let center = wake.placement.unwrap().center;
    let forward = (transform.rotation * Vec3::NEG_Z).xz().normalize();
    let side = Vec2::new(-forward.y, forward.x);
    let at = transform.translation.xz();
    // Inside the edge trim and the blend band.
    let half = 0.5 * n as f32 * cell - (wake.edge_trim_cells + 4.0) * cell;
    let mut heights = Vec::new();
    for lateral in [5.0f32, 7.5, 10.0, 12.5, 15.0, 20.0] {
        let mut sides = [0.0f32; 2];
        let mut reach = 0.0f32;
        for (k, sign) in [1.0f32, -1.0].into_iter().enumerate() {
            let (mut lo, mut hi) = (f32::MAX, f32::MIN);
            let mut back = -5.0f32;
            loop {
                let p = at - forward * back + side * sign * lateral;
                if (p - center).abs().max_element() > half {
                    break;
                }
                let e = sample(&eta, n, wake.world_to_cells(p));
                lo = lo.min(e);
                hi = hi.max(e);
                back += 0.25;
            }
            reach = back;
            sides[k] = hi - lo;
        }
        println!(
            "{lateral:4.1} m abeam: crest to trough {:.3} m port, {:.3} m starboard, over {reach:.0} m behind",
            sides[0], sides[1]
        );
        heights.push(0.5 * (sides[0] + sides[1]));
    }
    heights
}

/// The hull towed straight at 15 knots, held level and a little above her design waterline,
/// against the steady linear wave field of the pressure head she presses into the patch.
#[test]
#[ignore = "requires a graphics adapter"]
fn a_towed_hull_makes_the_wake_hull_theory_gives() {
    let _gpu = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let (mut app, boat) = boat_app();
    let speed = 7.7;
    // Held up about where she rides at speed. The water piles up under her forward half, so she
    // still displaces about her weight.
    let lift = 0.33;
    let start = app.world().get::<Transform>(boat).unwrap().translation;
    for frame in 0..(25.0 / DT) as usize {
        {
            let mut transform = app.world_mut().get_mut::<Transform>(boat).unwrap();
            transform.rotation = Quat::IDENTITY;
            transform.translation = start + Vec3::new(0.0, lift, -speed * DT * (frame + 1) as f32);
        }
        {
            let mut velocity = app.world_mut().get_mut::<Velocity>(boat).unwrap();
            velocity.linear = Vec3::new(0.0, 0.0, -speed);
            velocity.angular = Vec3::ZERO;
        }
        step(&mut app);
    }
    let heights = abeam_heights(&mut app, boat, speed);
    // The steady linear wave field of the head she presses with, about her weight, gives
    // 0.39-0.45 m at 10 m abeam within the patch (numpy, in the commit that added this test).
    // The obstruction under a moving hull had cut it to 0.15 m.
    assert!(
        (0.3..0.6).contains(&heights[2]),
        "the towed hull's wake is {} m crest to trough 10 m abeam",
        heights[2]
    );
}

/// Crest-to-trough heights of the Kelvin wake at full throttle on a line abeam, against what the
/// steady linear wave field of a pressure patch with the boat's weight gives.
#[test]
#[ignore = "requires a graphics adapter"]
fn the_wake_has_the_height_hull_theory_gives() {
    let _gpu = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let (mut app, boat) = boat_app();
    let mut track = Track::default();
    run(&mut app, boat, 25.0, 1.0, 0.0, &mut track);
    let speed = mean(&track.speed[track.speed.len() - 120..]);
    let heights = abeam_heights(&mut app, boat, speed);
    // A pressure patch of her 4.5 t at 15 knots, 3.5-7 m long and as wide as she is, makes
    // 0.34-0.46 m at 10 m abeam in deep water, over the 58 m behind her the patch holds. Allow
    // for the hull's own shape. With the obstruction pinning the water under her it was 0.21 m.
    let at_ten = heights[2];
    assert!(
        (0.25..0.7).contains(&at_ten),
        "the wake is {at_ten} m crest to trough 10 m abeam"
    );
}
