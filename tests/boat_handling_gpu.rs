//! The boat example's handling on the FFT ocean, with its wake patch: speed, steadiness and
//! turning. Run explicitly with a graphics adapter.
#[path = "../examples/common/boat.rs"]
mod boat;
#[path = "../examples/common/boat_drive.rs"]
mod boat_drive;
#[path = "../examples/common/boat_hull.rs"]
mod boat_hull;

use bevy::{
    app::PluginsState, prelude::*, render::renderer::RenderDevice, time::TimeUpdateStrategy,
    window::ExitCondition, winit::WinitPlugin,
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
    // Full thrust balances the drag at about 3.8 m/s in calm water; waves cost a little.
    assert!((3.0..4.3).contains(&speed), "speed {speed} m/s");
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
