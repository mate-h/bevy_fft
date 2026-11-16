//! Water forces on a voxelized cube, evaluated on the GPU against flat water. Run explicitly with
//! a graphics adapter.
use bevy::{
    app::PluginsState, prelude::*, time::TimeUpdateStrategy, window::ExitCondition,
    winit::WinitPlugin,
};
use bevy_fft::{bodies::*, water::*};
use std::{
    sync::Arc,
    thread,
    time::{Duration, Instant},
};

const RHO: f32 = 1025.0;
const G: f32 = 9.81;

fn step(app: &mut App) {
    app.update();
    assert!(app.should_exit().is_none(), "renderer must remain healthy");
    thread::sleep(Duration::from_millis(2));
}

fn body_app() -> App {
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
    .add_plugins((WaterPlugin, FloatingBodyPlugin))
    .insert_resource(WaterLodSettings {
        resolution: 64,
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
    ));
    while app.plugins_state() == PluginsState::Adding {
        bevy::tasks::tick_global_task_pools_on_main_thread();
        thread::sleep(Duration::from_millis(1));
    }
    app.finish();
    app.cleanup();
    step(&mut app);
    app
}

/// Sets the water velocity of every LOD texel. The flow lanes are X, Z, then vertical.
fn set_current(app: &mut App, velocity: Vec2) {
    let handle = app.world().resource::<WaterLodData>().flow.clone();
    let mut images = app.world_mut().resource_mut::<Assets<Image>>();
    let mut image = images.get_mut(&handle).unwrap();
    let data = image.data.as_mut().unwrap();
    for texel in data.chunks_exact_mut(16) {
        texel[0..4].copy_from_slice(&velocity.x.to_le_bytes());
        texel[4..8].copy_from_slice(&velocity.y.to_le_bytes());
    }
}

fn cube() -> Arc<VoxelizedMesh> {
    Arc::new(voxelize(&Mesh::from(Cuboid::new(1.0, 1.0, 1.0)), 0.25).unwrap())
}

/// Spawns `body` at `y` and steps until its forces have been read back a few times.
fn settle(app: &mut App, body: FloatingBody, y: f32) -> FloatingBody {
    let entity = app
        .world_mut()
        .spawn((body, Transform::from_xyz(0.0, y, 0.0)))
        .id();
    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        step(app);
        let body = app.world().get::<FloatingBody>(entity).unwrap();
        if body.frame >= 5 {
            let body = body.clone();
            app.world_mut().despawn(entity);
            return body;
        }
        assert!(Instant::now() < deadline, "force readback timed out");
    }
}

fn still(hull: Arc<VoxelizedMesh>) -> FloatingBody {
    let mut body = FloatingBody::new(hull, 500.0);
    body.estimate_velocity = false;
    body
}

fn close(a: f32, b: f32, tolerance: f32) -> bool {
    (a - b).abs() <= tolerance * b.abs().max(1.0)
}

#[test]
#[ignore = "requires a graphics adapter"]
fn buoyancy_follows_the_submerged_volume() {
    let mut app = body_app();
    let hull = cube();
    let half = settle(&mut app, still(hull.clone()), 0.0);
    assert!(close(half.submerged_volume, 0.5, 0.01), "{half:?}");
    assert!(close(half.force.y, RHO * G * 0.5, 0.01), "{:?}", half.force);
    assert!(half.force.xz().length() < 1.0, "{:?}", half.force);
    assert!(half.torque.length() < 1.0, "{:?}", half.torque);
    // The bottom and the lower half of four sides.
    assert!(close(half.wetted_area, 3.0, 0.01), "{}", half.wetted_area);
    assert_eq!(half.sampled_center, Vec3::ZERO);

    // Lowered 0.1 m, the water line cuts a layer of voxels across the whole 1 m² section.
    let lower = settle(&mut app, still(hull.clone()), -0.1);
    assert!(close(lower.submerged_volume, 0.6, 0.01), "{lower:?}");
    assert!(
        close(lower.waterplane_area, 1.0, 0.01),
        "{}",
        lower.waterplane_area
    );
    assert!(
        close(lower.sampled_center.y, -0.1, 1e-4),
        "{:?}",
        lower.sampled_center
    );

    let under = settle(&mut app, still(hull.clone()), -3.0);
    assert!(close(under.force.y, RHO * G, 0.01), "{:?}", under.force);
    assert!(close(under.wetted_area, 6.0, 0.01), "{}", under.wetted_area);

    let above = settle(&mut app, still(hull), 3.0);
    assert_eq!(above.force, Vec3::ZERO);
    assert_eq!(above.submerged_volume, 0.0);
}

#[test]
#[ignore = "requires a graphics adapter"]
fn drag_opposes_motion_relative_to_the_water() {
    let mut app = body_app();
    let settings = *app.world().resource::<FloatingBodySettings>();
    let speed = 2.0;
    // Pressure on the wetted front half face and suction on the back, both head on.
    let expected = 0.5
        * (settings.pressure_linear * speed
            + settings.pressure_quadratic * speed * speed
            + settings.suction_linear * speed
            + settings.suction_quadratic * speed * speed);

    let mut body = still(cube());
    body.linear_velocity = Vec3::X * speed;
    let moving = settle(&mut app, body, 0.0);
    // Friction on the wetted sides and bottom adds a little.
    assert!(
        moving.force.x < -expected && moving.force.x > -1.05 * expected,
        "{:?}, expected about -{expected}",
        moving.force
    );
    assert!(
        close(moving.force.y, RHO * G * 0.5, 0.01),
        "{:?}",
        moving.force
    );

    // A body at rest in a current feels the same push, downstream.
    set_current(&mut app, Vec2::X * speed);
    let drifting = settle(&mut app, still(cube()), 0.0);
    assert!(
        close(drifting.force.x, -moving.force.x, 0.01),
        "{:?} against {:?}",
        drifting.force,
        moving.force
    );
}

#[test]
#[ignore = "requires a graphics adapter"]
fn torque_is_about_the_centre_of_mass() {
    let mut app = body_app();
    let mut body = still(cube());
    // Buoyancy acts on the centre line, 0.3 m behind a centre of mass shifted along +X.
    body.center_of_mass = Vec3::new(0.3, 0.0, 0.0);
    let shifted = settle(&mut app, body, 0.0);
    let lift = RHO * G * 0.5;
    assert!(
        close(shifted.torque.z, -0.3 * lift, 0.01),
        "{:?}",
        shifted.torque
    );
    assert!(
        shifted.torque.x.abs() < 1.0 && shifted.torque.y.abs() < 1.0,
        "{:?}",
        shifted.torque
    );
}
