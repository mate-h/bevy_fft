//! Floating bodies pressing on a shallow water pool through the pressure head. Run explicitly
//! with a graphics adapter.
use bevy::{
    app::PluginsState,
    prelude::*,
    render::gpu_readback::{Readback, ReadbackComplete},
    time::TimeUpdateStrategy,
    window::ExitCondition,
    winit::WinitPlugin,
};
use bevy_fft::{bodies::*, shallow_water::*, water::*};
use std::{
    sync::Arc,
    thread,
    time::{Duration, Instant},
};

const CELLS: u32 = 64;
const DEPTH: f32 = 2.0;
const RHO: f32 = 1025.0;
const G: f32 = 9.81;

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

/// A walled pool of still water `DEPTH` deep, its surface at world height 0, with a pressure head.
fn pool_app() -> App {
    pool_app_with(|_| {})
}

/// [`pool_app`] with `extra` adding plugins before the app finishes building.
fn pool_app_with(extra: impl FnOnce(&mut App)) -> App {
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
    .add_plugins((ShallowWaterPlugin, FloatingBodyPlugin))
    .insert_resource(WaterLodClear)
    .insert_resource(WaterLodSettings {
        resolution: 128,
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
    let mut images = app.world_mut().resource_mut::<Assets<Image>>();
    let mut controller = ShallowWaterController::new(&mut images, CELLS, CELLS);
    let n = (CELLS * CELLS) as usize;
    controller.bed_source = images.add(bed_source_image(CELLS, CELLS, &vec![0.0; n]));
    controller.pressure_head = Some(images.add(shallow_r32_image(CELLS, CELLS)));
    controller.preset_index = SHALLOW_WATER_PRESET_BED_SOURCE;
    controller.fill_level = DEPTH;
    controller.friction = 0.3;
    controller.placement = Some(ShallowWaterPlacement {
        center: Vec2::ZERO,
        cell_size: 1.0,
        datum: -DEPTH,
        blend_cells: 0.0,
    });
    app.insert_resource(controller);
    extra(&mut app);
    while app.plugins_state() == PluginsState::Adding {
        bevy::tasks::tick_global_task_pools_on_main_thread();
        thread::sleep(Duration::from_millis(1));
    }
    app.finish();
    app.cleanup();
    step(&mut app);
    let bed_water = app
        .world()
        .resource::<ShallowWaterController>()
        .bed_water
        .clone();
    app.world_mut().spawn(Readback::texture(bed_water)).observe(
        |event: On<ReadbackComplete>, mut capture: ResMut<Capture>| {
            capture.bytes = event.data.clone();
            capture.serial += 1;
        },
    );
    app
}

/// Steps `frames` frames, then until a fresh readback of the pool has landed.
fn run_for(app: &mut App, frames: u32) {
    let deadline = Instant::now() + Duration::from_secs(120);
    for _ in 0..frames {
        step(app);
        assert!(Instant::now() < deadline, "pool run timed out");
    }
    let serial = app.world().resource::<Capture>().serial;
    while app.world().resource::<Capture>().serial <= serial + 1 {
        step(app);
        assert!(Instant::now() < deadline, "pool readback timed out");
    }
}

/// Water surface height above the bed per cell, row by row.
fn surface(app: &App) -> Vec<f32> {
    let bytes = &app.world().resource::<Capture>().bytes;
    let texel = bytes.len() / (CELLS * CELLS) as usize;
    (0..(CELLS * CELLS) as usize)
        .map(|i| {
            let lane = |l: usize| {
                let o = i * texel + l * 4;
                f32::from_le_bytes(bytes[o..o + 4].try_into().unwrap())
            };
            lane(0) + lane(1)
        })
        .collect()
}

fn at(surface: &[f32], x: u32, z: u32) -> f32 {
    surface[(z * CELLS + x) as usize]
}

/// 2 m by 2 m and 1 m tall, held in place, over four whole cells at the pool centre.
fn slab() -> FloatingBody {
    let hull = Arc::new(voxelize(&Mesh::from(Cuboid::new(2.0, 1.0, 2.0)), 0.25).unwrap());
    let mut body = FloatingBody::new(hull, 2000.0);
    body.estimate_velocity = false;
    body
}

#[test]
#[ignore = "requires a graphics adapter"]
fn a_body_presses_the_water_down_and_feels_its_whole_displacement() {
    let mut app = pool_app();
    let entity = app.world_mut().spawn((slab(), Transform::IDENTITY)).id();
    run_for(&mut app, 600);
    let s = surface(&app);
    let open = at(&s, 8, 8);
    let under = 0.25 * (at(&s, 31, 31) + at(&s, 32, 31) + at(&s, 31, 32) + at(&s, 32, 32));
    // The water under the slab meets its hull, half of it, 0.5 m, below the open water.
    let dent = open - under;
    assert!(
        (dent - 0.5).abs() < 0.02,
        "dent {dent} m, open {open}, under {under}"
    );
    // The 2 m³ it pushed out spread over the 4096 m² pool.
    assert!((open - DEPTH).abs() < 0.01, "open water at {open}");

    // Buoyancy reads the surface plus the head, so it sees the full 2 m³, not the dented water.
    let body = app.world().get::<FloatingBody>(entity).unwrap();
    let lift = RHO * G * 2.0;
    assert!(
        (body.buoyancy.y - lift).abs() < 0.02 * lift,
        "{:?} against {lift}",
        body.buoyancy
    );
}

#[test]
#[ignore = "requires a graphics adapter"]
fn a_towed_body_makes_waves() {
    let mut app = pool_app();
    let mut body = slab();
    body.estimate_velocity = true;
    let entity = app
        .world_mut()
        .spawn((body, Transform::from_xyz(-20.0, 0.0, 0.0)))
        .id();
    run_for(&mut app, 300);
    let still = surface(&app);
    // Tow it 30 m along +X at 3 m/s, then stop.
    for _ in 0..600 {
        let mut transform = app.world_mut().get_mut::<Transform>(entity).unwrap();
        transform.translation.x = (transform.translation.x + 3.0 / 60.0).min(10.0);
        step(&mut app);
    }
    run_for(&mut app, 1);
    let wake = surface(&app);
    // Leave out the dent under where the slab stopped.
    let mut waves = 0.0f32;
    for z in 0..CELLS {
        for x in 0..CELLS {
            let under_slab = (x as f32 - 41.5).abs() < 2.0 && (z as f32 - 31.5).abs() < 2.0;
            if !under_slab {
                waves = waves.max((at(&wake, x, z) - at(&still, x, z)).abs());
            }
        }
    }
    assert!(waves > 0.05, "largest wave {waves} m");
}

/// A 1.2 m crate of 450 kg/m³ dropped from 1 m onto the pool and moved by Rapier with the GPU
/// forces, the way the examples apply them. Returns its submerged share every frame.
fn dropped_crate(frames: u32) -> Vec<f32> {
    use bevy_rapier3d::prelude::*;
    let mut app = pool_app_with(|app| {
        app.add_plugins(RapierPhysicsPlugin::<NoUserData>::default());
        // Undamped water, as in a river or the sea, so waves the crate makes keep moving.
        app.world_mut()
            .resource_mut::<ShallowWaterController>()
            .friction = 0.0;
    });
    app.add_systems(
        Update,
        |settings: Res<FloatingBodySettings>,
         time: Res<Time>,
         mut bodies: Query<(
            &mut FloatingBody,
            &mut ExternalForce,
            &Velocity,
            &GlobalTransform,
        )>| {
            for (mut body, mut external, velocity, transform) in &mut bodies {
                body.linear_velocity = velocity.linear;
                body.angular_velocity = velocity.angular;
                let center = transform.transform_point(body.center_of_mass);
                let (force, torque) = body.compensated(
                    center,
                    velocity.linear,
                    velocity.angular,
                    time.delta_secs(),
                    &settings,
                );
                external.force = force;
                external.torque = torque;
            }
        },
    );
    let hull = Arc::new(voxelize(&Mesh::from(Cuboid::new(1.2, 1.2, 1.2)), 0.2).unwrap());
    let mass = 450.0 * hull.volume();
    let mut body = FloatingBody::new(hull, mass);
    body.estimate_velocity = false;
    let entity = app
        .world_mut()
        .spawn((
            body,
            Transform::from_xyz(0.0, 1.6, 0.0).with_rotation(Quat::from_rotation_y(0.4)),
            RigidBody::Dynamic,
            Collider::cuboid(0.6, 0.6, 0.6),
            ColliderMassProperties::Mass(mass),
            Velocity::default(),
            ExternalForce::default(),
        ))
        .id();
    let mut shares = Vec::new();
    for _ in 0..frames {
        step(&mut app);
        let body = app.world().get::<FloatingBody>(entity).unwrap();
        shares.push(body.submerged_volume / body.hull.volume());
    }
    shares
}

#[test]
#[ignore = "requires a graphics adapter"]
fn a_dropped_body_that_presses_the_water_settles() {
    // A dropped crate floats at its density ratio, 45 percent under, while it presses the water.
    // This pool, with 1 m cells and the forces read back without frame pacing, does not show
    // the instability the pool example had with 0.25 m cells and the head at 0.25 per frame,
    // where the crate bobbed between 2 and 94 percent under; it checks the settled state.
    let shares = dropped_crate(1200);
    let last = &shares[900..];
    let (low, high) = last
        .iter()
        .fold((f32::MAX, f32::MIN), |(lo, hi), &s| (lo.min(s), hi.max(s)));
    assert!(
        low > 0.3 && high < 0.6,
        "over the last 5 s the crate was {low} to {high} under"
    );
}

/// The latest readback of the nested grid's `bed_water`.
#[derive(Resource, Default)]
struct NestedCapture {
    bytes: Vec<u8>,
    serial: u64,
}

#[test]
#[ignore = "requires a graphics adapter"]
fn a_body_presses_the_nested_grid_down_too() {
    // A nested grid of 48 cells a third of a meter wide, 16 m across, around the pool centre.
    const NESTED: u32 = 48;
    let mut app = pool_app_with(|app| {
        let mut images = app.world_mut().resource_mut::<Assets<Image>>();
        let nested = ShallowWaterNestedGrid::new(&mut images, NESTED);
        app.world_mut()
            .resource_mut::<ShallowWaterController>()
            .nested = Some(nested);
        app.init_resource::<NestedCapture>();
    });
    let image = app
        .world()
        .resource::<ShallowWaterController>()
        .nested
        .as_ref()
        .unwrap()
        .images
        .bed_water
        .clone();
    app.world_mut().spawn(Readback::texture(image)).observe(
        |event: On<ReadbackComplete>, mut capture: ResMut<NestedCapture>| {
            capture.bytes = event.data.clone();
            capture.serial += 1;
        },
    );
    app.world_mut().spawn((slab(), Transform::IDENTITY));
    run_for(&mut app, 600);
    let serial = app.world().resource::<NestedCapture>().serial;
    while app.world().resource::<NestedCapture>().serial <= serial + 1 {
        step(&mut app);
    }
    let bytes = &app.world().resource::<NestedCapture>().bytes;
    let row = bytes.len() / NESTED as usize;
    let surface = |x: u32, z: u32| {
        let o = z as usize * row + x as usize * 16;
        let lane =
            |l: usize| f32::from_le_bytes(bytes[o + 4 * l..o + 4 * l + 4].try_into().unwrap());
        lane(0) + lane(1)
    };
    // The slab covers the middle 6 by 6 cells. Its hull sits 0.5 m deep.
    let open = surface(4, 4);
    let under = 0.25 * (surface(23, 23) + surface(24, 23) + surface(23, 24) + surface(24, 24));
    let dent = open - under;
    assert!(
        (dent - 0.5).abs() < 0.05,
        "nested dent {dent} m, open {open}, under {under}"
    );
}
