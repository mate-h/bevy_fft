//! Moving objects stir shallow water, and the stirred water carries floating bodies. Run
//! explicitly with a graphics adapter.
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
const DEPTH: f32 = 1.0;

#[derive(Resource, Default)]
struct Capture {
    bytes: Vec<u8>,
    serial: u64,
}

/// Where the mover is at time `t` in seconds.
#[derive(Resource, Clone, Copy)]
struct Path {
    start: Vec3,
    velocity: Vec3,
    /// Seconds after which it stops.
    until: f32,
}

fn follow_path(
    time: Res<Time>,
    path: Res<Path>,
    mut movers: Query<&mut Transform, With<ShallowWaterMover>>,
) {
    let t = time.elapsed_secs().min(path.until);
    for mut transform in &mut movers {
        transform.translation = path.start + path.velocity * t;
    }
}

fn step(app: &mut App) {
    app.update();
    assert!(app.should_exit().is_none(), "renderer must remain healthy");
    thread::sleep(Duration::from_millis(1));
}

/// A walled pool of still water `DEPTH` deep with its surface at world height 0, 1 m cells, a
/// mover on `path` and, with `bodies`, the floating body plugin.
fn pool_app(path: Path, bodies: bool) -> App {
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
    .add_plugins(ShallowWaterPlugin)
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
    .insert_resource(path)
    .add_systems(Update, follow_path)
    .init_resource::<Capture>();
    if bodies {
        app.add_plugins(FloatingBodyPlugin);
    }
    let mut images = app.world_mut().resource_mut::<Assets<Image>>();
    let mut controller = ShallowWaterController::new(&mut images, CELLS, CELLS);
    let n = (CELLS * CELLS) as usize;
    controller.bed_source = images.add(bed_source_image(CELLS, CELLS, &vec![0.0; n]));
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
    while app.plugins_state() == PluginsState::Adding {
        bevy::tasks::tick_global_task_pools_on_main_thread();
        thread::sleep(Duration::from_millis(1));
    }
    app.finish();
    app.cleanup();
    step(&mut app);
    app.world_mut().spawn((
        ShallowWaterMover {
            radius: 2.0,
            depth: 0.5,
            ..default()
        },
        Transform::from_translation(path.start),
    ));
    let velocity = app
        .world()
        .resource::<ShallowWaterController>()
        .velocity
        .clone();
    app.world_mut().spawn(Readback::texture(velocity)).observe(
        |event: On<ReadbackComplete>, mut capture: ResMut<Capture>| {
            capture.bytes = event.data.clone();
            capture.serial += 1;
        },
    );
    app
}

/// Steps `frames` frames, then until a fresh readback has landed, and returns the cell
/// velocities, row by row.
fn run_for(app: &mut App, frames: u32) -> Vec<Vec2> {
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
    let bytes = &app.world().resource::<Capture>().bytes;
    let texel = bytes.len() / (CELLS * CELLS) as usize;
    (0..(CELLS * CELLS) as usize)
        .map(|i| {
            let lane = |l: usize| {
                let o = i * texel + l * 4;
                f32::from_le_bytes(bytes[o..o + 4].try_into().unwrap())
            };
            Vec2::new(lane(0), lane(1))
        })
        .collect()
}

/// Mean velocity over the cells within 2 m of world `xz`.
fn mean_near(velocities: &[Vec2], xz: Vec2) -> Vec2 {
    let mut sum = Vec2::ZERO;
    let mut count = 0.0;
    for z in 0..CELLS {
        for x in 0..CELLS {
            let center = Vec2::new(x as f32, z as f32) + 0.5 - 0.5 * CELLS as f32;
            if center.distance(xz) <= 2.0 {
                sum += velocities[(z * CELLS + x) as usize];
                count += 1.0;
            }
        }
    }
    sum / count
}

fn fastest(velocities: &[Vec2]) -> f32 {
    velocities.iter().map(|v| v.length()).fold(0.0, f32::max)
}

#[test]
#[ignore = "requires a graphics adapter"]
fn an_object_stirs_the_water_only_while_it_moves_through_it() {
    // Wading along +x at 1.5 m/s for two seconds, then standing still.
    let wading = Path {
        start: Vec3::new(-10.0, 0.0, 0.0),
        velocity: Vec3::new(1.5, 0.0, 0.0),
        until: 2.0,
    };
    let mut app = pool_app(wading, false);
    let moving = run_for(&mut app, 100);
    let behind = mean_near(&moving, Vec2::new(-8.5, 0.0));
    assert!(
        behind.x > 0.2 && behind.x.abs() > 4.0 * behind.y.abs(),
        "the water behind a wading object moves along with it at {behind}"
    );

    // Standing in the water from the start leaves it still.
    let standing = Path {
        until: 0.0,
        ..wading
    };
    let still = fastest(&run_for(&mut pool_app(standing, false), 100));
    assert!(
        still < 1e-4,
        "an object at rest stirred the water to {still} m/s"
    );

    // Moving above the water, its lowest point 0.5 m clear of the surface, leaves it still too.
    let flying = Path {
        start: Vec3::new(-10.0, 1.0, 0.0),
        ..wading
    };
    let dry = fastest(&run_for(&mut pool_app(flying, false), 100));
    assert!(
        dry < 1e-4,
        "an object above the water stirred it to {dry} m/s"
    );
}

#[test]
#[ignore = "requires a graphics adapter"]
fn a_floating_body_drifts_with_the_water_an_object_pushes() {
    // An object wades toward a crate at rest and stops 4 m short of it. The water it pushed
    // carries on and drags the crate along +x.
    let path = Path {
        start: Vec3::new(-14.0, 0.0, 0.0),
        velocity: Vec3::new(1.5, 0.0, 0.0),
        until: 6.0,
    };
    let mut app = pool_app(path, true);
    let hull = Arc::new(voxelize(&Mesh::from(Cuboid::new(1.0, 0.5, 1.0)), 0.25).unwrap());
    let mut body = FloatingBody::new(hull, 500.0);
    body.estimate_velocity = false;
    let crate_at = app
        .world_mut()
        .spawn((body, Transform::from_xyz(0.0, 0.0, 0.0)))
        .id();
    let mut strongest = 0.0f32;
    let mut sideways = 0.0f32;
    for _ in 0..420 {
        step(&mut app);
        let force = app.world().get::<FloatingBody>(crate_at).unwrap().drag;
        strongest = strongest.max(force.x);
        sideways = sideways.max(force.z.abs());
    }
    assert!(
        strongest > 1.0 && strongest > 4.0 * sideways,
        "the pushed water dragged the crate with at most {strongest} N along x and {sideways} N across"
    );
}
