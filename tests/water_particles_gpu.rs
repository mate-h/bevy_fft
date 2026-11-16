//! The water particle pool on the GPU: ballistic arcs and the spawn and death counts. Run
//! explicitly with a graphics adapter.
use bevy::{
    app::PluginsState,
    prelude::*,
    render::gpu_readback::{Readback, ReadbackComplete},
    time::TimeUpdateStrategy,
    window::ExitCondition,
    winit::WinitPlugin,
};
use bevy_fft::particles::*;
use std::{
    thread,
    time::{Duration, Instant},
};

const G: f32 = 9.81;

#[derive(Resource, Default)]
struct Capture {
    particles: Vec<u8>,
    state: Vec<u8>,
    serial: u32,
}

fn step(app: &mut App) {
    app.update();
    assert!(app.should_exit().is_none(), "renderer must remain healthy");
    thread::sleep(Duration::from_millis(1));
}

fn particle_app(settings: WaterParticleSettings) -> App {
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
    .insert_resource(settings)
    .add_plugins(WaterParticlesPlugin)
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
    // Let the pipelines compile before anything spawns.
    for _ in 0..30 {
        step(&mut app);
    }
    app
}

/// Spawns `batch` on the next frame.
fn spawn(app: &mut App, batch: &[WaterParticle]) {
    app.world_mut()
        .resource_mut::<WaterParticleSpawns>()
        .0
        .extend_from_slice(batch);
}

/// Freezes the pool and reads it back with its counters.
fn read(app: &mut App) -> (Vec<WaterParticle>, [u32; 4]) {
    app.world_mut()
        .resource_mut::<WaterParticleSettings>()
        .enabled = false;
    let pool = app.world().resource::<WaterParticles>().clone();
    app.world_mut()
        .spawn(Readback::buffer(pool.particles.clone()))
        .observe(|event: On<ReadbackComplete>, mut c: ResMut<Capture>| {
            c.particles = event.data.clone();
            c.serial |= 1;
        });
    app.world_mut()
        .spawn(Readback::buffer(pool.state.clone()))
        .observe(|event: On<ReadbackComplete>, mut c: ResMut<Capture>| {
            c.state = event.data.clone();
            c.serial |= 2;
        });
    let deadline = Instant::now() + Duration::from_secs(30);
    while app.world().resource::<Capture>().serial != 3 {
        step(app);
        assert!(Instant::now() < deadline, "readback timed out");
    }
    let capture = app.world().resource::<Capture>();
    let particles = bytemuck::cast_slice::<u8, WaterParticle>(&capture.particles).to_vec();
    let state: [u32; 4] = bytemuck::cast_slice::<u8, u32>(&capture.state)[..4]
        .try_into()
        .unwrap();
    (particles, state)
}

#[test]
#[ignore = "requires a graphics adapter"]
fn particles_without_drag_follow_ballistic_arcs() {
    let mut app = particle_app(WaterParticleSettings {
        capacity: 256,
        drag: 0.0,
        ..default()
    });
    let launches = [
        (Vec3::new(0.0, 1.0, 0.0), Vec3::new(3.0, 8.0, 0.0)),
        (Vec3::new(5.0, 0.0, -2.0), Vec3::new(-1.0, 4.0, 2.5)),
        (Vec3::new(-3.0, 2.0, 1.0), Vec3::new(0.0, 12.0, -4.0)),
    ];
    let batch: Vec<WaterParticle> = launches
        .iter()
        .map(|&(position, velocity)| WaterParticle {
            position,
            size: 0.1,
            velocity,
            life: 100.0,
        })
        .collect();
    spawn(&mut app, &batch);
    for _ in 0..50 {
        step(&mut app);
    }
    let (particles, _) = read(&mut app);
    for (k, &(p0, v0)) in launches.iter().enumerate() {
        let p = particles[k];
        // The time since launch follows from the vertical speed lost to gravity.
        let t = (v0.y - p.velocity.y) / G;
        assert!(t > 0.5, "particle {k} only flew {t} s");
        let expected = p0 + v0 * t + 0.5 * Vec3::new(0.0, -G, 0.0) * t * t;
        assert!(
            (p.position - expected).length() < 1e-3,
            "particle {k} at {} against {expected} after {t} s",
            p.position
        );
        assert!(
            (p.velocity.xz() - v0.xz()).length() < 1e-5,
            "particle {k} changed its horizontal speed to {}",
            p.velocity
        );
        assert!((p.life - (100.0 - t)).abs() < 1e-3, "life {}", p.life);
    }
}

#[test]
#[ignore = "requires a graphics adapter"]
fn spawned_minus_died_is_alive() {
    let mut app = particle_app(WaterParticleSettings {
        capacity: 64,
        ..default()
    });
    // Five frames of 40 long-lived particles overrun the 64 slots, so the ring overwrites live
    // ones. Short-lived ones in between die of age.
    for frame in 0..5 {
        let life = if frame % 2 == 0 { 100.0 } else { 0.2 };
        let batch = vec![
            WaterParticle {
                position: Vec3::ZERO,
                size: 0.1,
                velocity: Vec3::Y,
                life,
            };
            40
        ];
        spawn(&mut app, &batch);
        step(&mut app);
    }
    for _ in 0..30 {
        step(&mut app);
    }
    let (particles, [cursor, spawned, died, alive]) = read(&mut app);
    assert_eq!(cursor, 200);
    assert_eq!(spawned, 200);
    assert_eq!(
        spawned - died,
        alive,
        "spawned {spawned}, died {died}, alive {alive}"
    );
    let live = particles.iter().filter(|p| p.life > 0.0).count() as u32;
    assert_eq!(live, alive);
    // The last 64 spawns hold the pool: 40 long-lived from the fifth frame and 24 short-lived
    // from the fourth, which have all died since.
    assert_eq!(alive, 40);
}
