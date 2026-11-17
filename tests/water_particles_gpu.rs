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
use bevy_fft::{particles::*, water::*};
use std::{
    thread,
    time::{Duration, Instant},
};

const G: f32 = 9.81;
/// The LOD data's one slice: 64 texels of 0.5 m around the origin.
const RES: u32 = 64;
const TEXEL: f32 = 0.5;

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
    // Still water at height 0: no writer fills the LOD data, so it starts each frame empty.
    app.insert_resource(WaterLodClear)
        .insert_resource(WaterLodSettings {
            slice_count: 1,
            resolution: RES,
            base_texel_size: TEXEL,
        });
    particle_app_with(app, settings)
}

fn particle_app_with(mut app: App, settings: WaterParticleSettings) -> App {
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
        collide: false,
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
        collide: false,
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

#[derive(Resource, Default)]
struct FoamCapture {
    bytes: Vec<u8>,
    serial: u32,
}

#[test]
#[ignore = "requires a graphics adapter"]
fn a_particle_lands_on_still_water_and_foams_its_texel() {
    let mut app = particle_app(WaterParticleSettings {
        capacity: 64,
        drag: 0.0,
        ..default()
    });
    let drop_at = Vec2::new(1.3, 2.7);
    spawn(
        &mut app,
        &[WaterParticle {
            position: Vec3::new(drop_at.x, 1.0, drop_at.y),
            size: TEXEL,
            velocity: Vec3::ZERO,
            life: 100.0,
        }],
    );
    // A drop from 1 m lands after 0.45 s, 27 frames. Read the foam a few frames after.
    for _ in 0..32 {
        step(&mut app);
    }
    let foam_state = app.world().resource::<WaterLodData>().foam_state.clone();
    app.init_resource::<FoamCapture>();
    app.world_mut()
        .spawn(Readback::texture(foam_state))
        .observe(|event: On<ReadbackComplete>, mut c: ResMut<FoamCapture>| {
            c.bytes = event.data.clone();
            c.serial += 1;
        });
    let deadline = Instant::now() + Duration::from_secs(30);
    while app.world().resource::<FoamCapture>().serial == 0 {
        step(&mut app);
        assert!(Instant::now() < deadline, "foam readback timed out");
    }
    let (particles, [_, spawned, died, alive]) = read(&mut app);
    assert_eq!((spawned, died, alive), (1, 1, 0));
    let p = particles[0];
    assert!(p.life <= 0.0, "the particle is still alive: {p:?}");
    // It falls about 7 cm per frame when it reaches the surface.
    assert!(
        p.position.y <= 0.0 && p.position.y > -0.1,
        "it died at {}",
        p.position
    );

    let bytes = &app.world().resource::<FoamCapture>().bytes;
    let row = bytes.len() / RES as usize;
    let active = |x: u32, y: u32| {
        let o = y as usize * row + x as usize * 8;
        f32::from_le_bytes(bytes[o..o + 4].try_into().unwrap())
    };
    let lod = app.world().resource::<WaterLodData>().slices;
    let centre = lod.slices[0].truncate().truncate();
    let texel = ((drop_at - centre) / TEXEL + 0.5 * RES as f32)
        .floor()
        .as_uvec2();
    assert!(
        active(texel.x, texel.y) > 0.5,
        "texel {texel} holds {} active foam",
        active(texel.x, texel.y)
    );
    for y in 0..RES {
        for x in 0..RES {
            if UVec2::new(x, y) != texel {
                assert_eq!(active(x, y), 0.0, "foam at ({x}, {y})");
            }
        }
    }
}

/// Spawned count after `frames` frames of crest spray at `rate` over a breaking patch covering
/// world `x` 8..24 and `z` -4..4, which runs across the edge of slice 0 at x = 16. Without
/// `breaking` the water is still.
fn crest_spray(rate: f32, frames: u32, breaking: bool) -> (u32, f32) {
    let mut app = App::new();
    // Two slices of 64 texels, 0.5 m and 1 m wide, around the origin. No writer touches them, so
    // the breaking uploaded here stays.
    app.insert_resource(WaterLodSettings {
        slice_count: 2,
        resolution: RES,
        base_texel_size: TEXEL,
    });
    let mut app = particle_app_with(
        app,
        WaterParticleSettings {
            collide: false,
            ..default()
        },
    );
    let lod = app.world().resource::<WaterLodData>().clone();
    if breaking {
        let mut images = app.world_mut().resource_mut::<Assets<Image>>();
        let mut foam = images.get_mut(&lod.foam).unwrap();
        let data = foam.data.as_mut().unwrap();
        for slice in 0..2u32 {
            let texel = TEXEL * (1 << slice) as f32;
            let centre = lod.slices.slices[slice as usize].truncate().truncate();
            for y in 0..RES {
                for x in 0..RES {
                    let p =
                        centre + (Vec2::new(x as f32, y as f32) + 0.5 - 0.5 * RES as f32) * texel;
                    if (8.0..24.0).contains(&p.x) && (-4.0..4.0).contains(&p.y) {
                        let o = (((slice * RES + y) * RES + x) * 8) as usize;
                        // 1.0 as f16.
                        data[o..o + 2].copy_from_slice(&0x3c00u16.to_le_bytes());
                    }
                }
            }
        }
    }
    for _ in 0..5 {
        step(&mut app);
    }
    {
        let mut settings = app.world_mut().resource_mut::<WaterParticleSettings>();
        settings.crest_spray.rate = rate;
        settings.crest_spray.cap = 1 << 20;
    }
    for _ in 0..frames {
        step(&mut app);
    }
    let (_, [_, spawned, ..]) = read(&mut app);
    // The slices sit where the test placed the patch only if they centre on the origin.
    let centre = lod.slices.slices[0].truncate().truncate();
    (spawned, centre.length())
}

#[test]
#[ignore = "requires a graphics adapter"]
fn breaking_crests_spray_in_proportion_to_their_area() {
    let (rate, frames) = (3000.0, 40);
    let (still, _) = crest_spray(rate, frames, false);
    assert_eq!(still, 0, "still water sprayed {still} particles");
    let (spawned, offset) = crest_spray(rate, frames, true);
    assert!(offset < 1e-3, "the slices are off the origin by {offset} m");
    // 128 m² of full breaking on a surface that does not rise emits at half the rate. The
    // emitter runs for `frames` frames, give or take one.
    let per_frame = rate * 128.0 / 60.0 * 0.5;
    let low = (frames - 1) as f32 * per_frame * 0.98;
    let high = (frames + 1) as f32 * per_frame * 1.02;
    assert!(
        (low..high).contains(&(spawned as f32)),
        "{spawned} particles, expected {} per frame over {frames} frames",
        per_frame
    );
}
