//! Floating bodies splash into the water particles. Run explicitly with a graphics adapter.
use bevy::{
    app::PluginsState,
    prelude::*,
    render::gpu_readback::{Readback, ReadbackComplete},
    time::TimeUpdateStrategy,
    window::ExitCondition,
    winit::WinitPlugin,
};
use bevy_fft::{bodies::*, particles::*, water::*};
use std::{
    sync::Arc,
    thread,
    time::{Duration, Instant},
};

#[derive(Resource, Default)]
struct Capture {
    state: Vec<u8>,
    serial: u32,
}

fn step(app: &mut App) {
    app.update();
    assert!(app.should_exit().is_none(), "renderer must remain healthy");
    thread::sleep(Duration::from_millis(1));
}

/// Particles spawned after a 1 m cube sits half in still water for 40 frames, reporting
/// `velocity`. Nothing moves it: the velocity only feeds the force pass.
fn splashes(velocity: Vec3) -> u32 {
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
    .insert_resource(WaterLodClear)
    .insert_resource(WaterLodSettings {
        resolution: 64,
        slice_count: 1,
        base_texel_size: 0.25,
    })
    .add_plugins((FloatingBodyPlugin, WaterParticlesPlugin))
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
    for _ in 0..20 {
        step(&mut app);
    }
    let hull = Arc::new(voxelize(&Mesh::from(Cuboid::new(1.0, 1.0, 1.0)), 0.25).unwrap());
    let mut body = FloatingBody::new(hull, 500.0);
    body.estimate_velocity = false;
    body.linear_velocity = velocity;
    app.world_mut()
        .spawn((body, Transform::from_xyz(0.0, 0.0, 0.0)));
    for _ in 0..40 {
        step(&mut app);
    }
    let state = app.world().resource::<WaterParticles>().state.clone();
    app.world_mut().spawn(Readback::buffer(state)).observe(
        |event: On<ReadbackComplete>, mut c: ResMut<Capture>| {
            c.state = event.data.clone();
            c.serial += 1;
        },
    );
    let deadline = Instant::now() + Duration::from_secs(30);
    while app.world().resource::<Capture>().serial == 0 {
        step(&mut app);
        assert!(Instant::now() < deadline, "readback timed out");
    }
    bytemuck::cast_slice::<u8, u32>(&app.world().resource::<Capture>().state)[1]
}

#[test]
#[ignore = "requires a graphics adapter"]
fn a_body_entering_the_water_splashes_and_one_at_rest_does_not() {
    let resting = splashes(Vec3::ZERO);
    assert_eq!(resting, 0, "a body at rest threw up {resting} particles");
    // The bottom face sits a voxel below the surface, so the waterline patches are the sides,
    // which a body falling straight down slides along. Falling with some sideways speed drives
    // one side into the water at 2 m/s, 1.5 above the threshold, over its 0.5 m² of wet area:
    // 2000 · 0.5 · 1.5 = 1500 particles a second, 25 a frame.
    let dropped = splashes(Vec3::new(2.0, -3.0, 0.0));
    assert!(
        (400..2000).contains(&dropped),
        "a body entering the water threw up {dropped} particles"
    );
}
