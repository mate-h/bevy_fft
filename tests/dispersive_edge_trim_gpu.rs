//! Tessendorf's edge trim on the periodic dispersive patch. Run explicitly with a graphics
//! adapter.
use bevy::{
    app::PluginsState,
    prelude::*,
    render::gpu_readback::{Readback, ReadbackComplete},
    time::TimeUpdateStrategy,
    window::ExitCondition,
    winit::WinitPlugin,
};
use bevy_fft::{dispersive::*, fft::FftPlugin};
use std::{
    thread,
    time::{Duration, Instant},
};

const N: u32 = 128;
/// `init_basin` fills the patch to this depth over a flat bed.
const REST_DEPTH: f32 = 4.0;

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

/// A splash 28 cells from the +X edge, run for `frames` frames of 1/60 s, then the surface
/// deviation per cell, row by row.
fn splash(edge_trim_cells: f32, frames: u32) -> Vec<f32> {
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
    .add_plugins((FftPlugin, DispersivePlugin))
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
    let world = app.world_mut();
    let mut controller = world.resource_scope(|world, mut images: Mut<Assets<Image>>| {
        DispersiveController::spawn(&mut world.commands(), &mut images, N)
    });
    controller.tile_world = N as f32;
    controller.edge_trim_cells = edge_trim_cells;
    let state = controller.state.clone();
    world.insert_resource(controller);
    world.spawn(Readback::texture(state)).observe(
        |event: On<ReadbackComplete>, mut capture: ResMut<Capture>| {
            capture.bytes = event.data.clone();
            capture.serial += 1;
        },
    );
    world.flush();

    let deadline = Instant::now() + Duration::from_secs(180);
    // Settle the basin, then splash for one frame.
    for _ in 0..30 {
        step(&mut app);
    }
    {
        let mut controller = app.world_mut().resource_mut::<DispersiveController>();
        let at = Vec2::new(N as f32 - 28.0, N as f32 * 0.5);
        controller.pointer = at;
        controller.pointer_prev = at;
        controller.brush_active = true;
    }
    step(&mut app);
    app.world_mut()
        .resource_mut::<DispersiveController>()
        .brush_active = false;
    for _ in 0..frames {
        step(&mut app);
        assert!(Instant::now() < deadline, "dispersive run timed out");
    }
    let serial = app.world().resource::<Capture>().serial;
    while app.world().resource::<Capture>().serial <= serial + 1 {
        step(&mut app);
        assert!(Instant::now() < deadline, "dispersive readback timed out");
    }
    let bytes = &app.world().resource::<Capture>().bytes;
    let texel = bytes.len() / (N * N) as usize;
    (0..(N * N) as usize)
        .map(|i| {
            f32::from_le_bytes(bytes[i * texel..i * texel + 4].try_into().unwrap()) - REST_DEPTH
        })
        .collect()
}

/// Sum of squared deviation over columns `x0..x1` of every row.
fn energy(eta: &[f32], x0: u32, x1: u32) -> f32 {
    (0..N)
        .flat_map(|y| (x0..x1).map(move |x| (y * N + x) as usize))
        .map(|i| eta[i] * eta[i])
        .sum()
}

#[test]
#[ignore = "requires a graphics adapter"]
fn the_trim_stops_waves_wrapping_around_and_keeps_the_inside() {
    // After 8 s the fastest waves, at √(g h) ≈ 6.3 m/s, have run about 50 m. Columns 14 to 30
    // can only have been reached by crossing the +X edge, 28 m from the splash, and wrapping.
    let frames = 480;
    let periodic = splash(0.0, frames);
    let trimmed = splash(12.0, frames);
    let wrapped_periodic = energy(&periodic, 14, 30);
    let wrapped_trimmed = energy(&trimmed, 14, 30);
    assert!(
        wrapped_periodic > 1e-4,
        "no waves wrapped without the trim: {wrapped_periodic}"
    );
    assert!(
        wrapped_trimmed < 0.1 * wrapped_periodic,
        "trimmed {wrapped_trimmed} against periodic {wrapped_periodic}"
    );
    // Waves that stay inside, between the splash and the far side, are untouched by the trim.
    let inside_periodic = energy(&periodic, 50, 90);
    let inside_trimmed = energy(&trimmed, 50, 90);
    assert!(
        (inside_trimmed / inside_periodic - 1.0).abs() < 0.25,
        "inside {inside_trimmed} against {inside_periodic}"
    );
}
