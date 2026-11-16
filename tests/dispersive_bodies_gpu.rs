//! Floating bodies on a dispersive patch: obstruction, reflection and wakes. Run explicitly with a
//! graphics adapter.
use bevy::{
    app::PluginsState,
    prelude::*,
    render::gpu_readback::{Readback, ReadbackComplete},
    time::TimeUpdateStrategy,
    window::ExitCondition,
    winit::WinitPlugin,
};
use bevy_fft::{bodies::*, dispersive::*, fft::FftPlugin, water::*};
use std::{
    sync::Arc,
    thread,
    time::{Duration, Instant},
};

const N: u32 = 128;
/// `init_basin` fills the patch to this depth over a flat bed.
const REST_DEPTH: f32 = 4.0;
const DT: f32 = 1.0 / 60.0;

#[derive(Resource, Default)]
struct Captures {
    state: Vec<u8>,
    obstruction: Vec<u8>,
    serial: u64,
}

fn step(app: &mut App) {
    app.update();
    assert!(app.should_exit().is_none(), "renderer must remain healthy");
    thread::sleep(Duration::from_millis(1));
}

/// A 128 m patch of 4 m water at the origin in deep mode, with body maps.
fn patch_app() -> App {
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
    .add_plugins((FftPlugin, FloatingBodyPlugin, DispersivePlugin))
    .insert_resource(WaterLodClear)
    .insert_resource(WaterLodSettings {
        resolution: 256,
        slice_count: 1,
        base_texel_size: 1.0,
    })
    .insert_resource(TimeUpdateStrategy::ManualDuration(Duration::from_secs_f32(
        DT,
    )))
    .insert_resource(bevy::render::error_handler::RenderErrorHandler(
        |error, _, _| {
            panic!("GPU validation failed: {error:?}");
        },
    ))
    .init_resource::<Captures>();
    while app.plugins_state() == PluginsState::Adding {
        bevy::tasks::tick_global_task_pools_on_main_thread();
        thread::sleep(Duration::from_millis(1));
    }
    app.finish();
    app.cleanup();
    let world = app.world_mut();
    let mut controller = world.resource_scope(|world, mut images: Mut<Assets<Image>>| {
        let mut controller = DispersiveController::spawn(&mut world.commands(), &mut images, N);
        controller.tile_world = N as f32;
        controller.deep_water_depth = Some(100.0);
        controller.placement = Some(DispersivePlacement {
            datum: 0.0,
            blend_cells: 4.0,
            ..default()
        });
        controller.enable_bodies(&mut images);
        controller
    });
    let state = controller.state.clone();
    let obstruction = controller.body_maps.as_ref().unwrap().obstruction.clone();
    controller.edge_trim_cells = 12.0;
    world.insert_resource(controller);
    world.spawn(Readback::texture(state)).observe(
        |event: On<ReadbackComplete>, mut captures: ResMut<Captures>| {
            captures.state = event.data.clone();
            captures.serial += 1;
        },
    );
    world.spawn(Readback::texture(obstruction)).observe(
        |event: On<ReadbackComplete>, mut captures: ResMut<Captures>| {
            captures.obstruction = event.data.clone();
        },
    );
    world.flush();
    for _ in 0..30 {
        step(&mut app);
    }
    app
}

fn settle(app: &mut App) {
    let deadline = Instant::now() + Duration::from_secs(60);
    let serial = app.world().resource::<Captures>().serial;
    while app.world().resource::<Captures>().serial <= serial + 1 {
        step(app);
        assert!(Instant::now() < deadline, "readback timed out");
    }
}

/// Surface deviation per cell, row by row.
fn eta(app: &App) -> Vec<f32> {
    let bytes = &app.world().resource::<Captures>().state;
    let texel = bytes.len() / (N * N) as usize;
    (0..(N * N) as usize)
        .map(|i| {
            f32::from_le_bytes(bytes[i * texel..i * texel + 4].try_into().unwrap()) - REST_DEPTH
        })
        .collect()
}

fn obstruction(app: &App) -> Vec<f32> {
    let bytes = &app.world().resource::<Captures>().obstruction;
    let texel = bytes.len() / (N * N) as usize;
    (0..(N * N) as usize)
        .map(|i| f32::from_le_bytes(bytes[i * texel..i * texel + 4].try_into().unwrap()))
        .collect()
}

/// Cell of world `xz` on the patch centred on the origin with 1 m cells.
fn cell(xz: Vec2) -> usize {
    let c = (xz + N as f32 * 0.5).floor().as_uvec2();
    (c.y * N + c.x) as usize
}

/// A 4 m by 2 m by 8 m barge, half under water.
fn barge() -> FloatingBody {
    let hull = Arc::new(voxelize(&Mesh::from(Cuboid::new(4.0, 2.0, 8.0)), 0.25).unwrap());
    let mut body = FloatingBody::new(hull, 8000.0);
    body.estimate_velocity = false;
    body
}

fn energy(eta: &[f32], cells: impl Iterator<Item = (u32, u32)>) -> f32 {
    cells.map(|(x, y)| eta[(y * N + x) as usize].powi(2)).sum()
}

/// Presses the brush down at `x` meters on the X axis for a quarter second.
fn splash(app: &mut App, x: f32) {
    for _ in 0..15 {
        let mut controller = app.world_mut().resource_mut::<DispersiveController>();
        let at = controller.world_to_cells(Vec2::new(x, 0.0));
        controller.pointer = at;
        controller.pointer_prev = at;
        controller.brush_radius = 3.0;
        // 1.4 m of head, so the splash's waves stand well above the float rounding of the dent.
        controller.brush_strength = 2.8;
        controller.brush_active = true;
        step(app);
    }
    app.world_mut()
        .resource_mut::<DispersiveController>()
        .brush_active = false;
}

#[test]
#[ignore = "requires a graphics adapter"]
fn a_hull_at_the_waterline_obstructs_its_cells() {
    let mut app = patch_app();
    app.world_mut().spawn((barge(), Transform::IDENTITY));
    for _ in 0..60 {
        step(&mut app);
    }
    settle(&mut app);
    let o = obstruction(&app);
    assert!(
        o[cell(Vec2::ZERO)] < 0.05,
        "under the hull {}",
        o[cell(Vec2::ZERO)]
    );
    assert!(o[cell(Vec2::new(1.5, 3.5))] < 0.05);
    assert!(
        o[cell(Vec2::new(10.0, 0.0))] > 0.95,
        "open water {}",
        o[cell(Vec2::new(10.0, 0.0))]
    );
    assert!(o[cell(Vec2::new(0.0, 10.0))] > 0.95);
}

#[test]
#[ignore = "requires a graphics adapter"]
fn a_still_hull_holds_its_dent_without_radiating() {
    // Band-limiting the bare surface each step smeared the dent onto the open water next to the
    // hull, the obstruction put it back under the hull, and the waves around it grew past 1 m in
    // 20 s.
    let mut app = patch_app();
    app.world_mut().spawn((wall(), Transform::IDENTITY));
    for _ in 0..1200 {
        step(&mut app);
    }
    settle(&mut app);
    let e = eta(&app);
    assert!(
        (e[cell(Vec2::ZERO)] + 1.0).abs() < 0.02,
        "dent {}",
        e[cell(Vec2::ZERO)]
    );
    let waves = (0..N * N)
        .filter(|&i| ((i % N) as i32 - 64).abs() > 3 || ((i / N) as i32 - 64).abs() > 12)
        .map(|i| e[i as usize].abs())
        .fold(0.0f32, f32::max);
    assert!(waves < 0.05, "waves around the hull reach {waves} m");
}

/// A 4 m wide, 24 m long barge across the X axis, half under water.
fn wall() -> FloatingBody {
    let hull = Arc::new(voxelize(&Mesh::from(Cuboid::new(4.0, 2.0, 24.0)), 0.25).unwrap());
    let mut body = FloatingBody::new(hull, 24000.0);
    body.estimate_velocity = false;
    body
}

/// The surface 15 s after a splash at x = 25 m, or at the same time without one. Waves about 10 m
/// long, at about 2 m/s, take that long to pass the barge.
fn after_splash(with_barge: bool, with_splash: bool) -> Vec<f32> {
    let mut app = patch_app();
    if with_barge {
        app.world_mut().spawn((wall(), Transform::IDENTITY));
    }
    // Let the barge settle into its dent.
    for _ in 0..300 {
        step(&mut app);
    }
    if with_splash {
        splash(&mut app, 25.0);
    } else {
        for _ in 0..15 {
            step(&mut app);
        }
    }
    for _ in 0..900 {
        step(&mut app);
    }
    // Hold the patch while the readback catches up, so both runs stop at the same step and the
    // barge's own settling cancels exactly.
    app.world_mut()
        .resource_mut::<DispersiveController>()
        .paused = true;
    settle(&mut app);
    eta(&app)
}

/// Energy of the splash's waves behind the barge (x from -50 to -4 m) and beyond the splash (x
/// from 24 to 50 m), where the waves the barge reflects have arrived by then, along the band
/// |z| < 12 m it spans. Subtracting the run without the splash leaves out the barge's own
/// settling.
fn shadow_and_front(with_barge: bool) -> (f32, f32) {
    let with = after_splash(with_barge, true);
    let without = after_splash(with_barge, false);
    let waves: Vec<f32> = with.iter().zip(&without).map(|(a, b)| a - b).collect();
    let band = |x0: f32, x1: f32| {
        let (x0, x1) = ((x0 + 64.0) as u32, (x1 + 64.0) as u32);
        (x0..x1).flat_map(move |x| (52..76).map(move |y| (x, y)))
    };
    (
        energy(&waves, band(-50.0, -4.0)),
        energy(&waves, band(24.0, 50.0)),
    )
}

#[test]
#[ignore = "requires a graphics adapter"]
fn a_hull_reflects_waves_and_shadows_the_water_behind_it() {
    // A 4 m wide barge, 1 m deep, against waves 10 to 20 m long leaves about half the energy
    // behind it and sends about a quarter more back past the splash. The Airy step is global and
    // long waves bend around the barge's ends, so some energy still reaches the water behind it,
    // as in Gilligan's eWave.
    let (open_behind, open_front) = shadow_and_front(false);
    let (barge_behind, barge_front) = shadow_and_front(true);
    assert!(
        barge_behind < 0.7 * open_behind,
        "behind the barge {barge_behind} against {open_behind} without it, front {barge_front} against {open_front}"
    );
    assert!(
        barge_front > 1.1 * open_front,
        "in front of the barge {barge_front} against {open_front} without it"
    );
}

#[test]
#[ignore = "requires a graphics adapter"]
fn a_towed_body_trails_a_wake_in_deep_mode() {
    let mut app = patch_app();
    let mut body = barge();
    body.estimate_velocity = true;
    let entity = app
        .world_mut()
        .spawn((body, Transform::from_xyz(-40.0, 0.0, 0.0)))
        .id();
    for _ in 0..60 {
        step(&mut app);
    }
    // Tow it 40 m along +X at 4 m/s.
    for _ in 0..600 {
        let mut transform = app.world_mut().get_mut::<Transform>(entity).unwrap();
        transform.translation.x += 4.0 * DT;
        step(&mut app);
    }
    settle(&mut app);
    let e = eta(&app);
    let at = app
        .world()
        .get::<Transform>(entity)
        .unwrap()
        .translation
        .xz();
    // Waves well away from the barge, behind it along its path.
    let wake = (0..N * N)
        .filter(|&i| {
            let xz = Vec2::new((i % N) as f32, (i / N) as f32) - N as f32 * 0.5 + 0.5;
            xz.distance(at) > 10.0 && xz.x < at.x && xz.x > -40.0
        })
        .map(|i| e[i as usize].abs())
        .fold(0.0f32, f32::max);
    assert!(wake > 0.05, "largest wake wave {wake} m");
    assert!(wake < 2.0, "wake wave {wake} m");
}
