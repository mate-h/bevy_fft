//! `ShallowWaterPlugin` with a nested grid that follows a moving point. Run explicitly with a
//! graphics adapter.
use bevy::{
    app::PluginsState,
    prelude::*,
    render::gpu_readback::{Readback, ReadbackComplete},
    time::TimeUpdateStrategy,
    window::ExitCondition,
    winit::WinitPlugin,
};
use bevy_fft::{shallow_water::*, water::*};
use std::{
    thread,
    time::{Duration, Instant},
};

const CELLS: u32 = 96;
const CHILD: u32 = 96;
const LEVEL: f32 = 2.0;

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

fn bed(x: f32, z: f32) -> f32 {
    0.6 + 0.3 * (x / 9.0).sin() * (z / 13.0).cos()
}

#[test]
#[ignore = "requires a graphics adapter"]
fn a_nested_grid_follows_its_point_and_draws_over_its_parent() {
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
    // The grid spans -48..48 m. The bed is sampled at cell centres.
    let heights: Vec<f32> = (0..CELLS * CELLS)
        .map(|i| bed((i % CELLS) as f32 - 47.5, (i / CELLS) as f32 - 47.5))
        .collect();
    controller.bed_source = images.add(bed_source_image(CELLS, CELLS, &heights));
    controller.preset_index = SHALLOW_WATER_PRESET_BED_SOURCE;
    controller.fill_level = LEVEL;
    controller.gravity = 9.81;
    controller.dt = 1.0 / 30.0;
    controller.placement = Some(ShallowWaterPlacement {
        center: Vec2::ZERO,
        cell_size: 1.0,
        datum: -LEVEL,
        blend_cells: 0.0,
    });
    let mut nested = ShallowWaterNestedGrid::new(&mut images, CHILD);
    nested.follow = Vec2::new(-20.0, 0.0);
    controller.nested = Some(nested);
    app.insert_resource(controller);
    while app.plugins_state() == PluginsState::Adding {
        bevy::tasks::tick_global_task_pools_on_main_thread();
        thread::sleep(Duration::from_millis(1));
    }
    app.finish();
    app.cleanup();

    let deadline = Instant::now() + Duration::from_secs(120);
    for frame in 0..120 {
        // The point walks 0.25 m per frame along +x.
        app.world_mut()
            .resource_mut::<ShallowWaterController>()
            .nested
            .as_mut()
            .unwrap()
            .follow = Vec2::new(-20.0 + 0.25 * frame as f32, 0.0);
        step(&mut app);
        assert!(Instant::now() < deadline, "run timed out");
    }
    let controller = app.world().resource::<ShallowWaterController>();
    let offset = controller.nest_offset().unwrap();
    // Centred on x = 9.75, so the 32 parent cells start at cell 42 along x and 32 along z.
    assert_eq!(offset, IVec2::new(42, 32));
    let child_bed = controller.nested.as_ref().unwrap().images.bed_water.clone();

    let mut heightfields: Vec<(i32, Vec2, f32)> = app
        .world_mut()
        .query::<&WaterHeightfield>()
        .iter(app.world())
        .map(|h| (h.order, h.center, h.cell_size))
        .collect();
    heightfields.sort_by_key(|h| h.0);
    assert_eq!(heightfields.len(), 2, "{heightfields:?}");
    assert_eq!(heightfields[0], (0, Vec2::ZERO, 1.0));
    assert_eq!(heightfields[1].0, 1);
    assert!((heightfields[1].1 - Vec2::new(10.0, 0.0)).length() < 1e-4);
    assert!((heightfields[1].2 - 1.0 / 3.0).abs() < 1e-6);

    app.world_mut()
        .resource_mut::<ShallowWaterController>()
        .paused = true;
    app.world_mut().spawn(Readback::texture(child_bed)).observe(
        |event: On<ReadbackComplete>, mut capture: ResMut<Capture>| {
            capture.bytes = event.data.clone();
            capture.serial += 1;
        },
    );
    while app.world().resource::<Capture>().serial < 2 {
        step(&mut app);
        assert!(Instant::now() < deadline, "readback timed out");
    }
    let bytes = &app.world().resource::<Capture>().bytes;
    let row = bytes.len() / CHILD as usize;
    let lane = |o: usize| f32::from_le_bytes(bytes[o..o + 4].try_into().unwrap());
    let mut worst = 0.0f32;
    for j in 0..CHILD as usize {
        for i in 0..CHILD as usize {
            let o = j * row + i * 16;
            let (b, h) = (lane(o), lane(o + 4));
            // The child's cells sit a third of a parent cell apart from the offset's corner.
            let x = -48.0 + offset.x as f32 + (i as f32 + 0.5) / 3.0;
            let z = -48.0 + offset.y as f32 + (j as f32 + 0.5) / 3.0;
            // The child interpolates the parent's bed samples, so it follows the bed closely.
            assert!((b - bed(x, z)).abs() < 0.01, "bed {b} at ({x}, {z})");
            worst = worst.max((b + h - LEVEL).abs());
        }
    }
    assert!(
        worst < 1e-3,
        "the nested grid's still water moved {worst} m"
    );
}
