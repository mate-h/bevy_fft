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
    controller.nested = vec![nested];
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
            .nested[0]
            .follow = Vec2::new(-20.0 + 0.25 * frame as f32, 0.0);
        step(&mut app);
        assert!(Instant::now() < deadline, "run timed out");
    }
    let controller = app.world().resource::<ShallowWaterController>();
    let offset = controller.nest_offset().unwrap();
    // Centred on x = 9.75, so the 32 parent cells start at cell 42 along x and 32 along z.
    assert_eq!(offset, IVec2::new(42, 32));
    let child_bed = controller.nested[0].images.bed_water.clone();

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

/// Reads `image` back after the next frames.
fn read_back(app: &mut App, image: Handle<Image>, deadline: Instant) -> Vec<u8> {
    app.world_mut().insert_resource(Capture::default());
    let entity = app
        .world_mut()
        .spawn(Readback::texture(image))
        .observe(
            |event: On<ReadbackComplete>, mut capture: ResMut<Capture>| {
                capture.bytes = event.data.clone();
                capture.serial += 1;
            },
        )
        .id();
    while app.world().resource::<Capture>().serial < 2 {
        step(app);
        assert!(Instant::now() < deadline, "readback timed out");
    }
    app.world_mut().despawn(entity);
    std::mem::take(&mut app.world_mut().resource_mut::<Capture>().bytes)
}

/// The worst bed error against `bed` and the worst still water error in a nested grid's
/// `bed_water` readback, for a grid of `cells` cells of `cell_size` with its corner at `corner`.
fn bed_and_level_error(bytes: &[u8], cells: u32, corner: Vec2, cell_size: f32) -> (f32, f32) {
    let row = bytes.len() / cells as usize;
    let lane = |o: usize| f32::from_le_bytes(bytes[o..o + 4].try_into().unwrap());
    let (mut bed_error, mut level_error) = (0.0f32, 0.0f32);
    for j in 0..cells as usize {
        for i in 0..cells as usize {
            let o = j * row + i * 16;
            let (b, h) = (lane(o), lane(o + 4));
            let at = corner + (Vec2::new(i as f32, j as f32) + 0.5) * cell_size;
            bed_error = bed_error.max((b - bed(at.x, at.y)).abs());
            level_error = level_error.max((b + h - LEVEL).abs());
        }
    }
    (bed_error, level_error)
}

/// Cells per edge of the second nested grid: 72 cells of 1/9 m, 8 m across.
const STREAM: u32 = 72;

/// The pool with 96 cells of 1/3 m nested in it and [`STREAM`] cells of 1/9 m inside them, the
/// second taking its steps from its stability limit, both centred on `follow`.
fn two_level_pool(follow: Vec2) -> App {
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
    let first = ShallowWaterNestedGrid::new(&mut images, CHILD);
    let mut stream = ShallowWaterNestedGrid::new(&mut images, STREAM);
    stream.steps = None;
    controller.nested = vec![first, stream];
    for nested in &mut controller.nested {
        nested.follow = follow;
    }
    app.insert_resource(controller);
    while app.plugins_state() == PluginsState::Adding {
        bevy::tasks::tick_global_task_pools_on_main_thread();
        thread::sleep(Duration::from_millis(1));
    }
    app.finish();
    app.cleanup();
    app
}

#[test]
#[ignore = "requires a graphics adapter"]
fn a_second_nested_grid_follows_its_point_inside_the_first() {
    let mut app = two_level_pool(Vec2::new(-20.0, 0.0));
    let deadline = Instant::now() + Duration::from_secs(180);
    step(&mut app);
    let start = app
        .world()
        .resource::<ShallowWaterController>()
        .nest_levels();
    assert_eq!(start.len(), 2);
    // The first grid steps three times per parent step of 1/30 s. The second's own limit,
    // 0.5 (1/9 m) / 7.7 m/s, needs two steps of the first grid's 1/90 s.
    assert_eq!((start[0].steps, start[1].steps), (3, 2));
    assert!((start[1].dt - 1.0 / 180.0).abs() < 1e-7);
    // Both have cells under 0.5 m, so both advect momentum conservatively.
    assert!(start[0].momentum_conserving && start[1].momentum_conserving);

    for frame in 1..120 {
        // The point walks 0.25 m per frame along +x.
        let follow = Vec2::new(-20.0 + 0.25 * frame as f32, 0.0);
        for nested in &mut app
            .world_mut()
            .resource_mut::<ShallowWaterController>()
            .nested
        {
            nested.follow = follow;
        }
        step(&mut app);
        assert!(Instant::now() < deadline, "run timed out");
    }
    let controller = app.world().resource::<ShallowWaterController>();
    let levels = controller.nest_levels();
    // Centred on x = 9.75: the first grid's corner is at (-6, -16), so the second grid's 24
    // cells of the first start 35 cells in along x and 36 along z.
    assert_eq!(levels[0].offset, IVec2::new(42, 32));
    assert_eq!(levels[1].offset, IVec2::new(35, 36));
    assert!((levels[1].corner - Vec2::new(-6.0 + 35.0 / 3.0, -16.0 + 12.0)).length() < 1e-4);
    // Both moved with the point: 30 cells of the pool, and 267 of the second grid.
    assert_eq!(levels[0].offset - start[0].offset, IVec2::new(30, 0));
    assert_eq!(
        levels[1].corner_cells - start[1].corner_cells,
        IVec2::new(267, 0)
    );
    let first_bed = controller.nested[0].images.bed_water.clone();
    let stream_bed = controller.nested[1].images.bed_water.clone();

    let mut heightfields: Vec<(i32, Vec2, f32)> = app
        .world_mut()
        .query::<&WaterHeightfield>()
        .iter(app.world())
        .map(|h| (h.order, h.center, h.cell_size))
        .collect();
    heightfields.sort_by_key(|h| h.0);
    assert_eq!(heightfields.len(), 3, "{heightfields:?}");
    assert_eq!(heightfields[0], (0, Vec2::ZERO, 1.0));
    assert_eq!(heightfields[1].0, 1);
    assert_eq!(heightfields[2].0, 2);
    assert!((heightfields[2].1 - levels[1].center).length() < 1e-4);
    assert!((heightfields[2].2 - 1.0 / 9.0).abs() < 1e-6);

    app.world_mut()
        .resource_mut::<ShallowWaterController>()
        .paused = true;
    let bytes = read_back(&mut app, first_bed, deadline);
    let (bed_error, first_level) =
        bed_and_level_error(&bytes, CHILD, levels[0].corner, levels[0].cell_size);
    assert!(
        bed_error < 0.01,
        "the first grid's bed is off by {bed_error} m"
    );
    let bytes = read_back(&mut app, stream_bed, deadline);
    let (bed_error, stream_level) =
        bed_and_level_error(&bytes, STREAM, levels[1].corner, levels[1].cell_size);
    // The second grid interpolates the first grid's bed, which interpolates the pool's samples.
    assert!(
        bed_error < 0.01,
        "the second grid's bed is off by {bed_error} m"
    );
    println!("still water error: first {first_level} m, second {stream_level} m");
    assert!(
        first_level < 1e-5 && stream_level < 1e-5,
        "still water moved {first_level} m in the first grid and {stream_level} m in the second"
    );
}

/// The water depth summed over a `bed_water` readback of `cells` per edge, and the furthest the
/// surface sits from [`LEVEL`]. Panics on a cell that isn't finite.
fn depth_sum_and_wave(bytes: &[u8], cells: u32) -> (f64, f32) {
    let row = bytes.len() / cells as usize;
    let lane = |o: usize| f32::from_le_bytes(bytes[o..o + 4].try_into().unwrap());
    let (mut sum, mut wave) = (0.0f64, 0.0f32);
    for j in 0..cells as usize {
        for i in 0..cells as usize {
            let o = j * row + i * 16;
            let (b, h) = (lane(o), lane(o + 4));
            assert!(
                b.is_finite() && h.is_finite(),
                "cell ({i}, {j}) is {b}, {h}"
            );
            sum += h as f64;
            wave = wave.max((b + h - LEVEL).abs());
        }
    }
    (sum, wave)
}

#[test]
#[ignore = "requires a graphics adapter"]
fn a_mover_stirs_the_second_nested_grid() {
    let mut app = two_level_pool(Vec2::ZERO);
    let deadline = Instant::now() + Duration::from_secs(180);
    for _ in 0..10 {
        step(&mut app);
    }
    let images = {
        let controller = app.world().resource::<ShallowWaterController>();
        [
            controller.bed_water.clone(),
            controller.nested[1].images.bed_water.clone(),
        ]
    };
    app.world_mut()
        .resource_mut::<ShallowWaterController>()
        .paused = true;
    let (before, _) = depth_sum_and_wave(&read_back(&mut app, images[0].clone(), deadline), CELLS);
    app.world_mut()
        .resource_mut::<ShallowWaterController>()
        .paused = false;

    // A wading point crosses the second grid along x at 3 m/s.
    let mover = app
        .world_mut()
        .spawn((
            Transform::from_xyz(-3.0, 0.0, 0.0),
            ShallowWaterMover::default(),
        ))
        .id();
    for frame in 0..120 {
        let x = -3.0 + 0.05 * frame as f32;
        app.world_mut()
            .entity_mut(mover)
            .get_mut::<Transform>()
            .unwrap()
            .translation
            .x = x;
        step(&mut app);
        assert!(Instant::now() < deadline, "run timed out");
    }
    app.world_mut()
        .resource_mut::<ShallowWaterController>()
        .paused = true;
    let (after, _) = depth_sum_and_wave(&read_back(&mut app, images[0].clone(), deadline), CELLS);
    let (_, wave) = depth_sum_and_wave(&read_back(&mut app, images[1].clone(), deadline), STREAM);
    let drift = ((after - before) / before).abs();
    println!("pool volume drift {drift:e}, second grid wave {wave} m");
    assert!(wave > 1e-3, "the second grid's surface moved only {wave} m");
    assert!(
        drift < 1e-5,
        "the pool's water changed by {drift:e} of its volume"
    );
}
