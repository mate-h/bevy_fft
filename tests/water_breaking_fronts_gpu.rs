//! Breaking front detection on the GPU and linking on the CPU. Run explicitly with a graphics
//! adapter.
use bevy::{
    app::PluginsState,
    asset::RenderAssetUsages,
    prelude::*,
    render::render_resource::{Extent3d, TextureDimension, TextureFormat, TextureUsages},
    time::TimeUpdateStrategy,
    window::ExitCondition,
    winit::WinitPlugin,
};
use bevy_fft::water::*;
use std::{
    thread,
    time::{Duration, Instant},
};

const CELLS: u32 = 128;
/// Height of the bore above the water in front of it, in meters.
const BORE: f32 = 0.6;
/// Width of the bore face in meters.
const FACE: f32 = 1.5;
const DEPTH: f32 = 1.0;
const SPEED: f32 = 1.5;

fn float_image(data: impl Iterator<Item = Vec4>) -> Image {
    let bytes: Vec<u8> = data
        .flat_map(|v| v.to_array())
        .flat_map(f32::to_le_bytes)
        .collect();
    let mut image = Image::new(
        Extent3d {
            width: CELLS,
            height: CELLS,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        bytes,
        TextureFormat::Rgba32Float,
        RenderAssetUsages::RENDER_WORLD,
    );
    image.texture_descriptor.usage |= TextureUsages::TEXTURE_BINDING;
    image
}

/// A straight bore through the grid centre travelling along `travel`. Behind its face the water
/// stands `BORE` higher and flows at `SPEED` along `travel`, then falls back gently 30 to 60 m
/// behind, so the back of the wave slopes the other way without being steep.
fn bore(travel: Vec2) -> (Image, Image) {
    let eta = |i: u32| {
        let p = Vec2::new((i % CELLS) as f32 + 0.5, (i / CELLS) as f32 + 0.5) - CELLS as f32 / 2.0;
        let s = p.dot(travel);
        let face = 0.5 * (1.0 - (s / (FACE * 0.5)).tanh());
        let t = ((s + 60.0) / 30.0).clamp(0.0, 1.0);
        BORE * face * t * t * (3.0 - 2.0 * t)
    };
    let n = CELLS * CELLS;
    let bed_water = float_image((0..n).map(|i| Vec4::new(-DEPTH, DEPTH + eta(i), 0.0, 0.0)));
    let velocity = float_image((0..n).map(|i| {
        let u = travel * SPEED * eta(i) / BORE;
        Vec4::new(u.x, u.y, 0.0, 0.0)
    }));
    (bed_water, velocity)
}

fn step(app: &mut App) {
    app.update();
    assert!(app.should_exit().is_none(), "renderer must remain healthy");
    thread::sleep(Duration::from_millis(2));
}

fn fronts_app(travel: Vec2) -> (App, Entity) {
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
    .add_plugins(WaterPlugin)
    .insert_resource(WaterLodClear)
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
    let (bed_water, velocity) = bore(travel);
    let mut images = app.world_mut().resource_mut::<Assets<Image>>();
    let bed_water = images.add(bed_water);
    let velocity = images.add(velocity);
    let entity = app
        .world_mut()
        .spawn((
            WaterHeightfield {
                dry: WaterHeightfieldDry::Level(0.0),
                ..WaterHeightfield::new(bed_water, velocity, UVec2::splat(CELLS))
            },
            WaterBreakingFronts::default(),
        ))
        .id();
    (app, entity)
}

/// Steps until the fronts have been read back a few times, so tracking has run.
fn settled_fronts(app: &mut App, entity: Entity) -> WaterBreakingFronts {
    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        step(app);
        let fronts = app.world().get::<WaterBreakingFronts>(entity).unwrap();
        if fronts.frame >= 5 {
            return fronts.clone();
        }
        assert!(Instant::now() < deadline, "front readback timed out");
    }
}

fn check_bore(travel: Vec2) {
    let (mut app, entity) = fronts_app(travel);
    let fronts = settled_fronts(&mut app, entity);
    assert!(fronts.cell_count > 0, "no front cells");
    assert_eq!(fronts.lines.len(), 1, "{:#?}", fronts.lines);
    let line = &fronts.lines[0];
    let ends = (
        line.points[0].position.xz(),
        line.points.last().unwrap().position.xz(),
    );
    let span = (ends.1 - ends.0).dot(travel.perp()).abs();
    assert!(span > 0.8 * CELLS as f32, "line spans {span} m");
    for p in &line.points {
        let s = p.position.xz().dot(travel);
        assert!(s.abs() < 3.0, "point {s} m off the face");
        assert!(
            p.direction.dot(travel) > 0.95,
            "direction {:?}",
            p.direction
        );
        assert!(p.slope > 0.1, "slope {}", p.slope);
    }
    assert!(line.age > 0.0, "the line was not tracked across readbacks");
}

#[test]
#[ignore = "requires a graphics adapter"]
fn a_bore_along_x_gives_one_line_on_its_face() {
    check_bore(Vec2::X);
}

#[test]
#[ignore = "requires a graphics adapter"]
fn a_rotated_bore_gives_one_line_on_its_face() {
    check_bore(Vec2::from_angle(10f32.to_radians()).rotate(Vec2::NEG_Y));
}
