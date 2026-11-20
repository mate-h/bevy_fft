//! How a heightfield draws its dry cells beside a stream that runs above the still-water level.
//! Run explicitly with a graphics adapter.
//!
//! The grid holds a stream 2 m wide whose surface stands 2 m above the still level, between banks
//! 3 m high. With `WaterHeightfieldDry::Level` the bank cells drop to the still level, so a mesh
//! vertex that misses the stream's two cells sinks under the banks and the stream vanishes from a
//! distance. With `WaterHeightfieldDry::Extend` the bank cells beside the stream carry its
//! surface, so the mesh stays up and the banks hide it.
use bevy::{
    app::PluginsState,
    asset::RenderAssetUsages,
    prelude::*,
    render::{
        gpu_readback::{Readback, ReadbackComplete},
        render_resource::{Extent3d, TextureDimension, TextureFormat, TextureUsages},
    },
    time::TimeUpdateStrategy,
    window::ExitCondition,
    winit::WinitPlugin,
};
use bevy_fft::water::*;
use std::{
    thread,
    time::{Duration, Instant},
};

const CELLS: u32 = 64;
/// The stream's columns, its bed and depth, and the banks' height, in meters.
const STREAM: std::ops::Range<u32> = 31..33;
const STREAM_BED: f32 = 1.5;
const STREAM_DEPTH: f32 = 0.5;
const BANK: f32 = 3.0;

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

#[derive(Resource, Default)]
struct Capture {
    bytes: Vec<u8>,
    serial: u64,
}

fn step(app: &mut App) {
    app.update();
    assert!(app.should_exit().is_none(), "renderer must remain healthy");
    thread::sleep(Duration::from_millis(2));
}

/// IEEE half to f32.
fn half(bits: u16) -> f32 {
    let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exponent = ((bits >> 10) & 0x1f) as i32;
    let mantissa = (bits & 0x3ff) as f32;
    sign * match exponent {
        0 => mantissa * 2f32.powi(-24),
        31 => f32::INFINITY,
        e => (1.0 + mantissa / 1024.0) * 2f32.powi(e - 15),
    }
}

/// Surface height of every texel of the one LOD slice, row by row, with `dry` for the grid.
fn heights(dry: WaterHeightfieldDry) -> Vec<f32> {
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
        resolution: CELLS,
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
    while app.plugins_state() == PluginsState::Adding {
        bevy::tasks::tick_global_task_pools_on_main_thread();
        thread::sleep(Duration::from_millis(1));
    }
    app.finish();
    app.cleanup();
    let n = CELLS * CELLS;
    let bed_water = float_image((0..n).map(|i| {
        if STREAM.contains(&(i % CELLS)) {
            Vec4::new(STREAM_BED, STREAM_DEPTH, 0.0, 0.0)
        } else {
            Vec4::new(BANK, 0.0, 0.0, 0.0)
        }
    }));
    let velocity = float_image((0..n).map(|_| Vec4::ZERO));
    let mut images = app.world_mut().resource_mut::<Assets<Image>>();
    let bed_water = images.add(bed_water);
    let velocity = images.add(velocity);
    app.world_mut().spawn(WaterHeightfield {
        dry,
        blend_cells: 0.0,
        ..WaterHeightfield::new(bed_water, velocity, UVec2::splat(CELLS))
    });
    // Let the passes run, then read the displacement slice back.
    for _ in 0..10 {
        step(&mut app);
    }
    let displacement = app.world().resource::<WaterLodData>().displacement.clone();
    app.world_mut()
        .spawn(Readback::texture(displacement))
        .observe(
            |event: On<ReadbackComplete>, mut capture: ResMut<Capture>| {
                capture.bytes = event.data.clone();
                capture.serial += 1;
            },
        );
    let deadline = Instant::now() + Duration::from_secs(60);
    while app.world().resource::<Capture>().serial < 3 {
        step(&mut app);
        assert!(Instant::now() < deadline, "readback timed out");
    }
    let bytes = &app.world().resource::<Capture>().bytes;
    let row = bytes.len() / CELLS as usize;
    (0..n as usize)
        .map(|i| {
            // Rgba16Float with the height in `.z`.
            let o = (i / CELLS as usize) * row + (i % CELLS as usize) * 8 + 4;
            half(u16::from_le_bytes([bytes[o], bytes[o + 1]]))
        })
        .collect()
}

/// Texels in the middle row whose surface is within 0.1 m of the stream's.
fn up_with_the_stream(heights: &[f32]) -> Vec<u32> {
    let surface = STREAM_BED + STREAM_DEPTH;
    let row = (CELLS / 2 * CELLS) as usize;
    (0..CELLS)
        .filter(|&x| (heights[row + x as usize] - surface).abs() < 0.1)
        .collect()
}

#[test]
#[ignore = "requires a graphics adapter"]
fn extend_keeps_a_raised_stream_surface_out_into_its_banks() {
    let level = up_with_the_stream(&heights(WaterHeightfieldDry::Level(0.0)));
    let extend = up_with_the_stream(&heights(WaterHeightfieldDry::Extend(0.0)));
    println!("at the stream's surface: Level {level:?}, Extend {extend:?}");
    assert!(
        level.iter().all(|x| STREAM.contains(x)),
        "Level put bank texels at the stream's surface: {level:?}"
    );
    let reach = (STREAM.start - 2)..(STREAM.end + 2);
    assert!(
        reach.clone().all(|x| extend.contains(&x)),
        "Extend should carry the surface two cells into each bank: {extend:?}"
    );
    assert!(
        extend.iter().all(|x| reach.contains(x)),
        "Extend carried the surface past two cells: {extend:?}"
    );
}
