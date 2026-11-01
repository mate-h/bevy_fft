//! Per-pixel foam on the water surface. Run explicitly with a graphics adapter.
//!
//! A flat, unlit, black water plane is drawn from above with fixed active and residual foam in
//! [`WaterLodData::foam_state`]. Each pixel then shows foam albedo times foam opacity, so the share
//! of pixels at albedo 0.5 or more is the Stage A coverage the whitecap calibration measures.
use bevy::{
    app::PluginsState,
    camera::{ImageRenderTarget, RenderTarget, ScalingMode},
    core_pipeline::tonemapping::Tonemapping,
    prelude::*,
    render::{
        gpu_readback::{Readback, ReadbackComplete},
        render_resource::{TextureFormat, TextureUsages},
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

const SIZE: u32 = 1024;
/// Albedo of dense foam in `surface.wesl`.
const FOAM_ALBEDO: f32 = 0.55;

#[derive(Resource, Default)]
struct Capture {
    pixels: Vec<u8>,
    serial: u64,
}

fn step(app: &mut App) {
    app.update();
    assert!(app.should_exit().is_none(), "renderer must remain healthy");
    thread::sleep(Duration::from_millis(2));
}

fn capture(app: &mut App, frames: u64) -> Vec<u8> {
    let first = app.world().resource::<Capture>().serial;
    let deadline = Instant::now() + Duration::from_secs(30);
    while app.world().resource::<Capture>().serial < first + frames {
        step(app);
        assert!(Instant::now() < deadline, "image readback timed out");
    }
    app.world().resource::<Capture>().pixels.clone()
}

fn linear(value: u8) -> f32 {
    let s = value as f32 / 255.0;
    if s <= 0.04045 {
        s / 12.92
    } else {
        ((s + 0.055) / 1.055).powf(2.4)
    }
}

fn foam_app() -> App {
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
    .insert_resource(WaterLodSettings {
        resolution: 64,
        slice_count: 1,
        base_texel_size: 1.0,
    })
    // Foam neither fades nor turns into residual foam, so the state set below stays put.
    .insert_resource(WaterFoamSettings {
        active_half_life: f32::INFINITY,
        residual_half_life: f32::INFINITY,
        residual_yield: 0.0,
        ..default()
    })
    // Steps that land the captures on different flow phases of the foam texture.
    .insert_resource(TimeUpdateStrategy::ManualDuration(Duration::from_secs_f32(
        0.37,
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
    let mut image = Image::new_target_texture(
        SIZE,
        SIZE,
        TextureFormat::Rgba8UnormSrgb,
        Some(TextureFormat::Rgba8UnormSrgb),
    );
    image.texture_descriptor.usage |= TextureUsages::COPY_SRC;
    let target = app.world_mut().resource_mut::<Assets<Image>>().add(image);
    // 6 m across 1024 pixels, finer than the foam texture, so the lace is resolved, and wide
    // enough to hold a few hundred foam cells.
    app.world_mut().spawn((
        Camera3d::default(),
        Msaa::Off,
        Tonemapping::None,
        Projection::Orthographic(OrthographicProjection {
            scaling_mode: ScalingMode::Fixed {
                width: 6.0,
                height: 6.0,
            },
            ..OrthographicProjection::default_3d()
        }),
        Transform::from_xyz(0.3, 4.0, -0.2).looking_at(Vec3::new(0.3, 0.0, -0.2), Vec3::NEG_Z),
        RenderTarget::Image(ImageRenderTarget {
            handle: target.clone(),
            scale_factor: 1.0,
        }),
    ));
    app.world_mut().spawn(Readback::texture(target)).observe(
        |event: On<ReadbackComplete>, mut capture: ResMut<Capture>| {
            capture.pixels = event.data.clone();
            capture.serial += 1;
        },
    );
    let mesh = app
        .world_mut()
        .resource_mut::<Assets<Mesh>>()
        .add(Plane3d::default().mesh().size(8.0, 8.0));
    let material = app
        .world_mut()
        .resource_mut::<Assets<WaterSurfaceMaterial>>()
        .add(WaterSurfaceMaterial {
            base: StandardMaterial {
                base_color: Color::BLACK,
                unlit: true,
                ..default()
            },
            extension: WaterSurfaceExtension::default(),
        });
    app.world_mut()
        .spawn((Mesh3d(mesh), MeshMaterial3d(material), WaterSurfaceTag));
    step(&mut app);
    app
}

fn set_foam_state(app: &mut App, active: f32, residual: f32) {
    let handle = app.world().resource::<WaterLodData>().foam_state.clone();
    let mut images = app.world_mut().resource_mut::<Assets<Image>>();
    let mut image = images.get_mut(&handle).unwrap();
    for texel in image.data.as_mut().unwrap().chunks_exact_mut(8) {
        texel[0..4].copy_from_slice(&active.to_le_bytes());
        texel[4..8].copy_from_slice(&residual.to_le_bytes());
    }
}

struct FoamStatistics {
    /// Share of pixels whose centre lies inside dense foam. A pixel centred on the boundary is
    /// half dense foam over the thickest residual foam, at opacity 0.5 + 0.5 * 0.27, so that is
    /// the cut. This matches the active coverage.
    dense: f32,
    /// Share of pixels at albedo 0.5 or more, the Stage A count of Monahan and Woolf (1989).
    /// Edge pixels below that albedo drop out.
    stage_a: f32,
    /// Share of pixels with any foam. Partly covered edge pixels count, so this runs a little
    /// above the covered share.
    any: f32,
    mean: f32,
}

fn foam_statistics(pixels: &[u8]) -> FoamStatistics {
    let albedo: Vec<f32> = pixels.chunks_exact(4).map(|p| linear(p[0])).collect();
    let n = albedo.len() as f32;
    let share = |min: f32| albedo.iter().filter(|a| **a >= min).count() as f32 / n;
    FoamStatistics {
        dense: share(FOAM_ALBEDO * (0.5 + 0.5 * 1.5 * 0.1 / 0.55)),
        stage_a: share(0.5),
        any: share(0.005),
        mean: albedo.iter().sum::<f32>() / n,
    }
}

#[test]
#[ignore = "requires a graphics adapter"]
fn rendered_foam_matches_sim_coverage() {
    let mut app = foam_app();

    set_foam_state(&mut app, 0.2, 0.3);
    for _ in 0..3 {
        let s = foam_statistics(&capture(&mut app, 7));
        eprintln!(
            "active 0.2, residual 0.3: dense {}, stage A {}, any {}, mean albedo {}",
            s.dense, s.stage_a, s.any, s.mean
        );
        assert!(
            (s.dense - 0.2).abs() < 0.02,
            "dense foam must cover the active share, got {}",
            s.dense
        );
        assert!(
            s.stage_a > 0.7 * 0.2,
            "most dense foam must reach albedo 0.5, got {}",
            s.stage_a
        );
        assert!(
            (0.5..0.58).contains(&s.any),
            "foam must cover active plus residual, got {}",
            s.any
        );
        let expected = FOAM_ALBEDO * (0.2 + 0.3 * 0.1 / 0.55);
        assert!(
            (s.mean / expected - 1.0).abs() < 0.05,
            "mean albedo {} against {expected}",
            s.mean
        );
    }

    // Residual foam alone is lace well below the Stage A albedo.
    set_foam_state(&mut app, 0.0, 0.3);
    let s = foam_statistics(&capture(&mut app, 30));
    eprintln!(
        "active 0, residual 0.3: dense {}, stage A {}, any {}, mean albedo {}",
        s.dense, s.stage_a, s.any, s.mean
    );
    assert!(s.dense < 0.001, "residual foam must stay dim, {}", s.dense);
    assert!(
        (0.3..0.36).contains(&s.any),
        "lace must cover 0.3, got {}",
        s.any
    );
    assert!(
        (s.mean / (0.3 * 0.1) - 1.0).abs() < 0.05,
        "residual foam reflects about 10 percent, mean {}",
        s.mean
    );

    // No coverage leaves no foam, also on the brightest pattern values.
    set_foam_state(&mut app, 0.0, 0.0);
    let s = foam_statistics(&capture(&mut app, 30));
    assert_eq!(s.any, 0.0, "foam without coverage, mean albedo {}", s.mean);
}

/// Fills the drift with the shear `u = (rate z, 0)`, which leaves the foam state in place but
/// stretches the advected texture coordinates.
fn set_shear(app: &mut App, rate: f32) {
    let handle = app.world().resource::<WaterLodData>().drift.clone();
    let mut images = app.world_mut().resource_mut::<Assets<Image>>();
    let mut image = images.get_mut(&handle).unwrap();
    let res = image.texture_descriptor.size.width as usize;
    let data = image.data.as_mut().unwrap();
    for y in 0..res {
        let z = y as f32 + 0.5 - res as f32 * 0.5;
        for x in 0..res {
            let o = (y * res + x) * 8;
            data[o..o + 4].copy_from_slice(&(rate * z).to_le_bytes());
            data[o + 4..o + 8].copy_from_slice(&0f32.to_le_bytes());
        }
    }
}

#[test]
#[ignore = "requires a graphics adapter"]
fn sheared_foam_keeps_its_coverage() {
    // A shear of 0.3 per second deforms the long layer past the target and keeps the short layer
    // under it, so the pattern blends both layers, by a weight that changes with the phases.
    let mut app = foam_app();
    set_shear(&mut app, 0.3);
    set_foam_state(&mut app, 0.2, 0.3);
    for _ in 0..4 {
        let s = foam_statistics(&capture(&mut app, 5));
        eprintln!(
            "sheared, active 0.2, residual 0.3: dense {}, stage A {}, any {}, mean albedo {}",
            s.dense, s.stage_a, s.any, s.mean
        );
        assert!(
            (s.dense - 0.2).abs() < 0.02,
            "dense foam must cover the active share, got {}",
            s.dense
        );
        let expected = FOAM_ALBEDO * (0.2 + 0.3 * 0.1 / 0.55);
        assert!(
            (s.mean / expected - 1.0).abs() < 0.05,
            "mean albedo {} against {expected}",
            s.mean
        );
    }
}
