//! The deferred water path against the forward one. Run explicitly with a graphics adapter.
//!
//! A flat, black water plane is seen straight down through an orthographic camera under a sun 20
//! degrees off the vertical, so every pixel sees the sun's specular lobe 10 degrees off its peak.
//! A smooth surface shows almost none of it there, and a rougher one more. The slope variance in
//! the foam lane's alpha must roughen the surface the same way in both paths, and the flow detail
//! that drift lays over the water must break up the flat shade in both.
use bevy::{
    app::PluginsState,
    camera::{Exposure, ImageRenderTarget, RenderTarget, ScalingMode},
    core_pipeline::{
        prepass::{DeferredPrepass, DepthPrepass},
        tonemapping::Tonemapping,
    },
    light::GlobalAmbientLight,
    pbr::DefaultOpaqueRendererMethod,
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

const SIZE: u32 = 256;
/// Material roughness, low enough that the slope variance dominates.
const ROUGHNESS: f32 = 0.1;
/// Mean square slope of the unresolved waves, near what the ocean drops from a distant slice.
const VARIANCE: f32 = 0.05;
/// Sun angle off the vertical, in degrees.
const SUN_TILT: f32 = 20.0;

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

fn capture(app: &mut App, frames: u64) -> Vec<f32> {
    let first = app.world().resource::<Capture>().serial;
    let deadline = Instant::now() + Duration::from_secs(30);
    while app.world().resource::<Capture>().serial < first + frames {
        step(app);
        assert!(Instant::now() < deadline, "image readback timed out");
    }
    app.world()
        .resource::<Capture>()
        .pixels
        .chunks_exact(4)
        .map(|p| linear(p[1]))
        .collect()
}

fn linear(value: u8) -> f32 {
    let s = value as f32 / 255.0;
    if s <= 0.04045 {
        s / 12.92
    } else {
        ((s + 0.055) / 1.055).powf(2.4)
    }
}

fn mean(values: &[f32]) -> f32 {
    values.iter().sum::<f32>() / values.len() as f32
}

fn deviation(values: &[f32]) -> f32 {
    let m = mean(values);
    (values.iter().map(|v| (v - m) * (v - m)).sum::<f32>() / values.len() as f32).sqrt()
}

/// Half float bits of a normal, positive `v`, or of zero.
fn f16_bits(v: f32) -> u16 {
    if v == 0.0 {
        return 0;
    }
    let bits = v.to_bits();
    let exponent = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    ((exponent as u32) << 10 | (bits >> 13) & 0x3ff) as u16
}

fn glitter_app(deferred: bool) -> App {
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
    .insert_resource(GlobalAmbientLight::NONE)
    .insert_resource(TimeUpdateStrategy::ManualDuration(Duration::from_secs_f32(
        1.0 / 30.0,
    )))
    .insert_resource(bevy::render::error_handler::RenderErrorHandler(
        |error, _, _| {
            panic!("GPU validation failed: {error:?}");
        },
    ))
    .init_resource::<Capture>();
    if deferred {
        app.insert_resource(DefaultOpaqueRendererMethod::deferred());
    }
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
    // 2 m across 256 pixels, fine enough that the flow detail's ripples are resolved.
    let camera = app
        .world_mut()
        .spawn((
            Camera3d::default(),
            Msaa::Off,
            Tonemapping::None,
            Exposure { ev100: 0.0 },
            Projection::Orthographic(OrthographicProjection {
                scaling_mode: ScalingMode::Fixed {
                    width: 2.0,
                    height: 2.0,
                },
                ..OrthographicProjection::default_3d()
            }),
            Transform::from_xyz(0.3, 4.0, -0.2).looking_at(Vec3::new(0.3, 0.0, -0.2), Vec3::NEG_Z),
            RenderTarget::Image(ImageRenderTarget {
                handle: target.clone(),
                scale_factor: 1.0,
            }),
        ))
        .id();
    if deferred {
        app.world_mut()
            .entity_mut(camera)
            .insert((DepthPrepass, DeferredPrepass));
    }
    let tilt = SUN_TILT.to_radians();
    app.world_mut().spawn((
        DirectionalLight {
            illuminance: 10.0,
            ..default()
        },
        Transform::default().looking_to(Vec3::new(tilt.sin(), -tilt.cos(), 0.0), Vec3::Z),
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
                perceptual_roughness: ROUGHNESS,
                ..default()
            },
            extension: WaterSurfaceExtension::default(),
        });
    app.world_mut()
        .spawn((Mesh3d(mesh), MeshMaterial3d(material), WaterSurfaceTag));
    step(&mut app);
    app
}

/// Fills the foam lane with ten meters of water whose unresolved waves have mean square slope
/// `variance`.
fn set_variance(app: &mut App, variance: f32) {
    let handle = app.world().resource::<WaterLodData>().foam.clone();
    let mut images = app.world_mut().resource_mut::<Assets<Image>>();
    let mut image = images.get_mut(&handle).unwrap();
    for texel in image.data.as_mut().unwrap().chunks_exact_mut(8) {
        texel[2..4].copy_from_slice(&f16_bits(10.0).to_le_bytes());
        texel[6..8].copy_from_slice(&f16_bits(variance).to_le_bytes());
    }
}

/// Sets a uniform drift of `speed` meters per second along X.
fn set_drift(app: &mut App, speed: f32) {
    let handle = app.world().resource::<WaterLodData>().drift.clone();
    let mut images = app.world_mut().resource_mut::<Assets<Image>>();
    let mut image = images.get_mut(&handle).unwrap();
    for texel in image.data.as_mut().unwrap().chunks_exact_mut(8) {
        texel[0..4].copy_from_slice(&speed.to_le_bytes());
        texel[4..8].copy_from_slice(&0.0f32.to_le_bytes());
    }
}

struct Shade {
    smooth: f32,
    rough: f32,
    still_deviation: f32,
    flowing_deviation: f32,
}

fn shade(deferred: bool) -> Shade {
    let mut app = glitter_app(deferred);
    set_drift(&mut app, 0.0);
    set_variance(&mut app, 0.0);
    let still = capture(&mut app, 4);
    set_variance(&mut app, VARIANCE);
    let rough = capture(&mut app, 4);
    set_drift(&mut app, 1.0);
    let flowing = capture(&mut app, 30);
    Shade {
        smooth: mean(&still),
        rough: mean(&rough),
        still_deviation: deviation(&rough),
        flowing_deviation: deviation(&flowing),
    }
}

#[test]
#[ignore = "requires a graphics adapter"]
fn deferred_water_matches_forward_roughness_and_flow_detail() {
    let forward = shade(false);
    let deferred = shade(true);
    for (name, s) in [("forward", &forward), ("deferred", &deferred)] {
        eprintln!(
            "{name}: smooth {:.4} rough {:.4} deviation still {:.4} flowing {:.4}",
            s.smooth, s.rough, s.still_deviation, s.flowing_deviation
        );
        assert!(
            s.rough > 3.0 * s.smooth + 0.01,
            "{name}: the slope variance must widen the sun's lobe"
        );
        assert!(
            s.flowing_deviation > 4.0 * s.still_deviation + 0.005,
            "{name}: drift must lay flow detail over the water"
        );
    }
    assert!(
        (deferred.rough - forward.rough).abs() < 0.1 * forward.rough,
        "deferred roughened shade {} differs from forward {}",
        deferred.rough,
        forward.rough
    );
    assert!(
        (deferred.flowing_deviation - forward.flowing_deviation).abs()
            < 0.25 * forward.flowing_deviation,
        "deferred flow detail {} differs from forward {}",
        deferred.flowing_deviation,
        forward.flowing_deviation
    );
}
