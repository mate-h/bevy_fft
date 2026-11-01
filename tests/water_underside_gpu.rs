//! The underside of the water surface. Run explicitly with a graphics adapter.
//!
//! A camera under a flat surface looks straight up at a uniform sky. Inside Snell's window the
//! sky refracts through, dimmed only by Fresnel reflection. Past the critical angle the surface
//! reflects everything, so with a medium that neither scatters nor absorbs the image turns black
//! exactly at the window's edge.
use bevy::{
    app::PluginsState,
    camera::{Exposure, ImageRenderTarget, RenderTarget},
    core_pipeline::tonemapping::{DebandDither, Tonemapping},
    light::EnvironmentMapLight,
    prelude::*,
    render::{
        gpu_readback::{Readback, ReadbackComplete},
        render_resource::{Face, TextureFormat, TextureUsages},
    },
    window::ExitCondition,
    winit::WinitPlugin,
};
use bevy_fft::water::*;
use std::{
    f32::consts::FRAC_PI_3,
    thread,
    time::{Duration, Instant},
};

const SIZE: u32 = 512;
/// Vertical field of view, wide enough to hold the 97 degree window.
const FOV: f32 = 2.0 * FRAC_PI_3;
const IOR: f32 = 1.333;
/// Linear brightness of the sky after exposure. Exposure at EV100 0 is 1 / 1.2.
const SKY: f32 = 0.5;

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

fn underside_app(medium: WaterMediumUniform) -> App {
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
        slice_count: 4,
        base_texel_size: 1.0,
    })
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
    let exposure = Exposure { ev100: 0.0 };
    let sky = {
        let mut images = app.world_mut().resource_mut::<Assets<Image>>();
        EnvironmentMapLight::solid_color(&mut images, Color::WHITE)
            .with_intensity(SKY / exposure.exposure())
    };
    app.world_mut().spawn((
        Camera3d::default(),
        Msaa::Off,
        Tonemapping::None,
        DebandDither::Disabled,
        exposure,
        sky,
        Projection::Perspective(PerspectiveProjection {
            fov: FOV,
            ..default()
        }),
        Transform::from_xyz(0.0, -2.0, 0.0).looking_at(Vec3::ZERO, Vec3::Z),
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
        .add(Plane3d::default().mesh().size(400.0, 400.0));
    let material = app
        .world_mut()
        .resource_mut::<Assets<WaterSurfaceMaterial>>()
        .add(WaterSurfaceMaterial {
            base: StandardMaterial {
                cull_mode: None::<Face>,
                ..default()
            },
            extension: WaterSurfaceExtension {
                medium,
                ..default()
            },
        });
    app.world_mut()
        .spawn((Mesh3d(mesh), MeshMaterial3d(material), WaterSurfaceTag));
    step(&mut app);
    app
}

/// Brightness along the centre row, from the image centre outward to the right.
fn centre_row(pixels: &[u8]) -> Vec<f32> {
    let row = SIZE / 2;
    (SIZE / 2..SIZE)
        .map(|x| linear(pixels[((row * SIZE + x) * 4) as usize]))
        .collect()
}

/// Angle from the zenith seen by the pixel `offset` pixels right of the centre, at its centre.
fn pixel_angle(offset: f32) -> f32 {
    let focal = SIZE as f32 * 0.5 / (FOV * 0.5).tan();
    (offset / focal).atan()
}

#[test]
#[ignore = "requires a graphics adapter"]
fn snells_window_ends_at_the_critical_angle() {
    let clear = WaterMediumUniform {
        absorption: Vec3::ZERO,
        scattering: Vec3::ZERO,
        diffuse_attenuation: Vec3::ZERO,
        ..WaterMediumUniform::jerlov(JerlovWaterType::OceanIB)
    };
    let mut app = underside_app(clear);
    let row = centre_row(&capture(&mut app, 4));

    // Straight up, only the normal-incidence Fresnel reflection is lost.
    let r0 = ((IOR - 1.0) / (IOR + 1.0)).powi(2);
    assert!(
        (row[0] / (SKY * (1.0 - r0)) - 1.0).abs() < 0.03,
        "zenith brightness {} against {}",
        row[0],
        SKY * (1.0 - r0)
    );

    // The first black pixel sits at the critical angle, within one pixel.
    let edge = row.iter().position(|v| *v == 0.0).expect("window edge");
    let critical = (1.0 / IOR).asin();
    let inside = pixel_angle(edge as f32 - 0.5);
    let outside = pixel_angle(edge as f32 + 0.5);
    eprintln!(
        "edge at pixel {edge}: {:.2} to {:.2} degrees, critical angle {:.2}",
        inside.to_degrees(),
        outside.to_degrees(),
        critical.to_degrees()
    );
    assert!(
        pixel_angle(edge as f32 - 1.5) <= critical && critical <= pixel_angle(edge as f32 + 0.5),
        "window edge at {:.2} degrees, critical angle {:.2}",
        pixel_angle(edge as f32).to_degrees(),
        critical.to_degrees()
    );
    assert!(
        row[edge..].iter().all(|v| *v == 0.0),
        "total internal reflection must stay black in a clear medium"
    );
}

#[test]
#[ignore = "requires a graphics adapter"]
fn total_internal_reflection_shows_the_lit_water() {
    // Jerlov II scatters strongly, so the reflected water body glows outside the window and the
    // medium between the eye and the surface veils the window.
    let mut app = underside_app(WaterMediumUniform::jerlov(JerlovWaterType::OceanII));
    let row = centre_row(&capture(&mut app, 4));
    let critical = (1.0 / IOR).asin();
    let edge = (0..row.len())
        .find(|&i| pixel_angle(i as f32) > critical)
        .unwrap();
    let outside = row[edge + 4];
    eprintln!(
        "zenith {}, past the edge {outside}, far {}",
        row[0],
        row[row.len() - 1]
    );
    assert!(outside > 0.0, "reflected water body must not be black");
    assert!(
        outside < row[0],
        "the window must be brighter than the reflection"
    );
}
