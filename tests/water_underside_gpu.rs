//! The underside of the water surface. Run explicitly with a graphics adapter.
//!
//! A camera under a flat surface looks straight up at a uniform sky. Inside Snell's window the
//! sky refracts through, dimmed only by Fresnel reflection. Past the critical angle the surface
//! reflects everything, so with a medium that neither scatters nor absorbs the image turns black
//! exactly at the window's edge.
//!
//! A third test lights the surface with a sun and no sky. The sun shows through as a lobe around
//! its refracted direction, and the radiance in the lobe integrates over the directions below to
//! the Fresnel transmittance times the sun's irradiance on the surface.
use bevy::{
    app::PluginsState,
    camera::{Exposure, Hdr, ImageRenderTarget, RenderTarget},
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
    f32::consts::{FRAC_PI_2, FRAC_PI_3},
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
    .insert_resource(WaterMedium(medium))
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
            extension: WaterSurfaceExtension::default(),
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

/// Value of a little-endian half float.
fn half(bits: u16) -> f32 {
    let exponent = ((bits >> 10) & 31) as i32;
    let mantissa = (bits & 1023) as f32;
    let sign = if bits >> 15 == 1 { -1.0 } else { 1.0 };
    if exponent == 0 {
        sign * mantissa * 2f32.powi(-24)
    } else {
        sign * (1.0 + mantissa / 1024.0) * 2f32.powi(exponent - 15)
    }
}

#[test]
#[ignore = "requires a graphics adapter"]
fn the_sun_shows_through_snells_window_with_its_transmitted_irradiance() {
    const SUN_SIZE: u32 = 256;
    const SUN_FOV: f32 = FRAC_PI_2;
    /// Sun illuminance in lux, small so the exposed lobe stays in half float range.
    const SUN: f32 = 2.0;
    /// GGX width 0.2, wide enough to span many pixels.
    const ROUGHNESS: f32 = 0.447_213_6;
    let elevation = 60f32.to_radians();
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
    // A clear medium, so only the refracted sun reaches the eye.
    .insert_resource(WaterMedium(WaterMediumUniform {
        absorption: Vec3::ZERO,
        scattering: Vec3::ZERO,
        diffuse_attenuation: Vec3::ZERO,
        ..default()
    }))
    .insert_resource(bevy::light::GlobalAmbientLight::NONE)
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
    let mut image = Image::new_target_texture(SUN_SIZE, SUN_SIZE, TextureFormat::Rgba16Float, None);
    image.texture_descriptor.usage |= TextureUsages::COPY_SRC;
    let target = app.world_mut().resource_mut::<Assets<Image>>().add(image);
    let exposure = Exposure { ev100: 0.0 };
    let camera = Transform::from_xyz(0.0, -1.0, 0.0).looking_to(Vec3::Y, Vec3::Z);
    app.world_mut().spawn((
        Camera3d::default(),
        Hdr,
        Msaa::Off,
        Tonemapping::None,
        DebandDither::Disabled,
        exposure,
        Projection::Perspective(PerspectiveProjection {
            fov: SUN_FOV,
            ..default()
        }),
        camera,
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
    let to_light = Vec3::new(elevation.cos(), elevation.sin(), 0.0);
    app.world_mut().spawn((
        DirectionalLight {
            illuminance: SUN,
            ..default()
        },
        Transform::default().looking_to(-to_light, Vec3::Z),
    ));
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
                perceptual_roughness: ROUGHNESS,
                ..default()
            },
            extension: WaterSurfaceExtension::default(),
        });
    app.world_mut()
        .spawn((Mesh3d(mesh), MeshMaterial3d(material), WaterSurfaceTag));
    step(&mut app);
    let pixels: Vec<f32> = capture(&mut app, 4)
        .chunks_exact(8)
        .map(|p| half(u16::from_le_bytes([p[0], p[1]])) / exposure.exposure())
        .collect();

    // Each pixel's view direction, and the solid angle it covers times the cosine to the normal,
    // which is the camera axis.
    let tan = (SUN_FOV * 0.5).tan();
    let pixel = 2.0 * tan / SUN_SIZE as f32;
    let direction = |x: u32, y: u32| {
        let u = (x as f32 + 0.5) * pixel - tan;
        let v = tan - (y as f32 + 0.5) * pixel;
        camera.rotation * Vec3::new(u, v, -1.0).normalize()
    };
    let mut integral = 0.0;
    let mut peak = (0.0, Vec3::ZERO);
    for y in 0..SUN_SIZE {
        for x in 0..SUN_SIZE {
            let radiance = pixels[(y * SUN_SIZE + x) as usize];
            let d = direction(x, y);
            integral += radiance * pixel * pixel * d.y.powi(4);
            if radiance > peak.0 {
                peak = (radiance, d);
            }
        }
    }
    let cos_i = elevation.sin();
    let sin_t = elevation.cos() / IOR;
    let refracted_up = Vec3::new(sin_t, (1.0 - sin_t * sin_t).sqrt(), 0.0);
    let r = {
        let cos_t = refracted_up.y;
        let rs = (cos_i - IOR * cos_t) / (cos_i + IOR * cos_t);
        let rp = (cos_t - IOR * cos_i) / (cos_t + IOR * cos_i);
        0.5 * (rs * rs + rp * rp)
    };
    let expected = (1.0 - r) * SUN * cos_i;
    let offset = peak.1.angle_between(refracted_up).to_degrees();
    eprintln!(
        "sun lobe: peak {} at {offset:.2} degrees from the refracted sun, integral {integral} against {expected}",
        peak.0
    );
    assert!(
        offset < 1.0,
        "lobe peaks {offset:.2} degrees from the refracted sun"
    );
    assert!(
        (integral / expected - 1.0).abs() < 0.05,
        "lobe integrates to {integral} against {expected}"
    );
}
