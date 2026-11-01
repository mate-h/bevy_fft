//! Caustic map of the water surface. Run explicitly with a graphics adapter.
//!
//! A sun straight overhead shines through a surface `h = A cos(k x)`. Each crest is a cylindrical
//! lens. For small slopes, a ray at `x` refracts by `(1 - 1/n) h'(x)` and lands at depth `d` at
//! `x + d (1 - 1/n) h'(x)`, so the light spreads by `1 - d / f` with
//! `f = 1 / ((1 - 1/n) A k^2 cos(k x))`. Halfway to the focus, crests show nearly twice the light
//! and troughs two thirds, and the mean stays one. The tests compare with the exact refraction
//! seen through the one-texel tent the map spreads each ray over.
use bevy::{
    app::PluginsState,
    camera::{Exposure, Hdr, ImageRenderTarget, RenderTarget, ScalingMode},
    core_pipeline::tonemapping::{DebandDither, Tonemapping},
    light::{GlobalAmbientLight, NotShadowCaster},
    prelude::*,
    render::{
        gpu_readback::{Readback, ReadbackComplete},
        render_resource::{TextureFormat, TextureUsages},
    },
    window::ExitCondition,
    winit::WinitPlugin,
};
use bevy_fft::water::*;
use std::{
    f32::consts::TAU,
    thread,
    time::{Duration, Instant},
};

const RESOLUTION: u32 = 128;
const TEXEL: f32 = 0.25;
const IOR: f32 = 1.333;
const WAVELENGTH: f32 = 4.0;
const AMPLITUDE: f32 = 0.05;
const K: f32 = TAU / WAVELENGTH;

#[derive(Resource, Default)]
struct Capture {
    data: Vec<u8>,
    serial: u64,
}

fn step(app: &mut App) {
    app.update();
    assert!(app.should_exit().is_none(), "renderer must remain healthy");
    thread::sleep(Duration::from_millis(2));
}

fn capture(app: &mut App) -> Vec<u8> {
    let first = app.world().resource::<Capture>().serial;
    let deadline = Instant::now() + Duration::from_secs(30);
    while app.world().resource::<Capture>().serial < first + 4 {
        step(app);
        assert!(Instant::now() < deadline, "readback timed out");
    }
    app.world().resource::<Capture>().data.clone()
}

/// Half float bits for `value`, rounded to nearest. Covers the normal range used here.
fn half_bits(value: f32) -> u16 {
    if value == 0.0 {
        return 0;
    }
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mantissa = bits & 0x7f_ffff;
    let rounded = (mantissa + 0x1000) >> 13;
    let (exponent, rounded) = if rounded == 0x400 {
        (exponent + 1, 0)
    } else {
        (exponent, rounded)
    };
    assert!(
        (1..31).contains(&exponent),
        "{value} is outside the half range"
    );
    sign | ((exponent as u16) << 10) | rounded as u16
}

fn half(bits: u16) -> f32 {
    let exponent = ((bits >> 10) & 0x1f) as i32;
    let mantissa = (bits & 0x3ff) as f32;
    let magnitude = match exponent {
        0 => mantissa * 2f32.powi(-24),
        e => (1.0 + mantissa / 1024.0) * 2f32.powi(e - 15),
    };
    if bits & 0x8000 != 0 {
        -magnitude
    } else {
        magnitude
    }
}

fn focal_depth() -> f32 {
    1.0 / ((1.0 - 1.0 / IOR) * AMPLITUDE * K * K)
}

/// World X of texel `i` of the single slice, which is centred on the camera at the origin.
fn texel_x(i: u32) -> f32 {
    (i as f32 + 0.5 - RESOLUTION as f32 * 0.5) * TEXEL
}

/// Where the ray through surface point `x` lands at `depth`, by exact refraction.
fn landing(x: f32, depth: f32) -> f32 {
    let n = Vec2::new(AMPLITUDE * K * (K * x).sin(), 1.0).normalize();
    let i = Vec2::NEG_Y;
    let eta = 1.0 / IOR;
    let cos_i = -i.dot(n);
    let cos_t = (1.0 - eta * eta * (1.0 - cos_i * cos_i)).sqrt();
    let t = eta * i + (eta * cos_i - cos_t) * n;
    x + t.x * (AMPLITUDE * (K * x).cos() + depth) / -t.y
}

/// Light landing under the one-texel tent around receiver point `c`, relative to a level surface.
fn expected_intensity(c: f32, depth: f32) -> f32 {
    let samples = 20000;
    let span = 4.0 * TEXEL;
    (0..samples)
        .map(|j| {
            let x = c - span + 2.0 * span * (j as f32 + 0.5) / samples as f32;
            (1.0 - ((landing(x, depth) - c) / TEXEL).abs()).max(0.0)
        })
        .sum::<f32>()
        * (2.0 * span / samples as f32)
        / TEXEL
}

fn caustics_app() -> App {
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
        resolution: RESOLUTION,
        slice_count: 1,
        base_texel_size: TEXEL,
    })
    .insert_resource(WaterCausticsSettings {
        enabled: true,
        slice: 0,
        receiver_depth: 0.5 * focal_depth(),
        rays_per_texel: 8,
        ior: IOR,
    })
    .insert_resource(GlobalAmbientLight::NONE)
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
    app.world_mut().spawn((
        DirectionalLight {
            illuminance: 2.0,
            ..default()
        },
        Transform::default().looking_to(Vec3::NEG_Y, Vec3::Z),
    ));
    app
}

/// Fills the LOD data with the sinusoidal surface. The data exists after the first update.
fn set_sinusoid(app: &mut App) {
    let displacement = app.world().resource::<WaterLodData>().displacement.clone();
    let mut images = app.world_mut().resource_mut::<Assets<Image>>();
    let mut image = images.get_mut(&displacement).unwrap();
    let data = image.data.as_mut().unwrap();
    for (n, texel) in data.chunks_exact_mut(8).enumerate() {
        let x = texel_x(n as u32 % RESOLUTION);
        let lanes = [
            -AMPLITUDE * K * (K * x).sin(),
            0.0,
            AMPLITUDE * (K * x).cos(),
            0.0,
        ];
        for (lane, value) in lanes.iter().enumerate() {
            texel[lane * 2..lane * 2 + 2].copy_from_slice(&half_bits(*value).to_le_bytes());
        }
    }
}

fn observe_into_capture(app: &mut App, readback: Readback) {
    app.world_mut().spawn(readback).observe(
        |event: On<ReadbackComplete>, mut capture: ResMut<Capture>| {
            capture.data = event.data.clone();
            capture.serial += 1;
        },
    );
}

#[test]
#[ignore = "requires a graphics adapter"]
fn wave_crests_focus_sunlight_at_the_lens_depth() {
    let mut app = caustics_app();
    app.world_mut().spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 10.0, 0.0).looking_at(Vec3::ZERO, Vec3::Z),
    ));
    step(&mut app);
    set_sinusoid(&mut app);
    let map = app.world().resource::<WaterCaustics>().map.clone();
    observe_into_capture(&mut app, Readback::texture(map));
    let map: Vec<f32> = capture(&mut app)
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    assert_eq!(map.len(), (RESOLUTION * RESOLUTION) as usize);

    // Columns away from the slice edges, where rays from outside the slice would be missing.
    let margin = 8;
    let column = |i: u32| {
        let rows = margin..RESOLUTION - margin;
        rows.clone()
            .map(|j| map[(j * RESOLUTION + i) as usize])
            .sum::<f32>()
            / rows.len() as f32
    };
    let interior: Vec<u32> = (margin..RESOLUTION - margin).collect();
    let mean = interior.iter().map(|&i| column(i)).sum::<f32>() / interior.len() as f32;
    eprintln!("focal depth {} m, mean intensity {mean}", focal_depth());
    assert!((mean - 1.0).abs() < 0.01, "mean intensity {mean}");

    let depth = 0.5 * focal_depth();
    let crest = interior
        .iter()
        .copied()
        .find(|&i| (K * texel_x(i)).cos() > 0.98)
        .unwrap();
    let trough = interior
        .iter()
        .copied()
        .find(|&i| (K * texel_x(i)).cos() < -0.98)
        .unwrap();
    for (label, i) in [("crest", crest), ("trough", trough)] {
        let measured = column(i);
        let expected = expected_intensity(texel_x(i), depth);
        eprintln!("{label}: {measured} against {expected}");
        assert!(
            (measured / expected - 1.0).abs() < 0.02,
            "{label}: {measured} against {expected}"
        );
    }
}

#[test]
#[ignore = "requires a graphics adapter"]
fn caustics_light_a_plate_on_the_receiver() {
    const SIZE: u32 = 256;
    const VIEW: f32 = 16.0;
    let depth = 0.5 * focal_depth();
    let mut app = caustics_app();
    // Clear water, so the plate shows only the sun and its caustics.
    app.insert_resource(WaterMedium(WaterMediumUniform {
        absorption: Vec3::ZERO,
        scattering: Vec3::ZERO,
        diffuse_attenuation: Vec3::ZERO,
        ..default()
    }));
    let mut image = Image::new_target_texture(SIZE, SIZE, TextureFormat::Rgba16Float, None);
    image.texture_descriptor.usage |= TextureUsages::COPY_SRC;
    let target = app.world_mut().resource_mut::<Assets<Image>>().add(image);
    app.world_mut().spawn((
        Camera3d::default(),
        WaterUnderwater,
        Hdr,
        Msaa::Off,
        Tonemapping::None,
        DebandDither::Disabled,
        Exposure { ev100: 0.0 },
        Projection::Orthographic(OrthographicProjection {
            scaling_mode: ScalingMode::Fixed {
                width: VIEW,
                height: VIEW,
            },
            ..OrthographicProjection::default_3d()
        }),
        Transform::from_xyz(0.0, 1.0 - depth, 0.0).looking_to(Vec3::NEG_Y, Vec3::Z),
        RenderTarget::Image(ImageRenderTarget {
            handle: target.clone(),
            scale_factor: 1.0,
        }),
    ));
    let (surface_mesh, plate_mesh) = {
        let mut assets = app.world_mut().resource_mut::<Assets<Mesh>>();
        (
            assets.add(Plane3d::default().mesh().size(32.0, 32.0)),
            assets.add(Plane3d::default().mesh().size(40.0, 40.0)),
        )
    };
    let surface_material = app
        .world_mut()
        .resource_mut::<Assets<WaterSurfaceMaterial>>()
        .add(WaterSurfaceMaterial::default());
    app.world_mut().spawn((
        Mesh3d(surface_mesh),
        MeshMaterial3d(surface_material),
        WaterSurfaceTag,
        NotShadowCaster,
    ));
    let plate_material = app
        .world_mut()
        .resource_mut::<Assets<StandardMaterial>>()
        .add(StandardMaterial {
            base_color: Color::WHITE,
            perceptual_roughness: 1.0,
            reflectance: 0.0,
            ..default()
        });
    app.world_mut().spawn((
        Mesh3d(plate_mesh),
        MeshMaterial3d(plate_material),
        Transform::from_xyz(0.0, -depth, 0.0),
    ));
    step(&mut app);
    set_sinusoid(&mut app);
    observe_into_capture(&mut app, Readback::texture(target));
    let pixels: Vec<f32> = capture(&mut app)
        .chunks_exact(8)
        .map(|p| half(u16::from_le_bytes([p[0], p[1]])))
        .collect();

    // Red lane along the middle rows. The view spans four wavelengths, so the mean over all
    // columns is the mean over whole periods.
    let column = |c: u32| {
        (SIZE / 2 - 8..SIZE / 2 + 8)
            .map(|r| pixels[(r * SIZE + c) as usize])
            .sum::<f32>()
            / 16.0
    };
    let mean = (0..SIZE).map(column).sum::<f32>() / SIZE as f32;
    let pixel = VIEW / SIZE as f32;
    // Pixel columns either side of x = 0, a crest, and of x = 2, a trough. The map texels nearest
    // them sit at x = 0.125 and x = 2.125, and the map is symmetric about both lines.
    let crest = 0.5 * (column(SIZE / 2 - 1) + column(SIZE / 2)) / mean;
    let trough_column = SIZE / 2 + (2.0 / pixel) as u32;
    let trough = 0.5 * (column(trough_column - 1) + column(trough_column)) / mean;
    let expected_crest = expected_intensity(0.125, depth);
    let expected_trough = expected_intensity(2.125, depth);
    eprintln!(
        "plate crest {crest} against {expected_crest}, trough {trough} against {expected_trough}"
    );
    assert!(mean > 0.0, "plate must be lit");
    assert!(
        (crest / expected_crest - 1.0).abs() < 0.03,
        "crest {crest} against {expected_crest}"
    );
    assert!(
        (trough / expected_trough - 1.0).abs() < 0.03,
        "trough {trough} against {expected_trough}"
    );
}

#[test]
#[ignore = "requires a graphics adapter"]
fn light_shafts_follow_the_caustics() {
    const SIZE: u32 = 256;
    const VIEW: f32 = 16.0;
    let depth = 0.5 * focal_depth();
    let mut app = caustics_app();
    // Clear enough that sunlight still scatters at the receiver depth.
    app.insert_resource(WaterMedium(WaterMediumUniform::jerlov(
        JerlovWaterType::OceanIB,
    )));
    let mut image = Image::new_target_texture(SIZE, SIZE, TextureFormat::Rgba16Float, None);
    image.texture_descriptor.usage |= TextureUsages::COPY_SRC;
    let target = app.world_mut().resource_mut::<Assets<Image>>().add(image);
    // Level view along the crests from the receiver depth. Each pixel column holds one X, so its
    // single-scattered light follows the caustic intensity of that X.
    app.world_mut().spawn((
        Camera3d::default(),
        WaterUnderwater,
        Hdr,
        Msaa::Off,
        Tonemapping::None,
        DebandDither::Disabled,
        Exposure { ev100: 0.0 },
        Projection::Orthographic(OrthographicProjection {
            scaling_mode: ScalingMode::Fixed {
                width: VIEW,
                height: VIEW,
            },
            ..OrthographicProjection::default_3d()
        }),
        Transform::from_xyz(0.0, -depth, 0.0),
        RenderTarget::Image(ImageRenderTarget {
            handle: target.clone(),
            scale_factor: 1.0,
        }),
    ));
    let mesh = app
        .world_mut()
        .resource_mut::<Assets<Mesh>>()
        .add(Plane3d::default().mesh().size(32.0, 32.0));
    let material = app
        .world_mut()
        .resource_mut::<Assets<WaterSurfaceMaterial>>()
        .add(WaterSurfaceMaterial::default());
    app.world_mut().spawn((
        Mesh3d(mesh),
        MeshMaterial3d(material),
        WaterSurfaceTag,
        NotShadowCaster,
    ));
    step(&mut app);
    set_sinusoid(&mut app);
    // Deep water, so the view runs on past the receiver.
    let foam = app.world().resource::<WaterLodData>().foam.clone();
    {
        let mut images = app.world_mut().resource_mut::<Assets<Image>>();
        let mut image = images.get_mut(&foam).unwrap();
        for texel in image.data.as_mut().unwrap().chunks_exact_mut(8) {
            texel[2..4].copy_from_slice(&half_bits(1000.0).to_le_bytes());
        }
    }
    observe_into_capture(&mut app, Readback::texture(target));
    let green = |data: Vec<u8>| -> Vec<f32> {
        data.chunks_exact(8)
            .map(|p| half(u16::from_le_bytes([p[2], p[3]])))
            .collect()
    };
    let with = green(capture(&mut app));
    app.world_mut()
        .resource_mut::<WaterCausticsSettings>()
        .enabled = false;
    let without = green(capture(&mut app));

    let pixel = VIEW / SIZE as f32;
    let difference = |c: u32| {
        (SIZE / 2 - 8..SIZE / 2 + 8)
            .map(|r| {
                let i = (r * SIZE + c) as usize;
                with[i] - without[i]
            })
            .sum::<f32>()
    };
    let pair = |c: u32| difference(c - 1) + difference(c);
    let crest = pair(SIZE / 2);
    let trough = pair(SIZE / 2 + (2.0 / pixel) as u32);
    let measured = crest / trough;
    let expected =
        (expected_intensity(0.125, depth) - 1.0) / (expected_intensity(2.125, depth) - 1.0);
    eprintln!("shaft contrast {measured} against {expected}");
    assert!(
        crest > 0.0 && trough < 0.0,
        "crest {crest}, trough {trough}"
    );
    assert!(
        (measured / expected - 1.0).abs() < 0.05,
        "shaft contrast {measured} against {expected}"
    );
}

#[test]
#[ignore = "requires a graphics adapter"]
fn caustics_light_a_bed_seen_from_above() {
    const SIZE: u32 = 256;
    const VIEW: f32 = 16.0;
    let depth = 0.5 * focal_depth();
    let mut app = caustics_app();
    let mut image = Image::new_target_texture(SIZE, SIZE, TextureFormat::Rgba16Float, None);
    image.texture_descriptor.usage |= TextureUsages::COPY_SRC;
    let target = app.world_mut().resource_mut::<Assets<Image>>().add(image);
    app.world_mut().spawn((
        Camera3d::default(),
        WaterTransmission,
        Hdr,
        Msaa::Off,
        Tonemapping::None,
        DebandDither::Disabled,
        Exposure { ev100: 0.0 },
        Projection::Orthographic(OrthographicProjection {
            scaling_mode: ScalingMode::Fixed {
                width: VIEW,
                height: VIEW,
            },
            ..OrthographicProjection::default_3d()
        }),
        Transform::from_xyz(0.0, 10.0, 0.0).looking_to(Vec3::NEG_Y, Vec3::Z),
        RenderTarget::Image(ImageRenderTarget {
            handle: target.clone(),
            scale_factor: 1.0,
        }),
    ));
    let (surface_mesh, plate_mesh) = {
        let mut assets = app.world_mut().resource_mut::<Assets<Mesh>>();
        (
            assets.add(Plane3d::default().mesh().size(32.0, 32.0)),
            assets.add(Plane3d::default().mesh().size(40.0, 40.0)),
        )
    };
    // Clear water with no reflectance and no refraction, so each pixel shows the plate straight
    // below it, lit by the sun alone.
    let surface_material = app
        .world_mut()
        .resource_mut::<Assets<WaterSurfaceMaterial>>()
        .add(WaterSurfaceMaterial {
            base: StandardMaterial {
                base_color: Color::WHITE,
                specular_transmission: 1.0,
                reflectance: 0.0,
                ior: 1.0,
                opaque_render_method: bevy::material::OpaqueRendererMethod::Forward,
                ..default()
            },
            extension: WaterSurfaceExtension {
                optics: WaterOpticsUniform {
                    enabled: 1,
                    extinction: Vec3::ZERO,
                    scattering: Vec3::ZERO,
                    refraction_strength: 0.0,
                    ssr_strength: 0.0,
                    contact_fade: 0.001,
                    ..default()
                },
                ..default()
            },
        });
    app.world_mut().spawn((
        Mesh3d(surface_mesh),
        MeshMaterial3d(surface_material),
        WaterSurfaceTag,
        NotShadowCaster,
    ));
    let plate_material = app
        .world_mut()
        .resource_mut::<Assets<StandardMaterial>>()
        .add(StandardMaterial {
            base_color: Color::WHITE,
            perceptual_roughness: 1.0,
            reflectance: 0.0,
            ..default()
        });
    app.world_mut().spawn((
        Mesh3d(plate_mesh),
        MeshMaterial3d(plate_material),
        Transform::from_xyz(0.0, -depth, 0.0),
    ));
    step(&mut app);
    set_sinusoid(&mut app);
    // The water column reaches past the plate.
    let foam = app.world().resource::<WaterLodData>().foam.clone();
    {
        let mut images = app.world_mut().resource_mut::<Assets<Image>>();
        let mut image = images.get_mut(&foam).unwrap();
        for texel in image.data.as_mut().unwrap().chunks_exact_mut(8) {
            texel[2..4].copy_from_slice(&half_bits(1000.0).to_le_bytes());
        }
    }
    observe_into_capture(&mut app, Readback::texture(target));
    let red = |data: Vec<u8>| -> Vec<f32> {
        data.chunks_exact(8)
            .map(|p| half(u16::from_le_bytes([p[0], p[1]])))
            .collect()
    };
    let with = red(capture(&mut app));
    app.world_mut()
        .resource_mut::<WaterCausticsSettings>()
        .enabled = false;
    let without = red(capture(&mut app));

    let pixel = VIEW / SIZE as f32;
    let sum = |image: &[f32], c: u32| {
        (SIZE / 2 - 8..SIZE / 2 + 8)
            .map(|r| image[(r * SIZE + c) as usize])
            .sum::<f32>()
    };
    // The sun is the only light, so the plate's brightness scales by the caustic intensity.
    let ratio =
        |c: u32| (sum(&with, c - 1) + sum(&with, c)) / (sum(&without, c - 1) + sum(&without, c));
    let crest = ratio(SIZE / 2);
    let trough = ratio(SIZE / 2 + (2.0 / pixel) as u32);
    let expected_crest = expected_intensity(0.125, depth);
    let expected_trough = expected_intensity(2.125, depth);
    eprintln!(
        "bed from above: crest {crest} against {expected_crest}, trough {trough} against {expected_trough}, plain {}",
        sum(&without, SIZE / 2) / 16.0
    );
    assert!(sum(&without, SIZE / 2) > 0.0, "plate must be lit");
    assert!(
        (crest / expected_crest - 1.0).abs() < 0.03,
        "crest {crest} against {expected_crest}"
    );
    assert!(
        (trough / expected_trough - 1.0).abs() < 0.03,
        "trough {trough} against {expected_trough}"
    );
}
