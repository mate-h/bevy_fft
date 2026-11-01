//! The underwater medium of the water viewer. Run explicitly with a graphics adapter.
//!
//! A sun straight overhead lights deep water. Looking level into open water, the medium shows only
//! the light it scatters. Monzon et al. (2024) Eq. 9 gives the multiple-scattered part in closed
//! form, and it falls with depth as the downwelling irradiance, `exp(-K_d y)` per channel. The
//! single-scattered sunlight of Eq. 5 adds a Henyey-Greenstein share. A lit plate under water
//! dims by `exp(-K_d y)` with its depth, as in Eq. 10.
use bevy::{
    app::PluginsState,
    camera::{Exposure, Hdr, ImageRenderTarget, RenderTarget},
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
    f32::consts::PI,
    thread,
    time::{Duration, Instant},
};

const SIZE: u32 = 256;
const IOR: f32 = 1.333;
/// Sun illuminance in lux. Small, so the exposed result stays near one.
const SUN: f32 = 6.0;
/// Henyey-Greenstein mean cosine and march reach of `medium.wesl`.
const PHASE_G: f32 = 0.8;
const SINGLE_SCATTERING_REACH: f32 = 8.0;
/// 1000 as a half float, the water column written into the LOD data.
const THOUSAND_F16: u16 = 0x63d0;

#[derive(Resource, Default)]
struct Capture {
    pixels: Vec<Vec3>,
    serial: u64,
}

#[derive(Component)]
struct Viewer;

#[derive(Component)]
struct Plate;

fn step(app: &mut App) {
    app.update();
    assert!(app.should_exit().is_none(), "renderer must remain healthy");
    thread::sleep(Duration::from_millis(2));
}

fn capture(app: &mut App) -> Vec<Vec3> {
    let first = app.world().resource::<Capture>().serial;
    let deadline = Instant::now() + Duration::from_secs(30);
    while app.world().resource::<Capture>().serial < first + 5 {
        step(app);
        assert!(Instant::now() < deadline, "image readback timed out");
    }
    app.world().resource::<Capture>().pixels.clone()
}

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

fn medium_app(shadows: bool) -> App {
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
    let mut image = Image::new_target_texture(SIZE, SIZE, TextureFormat::Rgba16Float, None);
    image.texture_descriptor.usage |= TextureUsages::COPY_SRC;
    let target = app.world_mut().resource_mut::<Assets<Image>>().add(image);
    app.world_mut().spawn((
        Viewer,
        Camera3d::default(),
        WaterUnderwater,
        Hdr,
        Msaa::Off,
        Tonemapping::None,
        DebandDither::Disabled,
        Exposure { ev100: 0.0 },
        Transform::from_xyz(0.0, -2.0, 0.0),
        RenderTarget::Image(ImageRenderTarget {
            handle: target.clone(),
            scale_factor: 1.0,
        }),
    ));
    app.world_mut().spawn(Readback::texture(target)).observe(
        |event: On<ReadbackComplete>, mut capture: ResMut<Capture>| {
            capture.pixels = event
                .data
                .chunks_exact(8)
                .map(|p| {
                    let lane = |i: usize| half(u16::from_le_bytes([p[i * 2], p[i * 2 + 1]]));
                    Vec3::new(lane(0), lane(1), lane(2))
                })
                .collect();
            capture.serial += 1;
        },
    );
    app.world_mut().spawn((
        DirectionalLight {
            illuminance: SUN,
            shadow_maps_enabled: shadows,
            ..default()
        },
        Transform::default().looking_to(Vec3::NEG_Y, Vec3::Z),
    ));
    let mesh = app
        .world_mut()
        .resource_mut::<Assets<Mesh>>()
        .add(Plane3d::default().mesh().size(400.0, 400.0));
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
    // Deep water everywhere, so no light returns from a bed.
    let foam = app.world().resource::<WaterLodData>().foam.clone();
    let mut images = app.world_mut().resource_mut::<Assets<Image>>();
    for texel in images
        .get_mut(&foam)
        .unwrap()
        .data
        .as_mut()
        .unwrap()
        .chunks_exact_mut(8)
    {
        texel[2..4].copy_from_slice(&THOUSAND_F16.to_le_bytes());
    }
    app
}

fn place_viewer(app: &mut App, transform: Transform) {
    let mut viewer = app
        .world_mut()
        .query_filtered::<&mut Transform, With<Viewer>>();
    *viewer.single_mut(app.world_mut()).unwrap() = transform;
}

fn centre(pixels: &[Vec3]) -> Vec3 {
    let mut sum = Vec3::ZERO;
    for y in SIZE / 2 - 2..SIZE / 2 + 2 {
        for x in SIZE / 2 - 2..SIZE / 2 + 2 {
            sum += pixels[(y * SIZE + x) as usize];
        }
    }
    sum / 16.0
}

/// Exposed radiance of open water seen level from depth `y` under an overhead sun: the
/// multiple-scattered part of Eq. 9, then the single-scattered part of Eq. 5.
fn open_water(medium: &WaterMediumUniform, y: f32) -> (Vec3, Vec3) {
    let exposure = Exposure { ev100: 0.0 }.exposure();
    let r0 = ((IOR - 1.0) / (IOR + 1.0)).powi(2);
    let e0 = SUN * (1.0 - r0);
    let sigma_t = medium.extinction();
    let multiple = medium.scattering * e0 * exposure * (-medium.diffuse_attenuation * y).exp()
        / (4.0 * PI * sigma_t);
    let reach = SINGLE_SCATTERING_REACH / sigma_t.min_element();
    let g2 = PHASE_G * PHASE_G;
    let phase = (1.0 - g2) / (4.0 * PI * (1.0 + g2).powf(1.5));
    let single = medium.scattering
        * phase
        * e0
        * exposure
        * (-sigma_t * y).exp()
        * (Vec3::ONE - (-sigma_t * reach).exp())
        / sigma_t;
    (multiple, single)
}

fn assert_close(label: &str, measured: Vec3, expected: Vec3, tolerance: f32) {
    let error = (measured / expected - Vec3::ONE).abs().max_element();
    eprintln!("{label}: measured {measured}, expected {expected}");
    assert!(
        error < tolerance,
        "{label}: measured {measured}, expected {expected}"
    );
}

#[test]
#[ignore = "requires a graphics adapter"]
fn medium_light_follows_the_downwelling_irradiance() {
    let mut app = medium_app(false);
    let medium = WaterMediumUniform::jerlov(JerlovWaterType::OceanII);
    app.insert_resource(WaterMedium(medium));

    // Level view into open water. The centre rays barely descend, so Eq. 9 reduces to
    // sigma_s E_D(0) exp(-K_d y) / (4 pi sigma_t). Single scattering of the overhead sun at a right
    // angle adds sigma_s p(0) E_D(0) exp(-sigma_t y) / sigma_t over the reach of the march.
    let (multiple_2, single_2) = open_water(&medium, 2.0);
    let (multiple_6, single_6) = open_water(&medium, 6.0);
    place_viewer(&mut app, Transform::from_xyz(0.0, -2.0, 0.0));
    let shallow = centre(&capture(&mut app));
    assert_close("open water at 2 m", shallow, multiple_2 + single_2, 0.03);
    place_viewer(&mut app, Transform::from_xyz(0.0, -6.0, 0.0));
    let deep = centre(&capture(&mut app));
    assert_close("open water at 6 m", deep, multiple_6 + single_6, 0.03);
    assert_close(
        "multiple scattering from 2 m to 6 m",
        (deep - single_6) / (shallow - single_2),
        (-medium.diffuse_attenuation * 4.0).exp(),
        0.02,
    );

    // A lit plate seen from one meter above it, in water that absorbs but does not scatter. Its
    // depth dims it by exp(-K_d y).
    app.insert_resource(WaterMedium(WaterMediumUniform {
        scattering: Vec3::ZERO,
        ..medium
    }));
    let mesh = app
        .world_mut()
        .resource_mut::<Assets<Mesh>>()
        .add(Plane3d::default().mesh().size(20.0, 20.0));
    let material = app
        .world_mut()
        .resource_mut::<Assets<StandardMaterial>>()
        .add(StandardMaterial {
            base_color: Color::WHITE,
            perceptual_roughness: 1.0,
            reflectance: 0.0,
            ..default()
        });
    app.world_mut().spawn((
        Plate,
        Mesh3d(mesh),
        MeshMaterial3d(material),
        Transform::from_xyz(0.0, -3.0, 0.0),
    ));
    let plate_at = |app: &mut App, depth: f32| {
        let mut plate = app
            .world_mut()
            .query_filtered::<&mut Transform, With<Plate>>();
        plate.single_mut(app.world_mut()).unwrap().translation.y = -depth;
        place_viewer(
            app,
            Transform::from_xyz(0.0, 1.0 - depth, 0.0).looking_to(Vec3::NEG_Y, Vec3::Z),
        );
        centre(&capture(app))
    };
    let upper = plate_at(&mut app, 3.0);
    let lower = plate_at(&mut app, 7.0);
    assert!(upper.min_element() > 0.0, "plate must be lit, got {upper}");
    assert_close(
        "plate from 3 m to 7 m",
        lower / upper,
        (-medium.diffuse_attenuation * 4.0).exp(),
        0.01,
    );
}

fn block(pixels: &[Vec3], cx: u32, cy: u32) -> Vec3 {
    let mut sum = Vec3::ZERO;
    for y in cy - 2..cy + 2 {
        for x in cx - 2..cx + 2 {
            sum += pixels[(y * SIZE + x) as usize];
        }
    }
    sum / 16.0
}

#[test]
#[ignore = "requires a graphics adapter"]
fn objects_under_water_shadow_the_sunlight_they_block() {
    let mut app = medium_app(true);
    let medium = WaterMediumUniform::jerlov(JerlovWaterType::OceanII);
    app.insert_resource(WaterMedium(medium));
    // A slab over the half of the water left of the viewer, which looks along -Z.
    let mesh = app
        .world_mut()
        .resource_mut::<Assets<Mesh>>()
        .add(Plane3d::default().mesh().size(50.0, 100.0));
    let material = app
        .world_mut()
        .resource_mut::<Assets<StandardMaterial>>()
        .add(StandardMaterial::default());
    app.world_mut().spawn((
        Mesh3d(mesh),
        MeshMaterial3d(material),
        Transform::from_xyz(-25.0, -1.0, 0.0),
    ));
    place_viewer(&mut app, Transform::from_xyz(0.0, -3.0, 0.0));
    let pixels = capture(&mut app);
    let (multiple, single) = open_water(&medium, 3.0);
    let shaded = block(&pixels, SIZE / 4, SIZE / 2);
    let lit = block(&pixels, 3 * SIZE / 4, SIZE / 2);
    assert_close("under the slab", shaded, multiple, 0.03);
    assert_close("beside the slab", lit, multiple + single, 0.03);
}
