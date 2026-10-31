//! Optical composition against an opaque receiver. Run explicitly with a graphics adapter.
use bevy::{
    app::PluginsState,
    camera::{ImageRenderTarget, RenderTarget, ScalingMode},
    core_pipeline::tonemapping::Tonemapping,
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
    thread,
    time::{Duration, Instant},
};

#[derive(Resource, Default)]
struct Capture {
    pixels: Vec<u8>,
    serial: u64,
}

fn step(app: &mut App) {
    app.update();
    assert!(app.should_exit().is_none(), "renderer must remain healthy");
    thread::sleep(Duration::from_millis(5));
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

fn pixel(image: &[u8], width: usize, x: usize, y: usize) -> &[u8] {
    &image[(y * width + x) * 4..(y * width + x) * 4 + 4]
}
fn linear(value: u8) -> f32 {
    let s = value as f32 / 255.0;
    if s <= 0.04045 {
        s / 12.92
    } else {
        ((s + 0.055) / 1.055).powf(2.4)
    }
}

#[test]
#[ignore = "requires a graphics adapter"]
fn transmission_preserves_bottom_and_measures_metric_thickness() {
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
        resolution: 16,
        slice_count: 1,
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
        64,
        64,
        TextureFormat::Rgba8UnormSrgb,
        Some(TextureFormat::Rgba8UnormSrgb),
    );
    image.texture_descriptor.usage |= TextureUsages::COPY_SRC;
    let target = app.world_mut().resource_mut::<Assets<Image>>().add(image);
    app.world_mut().spawn((
        Camera3d::default(),
        WaterTransmission,
        Msaa::Off,
        Tonemapping::None,
        Projection::Orthographic(OrthographicProjection {
            scaling_mode: ScalingMode::Fixed {
                width: 6.0,
                height: 6.0,
            },
            ..OrthographicProjection::default_3d()
        }),
        Transform::from_xyz(0.0, 4.0, 0.0).looking_at(Vec3::ZERO, Vec3::NEG_Z),
        RenderTarget::Image(ImageRenderTarget {
            handle: target.clone(),
            scale_factor: 1.0,
        }),
    ));
    let ground_mesh = app
        .world_mut()
        .resource_mut::<Assets<Mesh>>()
        .add(Plane3d::default().mesh().size(4.0, 8.0));
    let mut ground = Vec::new();
    for (x, color) in [
        (-2.0, Color::srgb(0.8, 0.1, 0.1)),
        (2.0, Color::srgb(0.1, 0.8, 0.1)),
    ] {
        let material = app
            .world_mut()
            .resource_mut::<Assets<StandardMaterial>>()
            .add(StandardMaterial {
                base_color: color,
                unlit: true,
                ..default()
            });
        ground.push(
            app.world_mut()
                .spawn((
                    Mesh3d(ground_mesh.clone()),
                    MeshMaterial3d(material),
                    Transform::from_xyz(x, -0.1, 0.0),
                ))
                .id(),
        );
    }
    app.world_mut()
        .spawn(Readback::texture(target.clone()))
        .observe(
            |event: On<ReadbackComplete>, mut capture: ResMut<Capture>| {
                capture.pixels = event.data.clone();
                capture.serial += 1;
            },
        );
    step(&mut app);
    let foam = app.world().resource::<WaterLodData>().foam.clone();
    {
        let mut images = app.world_mut().resource_mut::<Assets<Image>>();
        let mut foam_image = images.get_mut(&foam).unwrap();
        for texel in foam_image.data.as_mut().unwrap().chunks_exact_mut(8) {
            texel[2..4].copy_from_slice(&0x2e66_u16.to_le_bytes());
        }
    }
    let baseline = capture(&mut app, 60);
    assert!(
        pixel(&baseline, 64, 16, 32)[0] > 150 && pixel(&baseline, 64, 48, 32)[1] > 150,
        "colored receivers must render: left {:?}, right {:?}",
        pixel(&baseline, 64, 16, 32),
        pixel(&baseline, 64, 48, 32)
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
                    debug_view: 1,
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
    let water = app
        .world_mut()
        .spawn((
            Mesh3d(mesh),
            MeshMaterial3d(material.clone()),
            WaterSurfaceTag,
            bevy::light::NotShadowCaster,
        ))
        .id();
    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        let image = capture(&mut app, 10);
        let left = pixel(&image, 64, 16, 32);
        let right = pixel(&image, 64, 48, 32);
        if left[0].abs_diff(left[1]) <= 1
            && right[0].abs_diff(right[1]) <= 1
            && left[0].abs_diff(39) <= 3
            && left[0].abs_diff(right[0]) <= 1
        {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "optical-distance view must measure the 10 cm receiver"
        );
    }
    {
        let mut materials = app
            .world_mut()
            .resource_mut::<Assets<WaterSurfaceMaterial>>();
        let mut water = materials.get_mut(&material).unwrap();
        water.extension.optics.debug_view = 4;
        water.extension.optics.refraction_strength = 1.0;
        water.base.ior = 1.333;
    }
    let hits = capture(&mut app, 30);
    assert!(
        pixel(&hits, 64, 16, 32)[0] > 250,
        "refraction must find the submerged receiver"
    );
    {
        let mut materials = app
            .world_mut()
            .resource_mut::<Assets<WaterSurfaceMaterial>>();
        let mut water = materials.get_mut(&material).unwrap();
        water.extension.optics.debug_view = 0;
        water.extension.optics.refraction_strength = 0.0;
    }
    let clear = capture(&mut app, 30);
    let error = clear
        .iter()
        .zip(&baseline)
        .map(|(a, b)| a.abs_diff(*b))
        .max()
        .unwrap();
    assert!(
        error <= 2,
        "clear water must preserve receiver colors, max byte error {error}"
    );
    app.world_mut()
        .resource_mut::<Assets<WaterSurfaceMaterial>>()
        .get_mut(&material)
        .unwrap()
        .extension
        .optics
        .extinction = Vec3::splat(2.0);
    let attenuated = capture(&mut app, 30);
    for (x, channel) in [(16, 0), (48, 1)] {
        let ratio = linear(pixel(&attenuated, 64, x, 32)[channel])
            / linear(pixel(&baseline, 64, x, 32)[channel]);
        assert!(
            (ratio - (-0.2_f32).exp()).abs() < 0.025,
            "Beer-Lambert attenuation must use a 0.1 m path: {ratio}"
        );
    }
    app.world_mut()
        .resource_mut::<Assets<WaterSurfaceMaterial>>()
        .get_mut(&material)
        .unwrap()
        .extension
        .optics
        .debug_view = 1;
    app.world_mut()
        .resource_mut::<Assets<Image>>()
        .get_mut(&target)
        .unwrap()
        .resize(bevy::render::render_resource::Extent3d {
            width: 128,
            height: 64,
            depth_or_array_layers: 1,
        });
    let resized = capture(&mut app, 30);
    assert_eq!(resized.len(), 128 * 64 * 4);
    assert!(
        pixel(&resized, 128, 32, 32)[0].abs_diff(39) <= 3,
        "opaque depth must resize with the camera target"
    );
    app.world_mut()
        .resource_mut::<Assets<WaterSurfaceMaterial>>()
        .get_mut(&material)
        .unwrap()
        .extension
        .optics
        .debug_view = 0;
    for entity in ground {
        app.world_mut()
            .get_mut::<Transform>(entity)
            .unwrap()
            .translation
            .y = 0.1;
    }
    let emerged = capture(&mut app, 30);
    app.world_mut()
        .get_mut::<Visibility>(water)
        .unwrap()
        .set_if_neq(Visibility::Hidden);
    let dry = capture(&mut app, 30);
    assert_eq!(
        emerged, dry,
        "terrain above the interface must occlude water without a height offset"
    );
    eprintln!(
        "Validated bottom colors, metric attenuation, target resize, and emergent terrain occlusion"
    );
}
