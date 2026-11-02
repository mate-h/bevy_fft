//! GPU selection parity. Run explicitly on a machine with a graphics adapter.

use bevy::{
    app::PluginsState,
    camera::{ImageRenderTarget, RenderTarget, primitives::Frustum},
    prelude::*,
    render::{
        gpu_readback::{Readback, ReadbackComplete},
        render_resource::{TextureFormat, TextureUsages},
        view::NoIndirectDrawing,
    },
    window::ExitCondition,
    winit::WinitPlugin,
};
use bevy_fft::water::{WaterPlugin, WaterQuadtree, WaterQuadtreeBuffer, WaterSurfaceMaterial};
use std::{
    thread,
    time::{Duration, Instant},
};

#[derive(Resource, Default)]
struct Captured(Vec<u8>);

#[derive(Resource, Default)]
struct CapturedImage(Vec<u8>);

#[test]
#[ignore = "requires a graphics adapter"]
fn gpu_tiles_match_cpu_selection() {
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
    .insert_resource(bevy::render::error_handler::RenderErrorHandler(
        |error, _, _| panic!("GPU validation failed: {error:?}"),
    ))
    .init_resource::<Captured>()
    .init_resource::<CapturedImage>();
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
    let camera = app
        .world_mut()
        .spawn((
            Camera3d::default(),
            RenderTarget::Image(ImageRenderTarget {
                handle: target.clone(),
                scale_factor: 1.0,
            }),
            Transform::default(),
            Msaa::Off,
        ))
        .id();
    let tree = app.world_mut().spawn(WaterQuadtree::default()).id();
    app.update();
    let handle = app
        .world()
        .get::<WaterQuadtreeBuffer>(tree)
        .unwrap()
        .0
        .clone();
    app.world_mut().spawn(Readback::buffer(handle)).observe(
        |event: On<ReadbackComplete>, mut captured: ResMut<Captured>| {
            captured.0 = event.data.clone();
        },
    );
    for (i, (eye, cull, capacity)) in [
        (Vec3::new(3.0, 2.0, -5.0), false, 1024),
        (Vec3::new(-129.0, 7.0, 41.0), true, 1024),
        (Vec3::new(153.0, 500.0, -87.0), true, 1024),
        (Vec3::new(-513.0, 1.5, -257.0), false, 7),
    ]
    .into_iter()
    .enumerate()
    {
        let q = WaterQuadtree {
            frustum_cull: cull,
            max_tiles: capacity,
            levels: 9 + i as u32,
            ..default()
        };
        app.world_mut().entity_mut(tree).insert(q);
        *app.world_mut().get_mut::<Transform>(camera).unwrap() =
            Transform::from_translation(eye).looking_at(eye + Vec3::new(0.4, -0.15, -1.0), Vec3::Y);
        app.world_mut().resource_mut::<Captured>().0.clear();
        let start = Instant::now();
        loop {
            app.update();
            let data = &app.world().resource::<Captured>().0;
            if data.len() >= 32
                && read_f32(data, 0) == eye.x
                && read_f32(data, 4) == eye.y
                && read_f32(data, 8) == eye.z
            {
                let frustum = app.world().get::<Frustum>(camera).unwrap();
                let (expected, dropped) = q.select(eye, |bounds| {
                    !cull || frustum.intersects_obb_identity(bounds)
                });
                let count = read_u32(data, 16) as usize;
                assert_eq!(count, expected.len(), "tile count at {eye}");
                assert_eq!(read_u32(data, 28), dropped);
                assert_eq!(read_u32(data, 20), q.tile_cells);
                assert_eq!(read_u32(data, 24), 1);
                if dropped == 0 {
                    let mut expected = expected
                        .iter()
                        .map(|t| [t.min.x, t.min.y, t.size, t.level as f32])
                        .collect::<Vec<_>>();
                    let mut actual = (0..count)
                        .map(|i| std::array::from_fn(|j| read_f32(data, 32 + i * 32 + j * 4)))
                        .collect::<Vec<[f32; 4]>>();
                    let sort = |a: &[f32; 4], b: &[f32; 4]| {
                        a[0].total_cmp(&b[0])
                            .then(a[1].total_cmp(&b[1]))
                            .then(a[2].total_cmp(&b[2]))
                    };
                    expected.sort_by(sort);
                    actual.sort_by(sort);
                    assert_eq!(actual, expected, "tile bounds at {eye}");
                    for i in 0..count {
                        let level = read_f32(data, 32 + i * 32 + 12) as u32;
                        let (lo, hi) = q.morph_band(level);
                        assert!((read_f32(data, 32 + i * 32 + 16) - lo).abs() <= hi * 1e-6);
                        assert!((read_f32(data, 32 + i * 32 + 20) - hi).abs() <= hi * 1e-6);
                    }
                }
                eprintln!("GPU selection matches CPU: {count} tiles, {dropped} dropped at {eye}");
                break;
            }
            assert!(
                start.elapsed() < Duration::from_secs(60),
                "GPU readback timed out at {eye}"
            );
            thread::sleep(Duration::from_millis(5));
        }
    }

    let material = app
        .world_mut()
        .resource_mut::<Assets<WaterSurfaceMaterial>>()
        .add(WaterSurfaceMaterial {
            base: StandardMaterial {
                base_color: Color::srgb(0.05, 0.25, 0.7),
                unlit: true,
                ..default()
            },
            extension: default(),
        });
    app.world_mut()
        .entity_mut(tree)
        .insert((WaterQuadtree::default(), MeshMaterial3d(material)));
    *app.world_mut().get_mut::<Transform>(camera).unwrap() =
        Transform::from_xyz(3.0, 2.0, -5.0).looking_at(Vec3::ZERO, Vec3::Y);
    app.world_mut()
        .spawn(Readback::texture(target.clone()))
        .observe(
            |event: On<ReadbackComplete>, mut captured: ResMut<CapturedImage>| {
                captured.0 = event.data.clone();
            },
        );
    let indirect = capture_settled_image(&mut app);
    assert!(
        indirect.chunks_exact(4).any(|p| p != &indirect[..4]),
        "water must be visible in the indirect draw"
    );
    app.world_mut()
        .get_mut::<WaterQuadtree>(tree)
        .unwrap()
        .gpu_selection = false;
    let cpu = capture_settled_image(&mut app);
    assert_eq!(
        indirect, cpu,
        "GPU indirect and CPU pool must render the same flat surface"
    );
    app.world_mut()
        .get_mut::<WaterQuadtree>(tree)
        .unwrap()
        .gpu_selection = true;
    app.world_mut().despawn(camera);
    app.world_mut().spawn((
        Camera3d::default(),
        RenderTarget::Image(ImageRenderTarget {
            handle: target,
            scale_factor: 1.0,
        }),
        Transform::from_xyz(3.0, 2.0, -5.0).looking_at(Vec3::ZERO, Vec3::Y),
        Msaa::Off,
        NoIndirectDrawing,
    ));
    let direct = capture_settled_image(&mut app);
    assert_eq!(
        indirect, direct,
        "the direct-draw fallback must preserve the surface"
    );
    eprintln!("GPU indirect, CPU pool, and direct fallback render identical images");
}

fn capture_settled_image(app: &mut App) -> Vec<u8> {
    for _ in 0..120 {
        app.update();
        thread::sleep(Duration::from_millis(5));
    }
    let image = app.world().resource::<CapturedImage>().0.clone();
    assert!(!image.is_empty(), "render target readback must complete");
    image
}

fn read_u32(data: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap())
}
fn read_f32(data: &[u8], offset: usize) -> f32 {
    f32::from_bits(read_u32(data, offset))
}
