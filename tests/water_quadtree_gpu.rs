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

/// Cells per side of the test's water heightfield, one meter each, centred on the origin.
const WET_CELLS: u32 = 64;

fn float_image(texels: impl Iterator<Item = Vec4>) -> Image {
    let data = texels
        .flat_map(|v| v.to_array())
        .flat_map(f32::to_le_bytes)
        .collect::<Vec<_>>();
    Image::new(
        bevy::render::render_resource::Extent3d {
            width: WET_CELLS,
            height: WET_CELLS,
            depth_or_array_layers: 1,
        },
        bevy::render::render_resource::TextureDimension::D2,
        data,
        TextureFormat::Rgba32Float,
        bevy::asset::RenderAssetUsages::default(),
    )
}

/// Tiles of a GPU selection readback taken with the eye at `eye`, as min X, min Z and size.
fn selected_tiles(app: &mut App, eye: Vec3) -> Vec<Vec3> {
    app.world_mut().resource_mut::<Captured>().0.clear();
    let start = Instant::now();
    loop {
        app.update();
        let data = &app.world().resource::<Captured>().0;
        if data.len() >= 32 && read_f32(data, 0) == eye.x && read_f32(data, 8) == eye.z {
            let count = read_u32(data, 16) as usize;
            return (0..count)
                .map(|i| {
                    let at = |j: usize| read_f32(data, 32 + i * 32 + j * 4);
                    Vec3::new(at(0), at(1), at(2))
                })
                .collect();
        }
        assert!(
            start.elapsed() < Duration::from_secs(60),
            "GPU readback timed out"
        );
        thread::sleep(Duration::from_millis(5));
    }
}

#[test]
#[ignore = "requires a graphics adapter"]
fn gpu_selection_drops_tiles_with_no_water_under_them() {
    use bevy_fft::water::{
        WaterHeightfield, WaterHeightfieldDry, WaterLodClear, WaterLodData, WaterSurfaceExtension,
        WaterSurfaceUniform,
    };
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
    .insert_resource(bevy::render::error_handler::RenderErrorHandler(
        |error, _, _| panic!("GPU validation failed: {error:?}"),
    ))
    .init_resource::<Captured>();
    while app.plugins_state() == PluginsState::Adding {
        bevy::tasks::tick_global_task_pools_on_main_thread();
        thread::sleep(Duration::from_millis(1));
    }
    app.finish();
    app.cleanup();
    // A square of water one meter deep at the origin; everything else is dry.
    let n = (WET_CELLS * WET_CELLS) as usize;
    let mut images = app.world_mut().resource_mut::<Assets<Image>>();
    let bed_water = images.add(float_image((0..n).map(|_| Vec4::new(0.0, 1.0, 0.0, 0.0))));
    let velocity = images.add(float_image((0..n).map(|_| Vec4::ZERO)));
    app.world_mut().spawn(WaterHeightfield {
        dry: WaterHeightfieldDry::Level(0.0),
        ..WaterHeightfield::new(bed_water, velocity, UVec2::splat(WET_CELLS))
    });
    let eye = Vec3::new(10.0, 30.0, 10.0);
    let camera = app
        .world_mut()
        .spawn((Camera3d::default(), Transform::from_translation(eye)))
        .id();
    let material = |dry_depth: f32| WaterSurfaceMaterial {
        base: default(),
        extension: WaterSurfaceExtension {
            settings: WaterSurfaceUniform {
                dry_depth,
                ..WaterSurfaceExtension::default().settings
            },
            ..default()
        },
    };
    let culled = app
        .world_mut()
        .resource_mut::<Assets<WaterSurfaceMaterial>>()
        .add(material(0.05));
    let quadtree = WaterQuadtree {
        frustum_cull: false,
        dry_culling: true,
        ..default()
    };
    let tree = app
        .world_mut()
        .spawn((quadtree, MeshMaterial3d(culled.clone())))
        .id();
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
    // Let the heightfield reach the LOD slices.
    for _ in 0..30 {
        app.update();
    }
    let kept = selected_tiles(&mut app, eye);

    let (all, _) = quadtree.select(eye, |_| true);
    let lod = app.world().resource::<WaterLodData>().slices;
    let outer = lod.slices[lod.count as usize - 1];
    let outer_half = 0.5 * outer.w;
    // A node widened by the margin on each side is tested on cells less than its widened width,
    // so a kept tile can sit up to that width plus the margin from the water.
    let wet_half = 0.5 * WET_CELLS as f32;
    for tile in &kept {
        let (lo, size) = (tile.truncate(), tile.z);
        let reach = 3.0 * quadtree.horizontal_margin + size + 1.0;
        let near_water = lo.cmple(Vec2::splat(wet_half + reach)).all()
            && (lo + size).cmpge(Vec2::splat(-wet_half - reach)).all();
        // Widened by the margin as the selection widens it; past the outermost slice the surface
        // draws open water, so those tiles stay.
        let m = quadtree.horizontal_margin;
        let past_slices = (lo - m - outer.xy()).cmplt(Vec2::splat(-outer_half)).any()
            || (lo + size + m - outer.xy())
                .cmpgt(Vec2::splat(outer_half))
                .any();
        assert!(
            near_water || past_slices,
            "kept tile at {lo} of size {size} has no water near it"
        );
    }
    let near: Vec<_> = kept
        .iter()
        .filter(|t| t.x.abs() < wet_half && t.y.abs() < wet_half)
        .collect();
    assert!(!near.is_empty(), "the tiles over the water must stay");
    assert!(
        kept.len() * 5 < all.len() * 3,
        "dry culling kept {} of {} tiles",
        kept.len(),
        all.len()
    );
    eprintln!("dry culling kept {} of {} tiles", kept.len(), all.len());

    // Without a dry depth nothing is culled.
    let open = app
        .world_mut()
        .resource_mut::<Assets<WaterSurfaceMaterial>>()
        .add(material(0.0));
    app.world_mut()
        .entity_mut(tree)
        .insert(MeshMaterial3d(open));
    // A new eye, so the readback that answers is from after the change.
    let eye = eye + Vec3::X;
    app.world_mut()
        .get_mut::<Transform>(camera)
        .unwrap()
        .translation = eye;
    let unculled = selected_tiles(&mut app, eye);
    let (expected, _) = quadtree.select(eye, |_| true);
    assert_eq!(unculled.len(), expected.len(), "dry depth 0 must not cull");
}
