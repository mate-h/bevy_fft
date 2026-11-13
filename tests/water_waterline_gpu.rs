//! Waterline mask of the water viewer. Run explicitly with a graphics adapter.
//!
//! A level camera sits a few millimetres from a flat water surface, so the near plane cuts the
//! surface at a row the projection fixes. The mask must change sign at that row and report the
//! eye's height above the water.
//!
//! A second test tilts the camera down so the cut crosses the middle of the image, and finds the
//! meniscus darkening the rows on both sides of the crossing.
use bevy::{
    app::PluginsState,
    camera::{Exposure, Hdr, ImageRenderTarget, RenderTarget},
    core_pipeline::tonemapping::{DebandDither, Tonemapping},
    light::EnvironmentMapLight,
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
    f32::consts::FRAC_PI_4,
    thread,
    time::{Duration, Instant},
};

const SIZE: u32 = 256;
const NEAR: f32 = 0.1;
const FOV: f32 = FRAC_PI_4;
/// 0.25 as a half float.
const QUARTER_F16: u16 = 0x3400;

#[derive(Resource, Default)]
struct Capture {
    mask: Vec<f32>,
    serial: u64,
    /// Red lane of the viewer's image.
    color: Vec<f32>,
}

#[derive(Component)]
struct Viewer;

fn step(app: &mut App) {
    app.update();
    assert!(app.should_exit().is_none(), "renderer must remain healthy");
    thread::sleep(Duration::from_millis(2));
}

/// Mask from a readback that started after the scene last changed.
fn capture(app: &mut App) -> Vec<f32> {
    let first = app.world().resource::<Capture>().serial;
    let deadline = Instant::now() + Duration::from_secs(30);
    while app.world().resource::<Capture>().serial < first + 4 {
        step(app);
        assert!(Instant::now() < deadline, "mask readback timed out");
    }
    app.world().resource::<Capture>().mask.clone()
}

fn waterline_app() -> App {
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
    let exposure = Exposure { ev100: 0.0 };
    let sky = {
        let mut images = app.world_mut().resource_mut::<Assets<Image>>();
        EnvironmentMapLight::solid_color(&mut images, Color::WHITE)
            .with_intensity(0.5 / exposure.exposure())
    };
    app.world_mut().spawn((
        Viewer,
        Camera3d::default(),
        WaterUnderwater,
        Hdr,
        Msaa::Off,
        Tonemapping::None,
        DebandDither::Disabled,
        exposure,
        sky,
        Projection::Perspective(PerspectiveProjection {
            fov: FOV,
            near: NEAR,
            ..default()
        }),
        Transform::default(),
        RenderTarget::Image(ImageRenderTarget {
            handle: target.clone(),
            scale_factor: 1.0,
        }),
    ));
    app.world_mut().spawn(Readback::texture(target)).observe(
        |event: On<ReadbackComplete>, mut capture: ResMut<Capture>| {
            capture.color = event
                .data
                .chunks_exact(8)
                .map(|p| half(u16::from_le_bytes([p[0], p[1]])))
                .collect();
        },
    );
    let mesh = app
        .world_mut()
        .resource_mut::<Assets<Mesh>>()
        .add(Plane3d::default().mesh().size(8.0, 8.0));
    let material = app
        .world_mut()
        .resource_mut::<Assets<WaterSurfaceMaterial>>()
        .add(WaterSurfaceMaterial::default());
    app.world_mut().spawn((
        Mesh3d(mesh),
        MeshMaterial3d(material),
        WaterSurfaceTag,
        Transform::default(),
    ));
    // The mask takes the target's size on the first update, so read back the sized one.
    step(&mut app);
    let mask = app.world().resource::<WaterWaterline>().mask.clone();
    app.world_mut().spawn(Readback::texture(mask)).observe(
        |event: On<ReadbackComplete>, mut capture: ResMut<Capture>| {
            capture.mask = event
                .data
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect();
            capture.serial += 1;
        },
    );
    app
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

fn place(app: &mut App, eye: Vec3, plane: f32) {
    let mut viewer = app
        .world_mut()
        .query_filtered::<&mut Transform, With<Viewer>>();
    *viewer.single_mut(app.world_mut()).unwrap() = Transform::from_translation(eye);
    let mut surface = app
        .world_mut()
        .query_filtered::<&mut Transform, With<WaterSurfaceTag>>();
    surface.single_mut(app.world_mut()).unwrap().translation.y = plane;
}

fn set_height_quarter(app: &mut App, on: bool) {
    let handle = app.world().resource::<WaterLodData>().displacement.clone();
    let mut images = app.world_mut().resource_mut::<Assets<Image>>();
    let mut image = images.get_mut(&handle).unwrap();
    let bits = if on { QUARTER_F16 } else { 0 };
    for texel in image.data.as_mut().unwrap().chunks_exact_mut(8) {
        texel[4..6].copy_from_slice(&bits.to_le_bytes());
    }
}

/// Mask row, as a pixel coordinate from the top, where a level camera's near plane meets water
/// `depth` meters above the eye.
fn expected_crossing(depth: f32) -> f32 {
    let half_height = NEAR * (FOV * 0.5).tan();
    let ndc = depth / half_height;
    (1.0 - ndc) * 0.5 * SIZE as f32
}

/// First row from the top whose centre column is under water.
fn first_wet_row(mask: &[f32]) -> u32 {
    let column = SIZE / 2;
    (0..SIZE)
        .find(|row| mask[(row * SIZE + column) as usize] < 0.0)
        .unwrap_or(SIZE)
}

fn check_crossing(app: &mut App, eye: Vec3, plane: f32, surface: f32) {
    place(app, eye, plane);
    let mask = capture(app);
    assert_eq!(mask.len(), (SIZE * SIZE) as usize);
    let expected = expected_crossing(surface - eye.y);
    let row = first_wet_row(&mask);
    eprintln!("eye {eye}, surface {surface}: wet from row {row}, crossing at {expected}");
    assert!(
        (row as f32 + 0.5 - expected).abs() <= 1.0,
        "waterline at row {row}, expected {expected}"
    );
    // Every row of a level camera is one height, so the sign changes nowhere else.
    for r in 0..SIZE {
        let line = &mask[(r * SIZE) as usize..((r + 1) * SIZE) as usize];
        let wet = r >= row;
        assert!(
            line.iter().all(|h| (*h < 0.0) == wet),
            "row {r} is not uniformly {}",
            if wet { "under water" } else { "above water" }
        );
    }
    let waterline = app.world().resource::<WaterWaterline>();
    let eye_height = waterline.eye_height.expect("eye height readback");
    assert!(
        (eye_height - (eye.y - surface)).abs() < 1e-4,
        "eye height {eye_height}, expected {}",
        eye.y - surface
    );
    assert_eq!(waterline.camera_below_surface(), eye.y < surface);
}

#[test]
#[ignore = "requires a graphics adapter"]
fn waterline_mask_follows_the_near_plane_cut() {
    let mut app = waterline_app();

    // Eye 1 cm above the water and 1 cm below it.
    check_crossing(&mut app, Vec3::new(0.3, 0.01, -0.2), 0.0, 0.0);
    check_crossing(&mut app, Vec3::new(0.3, -0.01, -0.2), 0.0, 0.0);
    // A raised mesh plane moves the water with it.
    check_crossing(&mut app, Vec3::new(1.0, 2.005, 0.5), 2.0, 2.0);

    // The displaced height counts as well.
    set_height_quarter(&mut app, true);
    check_crossing(&mut app, Vec3::new(0.0, 0.26, 0.0), 0.0, 0.25);

    // Deep under water, the whole near plane is wet.
    place(&mut app, Vec3::new(0.0, -3.0, 0.0), 0.0);
    let mask = capture(&mut app);
    assert!(mask.iter().all(|h| (*h + 3.25).abs() < 0.05));
    assert!(
        app.world()
            .resource::<WaterWaterline>()
            .camera_below_surface()
    );
    set_height_quarter(&mut app, false);
}

#[test]
#[ignore = "requires a graphics adapter"]
fn meniscus_darkens_the_rows_at_the_near_plane_cut() {
    let mut app = waterline_app();
    // Tilted 30 degrees down, the near plane centre sits 5 cm below the eye, so the cut crosses
    // the middle of the image and every ray on either side runs down through a smooth scene.
    {
        let mut viewer = app
            .world_mut()
            .query_filtered::<&mut Transform, With<Viewer>>();
        *viewer.single_mut(app.world_mut()).unwrap() = Transform::from_xyz(0.3, 0.05, -0.2)
            .with_rotation(Quat::from_rotation_x(-30f32.to_radians()));
    }
    let mask = capture(&mut app);
    let color = app.world().resource::<Capture>().color.clone();
    assert_eq!(color.len(), (SIZE * SIZE) as usize);
    let wet = first_wet_row(&mask);
    assert!((SIZE / 4..3 * SIZE / 4).contains(&wet), "cut at row {wet}");
    let row = |r: u32| {
        (SIZE / 2 - 32..SIZE / 2 + 32)
            .map(|c| color[(r * SIZE + c) as usize])
            .sum::<f32>()
            / 64.0
    };
    // The unshaded brightness of a row, extrapolated by a line through rows four to seven past
    // the band on the same side.
    let baseline = |r: u32, rows: [u32; 4]| {
        let n = rows.len() as f32;
        let mx = rows.iter().map(|&x| x as f32).sum::<f32>() / n;
        let my = rows.iter().map(|&x| row(x)).sum::<f32>() / n;
        let slope = rows
            .iter()
            .map(|&x| (x as f32 - mx) * (row(x) - my))
            .sum::<f32>()
            / rows.iter().map(|&x| (x as f32 - mx).powi(2)).sum::<f32>();
        my + slope * (r as f32 - mx)
    };
    let dry = [wet - 8, wet - 7, wet - 6, wet - 5];
    let under = [wet + 4, wet + 5, wet + 6, wet + 7];
    let mut failures = Vec::new();
    for (r, rows, expected) in [
        (wet - 4, dry, 1.0),
        (wet - 3, dry, 0.9),
        (wet - 2, dry, 0.81),
        (wet - 1, dry, 0.729),
        (wet, under, 0.729),
        (wet + 1, under, 0.81),
        (wet + 2, under, 0.9),
        (wet + 3, under, 1.0),
    ] {
        let base = baseline(r, rows);
        let ratio = row(r) / base;
        eprintln!(
            "row {r}: {} over {base}, x{ratio:.3} against x{expected}",
            row(r)
        );
        if base <= 0.0 || (ratio - expected).abs() > 0.03 {
            failures.push((r, ratio, expected));
        }
    }
    assert!(failures.is_empty(), "meniscus rows off: {failures:?}");
}
