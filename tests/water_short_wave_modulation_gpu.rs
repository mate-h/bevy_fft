//! Short waves modulated by a dispersive patch's waves, in the forward and the deferred water
//! paths. Run explicitly with a graphics adapter.
//!
//! A black water plane is seen straight down through an orthographic camera 32 m across, under a
//! sun 50 degrees off the vertical, so every pixel sees the sun's specular lobe from 25 degrees off
//! its peak, out in its tail, where rougher water shows more of it. (Inside the lobe, rougher
//! water would show less.) The LOD data below the patch holds unresolved waves
//! of one mean square slope everywhere. A paused patch in add mode holds a long, low wave along X,
//! 32 m long with its crest under the middle of the view and its troughs at the sides. Its own
//! slopes stay under 0.01, too small to move the lobe. The short-wave modulation must roughen the
//! water over the crest and smooth it in the troughs, in both paths, and a flat patch must leave
//! the water as it is without one.
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
use bevy_fft::{dispersive::*, fft::FftPlugin, water::*};
use std::{
    thread,
    time::{Duration, Instant},
};

const SIZE: u32 = 256;
/// Meters the view spans.
const VIEW: f32 = 32.0;
/// Material roughness, low enough that the slope variance dominates.
const ROUGHNESS: f32 = 0.1;
/// Mean square slope of the unresolved waves below the patch.
const VARIANCE: f32 = 0.05;
/// Sun angle off the vertical, in degrees.
const SUN_TILT: f32 = 50.0;
/// Patch cells along each edge, one meter each.
const N: u32 = 128;
const REST_DEPTH: f32 = 4.0;
/// The patch's wave: 32 m long and 5 cm high, a slope amplitude `k a` near 0.01.
const WAVELENGTH: f32 = 32.0;
const AMPLITUDE: f32 = 0.05;
/// Modulation for the test, larger than the default so that the patch's gentle wave, which keeps
/// its own slopes out of the shading, still modulates the short waves by a factor near 1.5.
const MODULATION: f32 = 40.0;

#[derive(Resource, Default)]
struct Capture {
    pixels: Vec<u8>,
    serial: u64,
}

/// One frame. The LOD data below the patch is uploaded again first, since no base writer
/// rewrites it, and the patch adds its deviation onto whatever the slices hold.
fn step(app: &mut App) {
    reset_lod(app);
    app.update();
    assert!(app.should_exit().is_none(), "renderer must remain healthy");
    thread::sleep(Duration::from_millis(2));
}

/// Column means of the green channel in linear units, once `frames` readbacks have arrived.
fn capture(app: &mut App, frames: u64) -> Vec<f32> {
    let first = app.world().resource::<Capture>().serial;
    let deadline = Instant::now() + Duration::from_secs(30);
    while app.world().resource::<Capture>().serial < first + frames {
        step(app);
        assert!(Instant::now() < deadline, "image readback timed out");
    }
    let pixels = &app.world().resource::<Capture>().pixels;
    let mut columns = vec![0.0; SIZE as usize];
    for (i, p) in pixels.chunks_exact(4).enumerate() {
        columns[i % SIZE as usize] += linear(p[1]) / SIZE as f32;
    }
    columns
}

fn linear(value: u8) -> f32 {
    let s = value as f32 / 255.0;
    if s <= 0.04045 {
        s / 12.92
    } else {
        ((s + 0.055) / 1.055).powf(2.4)
    }
}

/// Mean over the columns whose world X is within 2 m of `x`.
fn band(columns: &[f32], x: f32) -> f32 {
    let picked: Vec<f32> = columns
        .iter()
        .enumerate()
        .filter(|(i, _)| {
            let cx = (*i as f32 + 0.5) / SIZE as f32 * VIEW - 0.5 * VIEW;
            (cx.abs() - x.abs()).abs() < 2.0
        })
        .map(|(_, v)| *v)
        .collect();
    picked.iter().sum::<f32>() / picked.len() as f32
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

/// Flat water ten meters deep whose unresolved waves have mean square slope [`VARIANCE`], at rest.
fn reset_lod(app: &mut App) {
    // The LOD data is created on the first frame.
    let Some(lod) = app.world().get_resource::<WaterLodData>().cloned() else {
        return;
    };
    let mut images = app.world_mut().resource_mut::<Assets<Image>>();
    {
        let mut foam = images.get_mut(&lod.foam).unwrap();
        for texel in foam.data.as_mut().unwrap().chunks_exact_mut(8) {
            texel.fill(0);
            texel[2..4].copy_from_slice(&f16_bits(10.0).to_le_bytes());
            texel[6..8].copy_from_slice(&f16_bits(VARIANCE).to_le_bytes());
        }
    }
    for handle in [&lod.displacement, &lod.flow, &lod.drift] {
        images
            .get_mut(handle)
            .unwrap()
            .data
            .as_mut()
            .unwrap()
            .fill(0);
    }
}

/// Sets the paused patch's surface to `amplitude` times the test wave, at rest.
fn set_patch(app: &mut App, amplitude: f32) {
    let state = app.world().resource::<DispersiveController>().state.clone();
    let mut images = app.world_mut().resource_mut::<Assets<Image>>();
    let mut image = images.get_mut(&state).unwrap();
    let data = image.data.as_mut().unwrap();
    for y in 0..N {
        for x in 0..N {
            let wx = x as f32 + 0.5 - 0.5 * N as f32;
            let eta = amplitude * (std::f32::consts::TAU * wx / WAVELENGTH).cos();
            let o = ((y * N + x) * 16) as usize;
            data[o..o + 4].copy_from_slice(&(REST_DEPTH + eta).to_le_bytes());
            data[o + 4..o + 16].fill(0);
        }
    }
}

fn set_modulation(app: &mut App, modulation: Option<f32>) {
    let mut controller = app.world_mut().resource_mut::<DispersiveController>();
    controller.placement = modulation.map(|m| DispersivePlacement {
        blend_cells: 4.0,
        short_wave_modulation: m,
        ..default()
    });
}

fn modulation_app(deferred: bool) -> App {
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
    .add_plugins((FftPlugin, WaterPlugin, DispersivePlugin))
    .insert_resource(WaterLodSettings {
        resolution: 256,
        slice_count: 1,
        base_texel_size: 0.25,
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
    let world = app.world_mut();
    let mut controller = world.resource_scope(|world, mut images: Mut<Assets<Image>>| {
        DispersiveController::spawn(&mut world.commands(), &mut images, N)
    });
    controller.tile_world = N as f32;
    controller.paused = true;
    controller.placement = Some(DispersivePlacement::default());
    world.insert_resource(controller);

    let mut image = Image::new_target_texture(
        SIZE,
        SIZE,
        TextureFormat::Rgba8UnormSrgb,
        Some(TextureFormat::Rgba8UnormSrgb),
    );
    image.texture_descriptor.usage |= TextureUsages::COPY_SRC;
    let target = world.resource_mut::<Assets<Image>>().add(image);
    let camera = world
        .spawn((
            Camera3d::default(),
            Msaa::Off,
            Tonemapping::None,
            Exposure { ev100: 0.0 },
            Projection::Orthographic(OrthographicProjection {
                scaling_mode: ScalingMode::Fixed {
                    width: VIEW,
                    height: VIEW,
                },
                ..OrthographicProjection::default_3d()
            }),
            Transform::from_xyz(0.0, 20.0, 0.0).looking_at(Vec3::ZERO, Vec3::NEG_Z),
            RenderTarget::Image(ImageRenderTarget {
                handle: target.clone(),
                scale_factor: 1.0,
            }),
        ))
        .id();
    if deferred {
        world
            .entity_mut(camera)
            .insert((DepthPrepass, DeferredPrepass));
    }
    let tilt = SUN_TILT.to_radians();
    world.spawn((
        DirectionalLight {
            illuminance: 10.0,
            ..default()
        },
        Transform::default().looking_to(Vec3::new(tilt.sin(), -tilt.cos(), 0.0), Vec3::Z),
    ));
    world.spawn(Readback::texture(target)).observe(
        |event: On<ReadbackComplete>, mut capture: ResMut<Capture>| {
            capture.pixels = event.data.clone();
            capture.serial += 1;
        },
    );
    let mesh = world
        .resource_mut::<Assets<Mesh>>()
        .add(Plane3d::default().mesh().size(48.0, 48.0));
    let material = world
        .resource_mut::<Assets<WaterSurfaceMaterial>>()
        .add(WaterSurfaceMaterial {
            base: StandardMaterial {
                base_color: Color::BLACK,
                perceptual_roughness: ROUGHNESS,
                ..default()
            },
            extension: WaterSurfaceExtension::default(),
        });
    world.spawn((Mesh3d(mesh), MeshMaterial3d(material), WaterSurfaceTag));
    world.flush();
    // Lets the patch's `init_basin` run before its state is overwritten.
    for _ in 0..30 {
        step(&mut app);
    }
    app
}

struct Shade {
    /// No patch at all.
    none: Vec<f32>,
    /// A flat patch with the test modulation.
    flat: Vec<f32>,
    /// The wave without modulation, and with it.
    still: Vec<f32>,
    modulated: Vec<f32>,
}

fn shade(deferred: bool) -> Shade {
    let mut app = modulation_app(deferred);
    set_modulation(&mut app, None);
    let none = capture(&mut app, 6);
    set_patch(&mut app, 0.0);
    set_modulation(&mut app, Some(MODULATION));
    let flat = capture(&mut app, 6);
    set_patch(&mut app, AMPLITUDE);
    set_modulation(&mut app, Some(0.0));
    let still = capture(&mut app, 6);
    set_modulation(&mut app, Some(MODULATION));
    let modulated = capture(&mut app, 6);
    Shade {
        none,
        flat,
        still,
        modulated,
    }
}

#[test]
#[ignore = "requires a graphics adapter"]
fn the_patch_roughens_short_waves_on_crests_and_smooths_them_in_troughs() {
    let forward = shade(false);
    let deferred = shade(true);
    for (name, s) in [("forward", &forward), ("deferred", &deferred)] {
        let none = (band(&s.none, 0.0), band(&s.none, 16.0));
        let flat = (band(&s.flat, 0.0), band(&s.flat, 16.0));
        let still = (band(&s.still, 0.0), band(&s.still, 16.0));
        let crest = band(&s.modulated, 0.0);
        let trough = band(&s.modulated, 16.0);
        eprintln!(
            "{name}: none {:.4}/{:.4} flat {:.4}/{:.4} unmodulated wave {:.4}/{:.4} \
             modulated crest {crest:.4} trough {trough:.4}",
            none.0, none.1, flat.0, flat.1, still.0, still.1
        );
        assert!(none.0 > 0.005, "{name}: the sun's lobe must show");
        for (a, b) in [(flat.0, none.0), (flat.1, none.1)] {
            assert!(
                (a - b).abs() < 0.01 * b,
                "{name}: a flat patch must leave the water as it is, {a} against {b}"
            );
        }
        assert!(
            (still.0 - still.1).abs() < 0.03 * still.1 && (still.0 - none.0).abs() < 0.03 * none.0,
            "{name}: the wave's own slopes must not move the lobe, {} and {}",
            still.0,
            still.1
        );
        // The variance there is about 1.5 times and 0.7 times the open water's, which moves
        // the lobe's tail by about a tenth either way.
        assert!(
            crest > 1.06 * still.0,
            "{name}: the crest must roughen the water, {crest} against {}",
            still.0
        );
        assert!(
            trough < 0.94 * still.1,
            "{name}: the trough must smooth the water, {trough} against {}",
            still.1
        );
    }
    let fc = band(&forward.modulated, 0.0);
    let dc = band(&deferred.modulated, 0.0);
    assert!(
        (fc - dc).abs() < 0.1 * fc,
        "deferred crest {dc} differs from forward {fc}"
    );
}
