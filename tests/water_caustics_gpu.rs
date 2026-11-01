//! Caustic map of the water surface. Run explicitly with a graphics adapter.
//!
//! A sun straight overhead shines through a surface `h = A cos(k x)`. Each crest is a cylindrical
//! lens. For small slopes, a ray at `x` refracts by `(1 - 1/n) h'(x)` and lands at depth `d` at
//! `x + d (1 - 1/n) h'(x)`, so the light spreads by `1 - d / f` with
//! `f = 1 / ((1 - 1/n) A k^2 cos(k x))`. Halfway to the focus, crests show nearly twice the light
//! and troughs two thirds, and the mean stays one. The test compares texels with the exact refraction
//! seen through the one-texel tent the map spreads each ray over.
use bevy::{
    app::PluginsState,
    prelude::*,
    render::gpu_readback::{Readback, ReadbackComplete},
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

#[derive(Resource, Default)]
struct Capture {
    map: Vec<f32>,
    serial: u64,
}

fn step(app: &mut App) {
    app.update();
    assert!(app.should_exit().is_none(), "renderer must remain healthy");
    thread::sleep(Duration::from_millis(2));
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

fn focal_depth() -> f32 {
    let k = TAU / WAVELENGTH;
    1.0 / ((1.0 - 1.0 / IOR) * AMPLITUDE * k * k)
}

#[test]
#[ignore = "requires a graphics adapter"]
fn wave_crests_focus_sunlight_at_the_lens_depth() {
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
        Camera3d::default(),
        Transform::from_xyz(0.0, 10.0, 0.0).looking_at(Vec3::ZERO, Vec3::Z),
    ));
    app.world_mut().spawn((
        DirectionalLight::default(),
        Transform::default().looking_to(Vec3::NEG_Y, Vec3::Z),
    ));
    step(&mut app);

    // The single slice is centred on the camera at the origin, so texel `i` sits at
    // x = (i + 0.5 - RESOLUTION / 2) * TEXEL.
    let k = TAU / WAVELENGTH;
    let x_of = |i: u32| (i as f32 + 0.5 - RESOLUTION as f32 * 0.5) * TEXEL;
    let displacement = app.world().resource::<WaterLodData>().displacement.clone();
    {
        let mut images = app.world_mut().resource_mut::<Assets<Image>>();
        let mut image = images.get_mut(&displacement).unwrap();
        let data = image.data.as_mut().unwrap();
        for (n, texel) in data.chunks_exact_mut(8).enumerate() {
            let x = x_of(n as u32 % RESOLUTION);
            let lanes = [
                -AMPLITUDE * k * (k * x).sin(),
                0.0,
                AMPLITUDE * (k * x).cos(),
                0.0,
            ];
            for (lane, value) in lanes.iter().enumerate() {
                texel[lane * 2..lane * 2 + 2].copy_from_slice(&half_bits(*value).to_le_bytes());
            }
        }
    }
    let map = app.world().resource::<WaterCaustics>().map.clone();
    app.world_mut().spawn(Readback::texture(map)).observe(
        |event: On<ReadbackComplete>, mut capture: ResMut<Capture>| {
            capture.map = event
                .data
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect();
            capture.serial += 1;
        },
    );
    let deadline = Instant::now() + Duration::from_secs(30);
    while app.world().resource::<Capture>().serial < 4 {
        step(&mut app);
        assert!(Instant::now() < deadline, "caustics readback timed out");
    }
    let map = app.world().resource::<Capture>().map.clone();
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

    // The map spreads each ray over the receiver texels around its hit with a tent one texel wide,
    // so the expected value is the light landing under that tent, from the exact refraction.
    let depth = 0.5 * focal_depth();
    let landing = |x: f32| {
        let s = AMPLITUDE * k * (k * x).sin();
        let n = Vec2::new(s, 1.0).normalize();
        let i = Vec2::NEG_Y;
        let eta = 1.0 / IOR;
        let cos_i = -i.dot(n);
        let cos_t = (1.0 - eta * eta * (1.0 - cos_i * cos_i)).sqrt();
        let t = eta * i + (eta * cos_i - cos_t) * n;
        let drop = AMPLITUDE * (k * x).cos() + depth;
        x + t.x * drop / -t.y
    };
    let expected = |i: u32| {
        let c = x_of(i);
        let samples = 20000;
        let span = 4.0 * TEXEL;
        (0..samples)
            .map(|j| {
                let x = c - span + 2.0 * span * (j as f32 + 0.5) / samples as f32;
                (1.0 - ((landing(x) - c) / TEXEL).abs()).max(0.0)
            })
            .sum::<f32>()
            * (2.0 * span / samples as f32)
            / TEXEL
    };
    let crest = interior
        .iter()
        .copied()
        .find(|&i| (k * x_of(i)).cos() > 0.98)
        .unwrap();
    let trough = interior
        .iter()
        .copied()
        .find(|&i| (k * x_of(i)).cos() < -0.98)
        .unwrap();
    for (label, i) in [("crest", crest), ("trough", trough)] {
        let measured = column(i);
        eprintln!("{label}: {measured} against {}", expected(i));
        assert!(
            (measured / expected(i) - 1.0).abs() < 0.02,
            "{label}: {measured} against {}",
            expected(i)
        );
    }
}
