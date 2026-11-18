//! The GPU FFT against a CPU DFT, and its forward and inverse round trip. Sizes up to 512 run each
//! axis in one `fft_dit_line` dispatch, larger ones one radix-2 dispatch per stage. Run explicitly
//! with a graphics adapter.
use bevy::{
    app::PluginsState,
    asset::RenderAssetUsages,
    prelude::*,
    render::{
        gpu_readback::{Readback, ReadbackComplete},
        render_resource::{Extent3d, TextureDimension, TextureFormat},
    },
    window::ExitCondition,
    winit::WinitPlugin,
};
use bevy_fft::fft::{FftInputTexture, FftPlugin, FftSchedule, FftSource, resources::FftTextures};
use std::{
    f64::consts::TAU,
    sync::Mutex,
    thread,
    time::{Duration, Instant},
};

/// GPU tests in this file take turns.
static GPU: Mutex<()> = Mutex::new(());

/// Lanes compared: the forward pass sets the fourth to 1.
const LANES: usize = 3;

#[derive(Resource, Default)]
struct Capture {
    re: Vec<u8>,
    im: Vec<u8>,
    serial: u64,
}

fn step(app: &mut App) {
    app.update();
    assert!(app.should_exit().is_none(), "renderer must remain healthy");
    thread::sleep(Duration::from_millis(1));
}

/// Deterministic values in [-1, 1].
fn noise(seed: u32) -> f32 {
    let mut x = seed.wrapping_mul(0x9E37_79B9) ^ 0x85EB_CA6B;
    x ^= x >> 16;
    x = x.wrapping_mul(0x7FEB_352D);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846C_A68B);
    x ^= x >> 16;
    x as f32 / u32::MAX as f32 * 2.0 - 1.0
}

/// An `n`×`n` RGBA32F image of noise in the first three lanes, offset by `seed`.
fn input(n: u32, seed: u32) -> (Image, Vec<[f32; 4]>) {
    let texels: Vec<[f32; 4]> = (0..n * n)
        .map(|i| {
            let s = (i * 4 + seed) * 7;
            [noise(s), noise(s + 1), noise(s + 2), 0.0]
        })
        .collect();
    let bytes = texels
        .iter()
        .flat_map(|t| t.iter().flat_map(|v| v.to_le_bytes()))
        .collect();
    let image = Image::new(
        Extent3d {
            width: n,
            height: n,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        bytes,
        TextureFormat::Rgba32Float,
        RenderAssetUsages::all(),
    );
    (image, texels)
}

/// Runs `schedule` on `re + i im` and returns the buffer it ends in, real and imaginary texels.
fn run(n: u32, schedule: FftSchedule, re: Image, im: Image) -> (Vec<[f32; 4]>, Vec<[f32; 4]>) {
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
    .add_plugins(FftPlugin)
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
    let world = app.world_mut();
    let (re, im) = {
        let mut images = world.resource_mut::<Assets<Image>>();
        (images.add(re), images.add(im))
    };
    let mut source = FftSource::square_forward_then_inverse(n);
    source.schedule = schedule;
    let entity = world
        .spawn((
            source,
            FftInputTexture {
                real: re,
                imag: Some(im),
            },
        ))
        .id();
    world.flush();
    let deadline = Instant::now() + Duration::from_secs(60);
    let textures = loop {
        step(&mut app);
        if let Some(t) = app.world().get::<FftTextures>(entity) {
            break t.clone();
        }
        assert!(Instant::now() < deadline, "the FFT textures never appeared");
    };
    let (out_re, out_im) = match schedule {
        FftSchedule::Forward => (textures.buffer_c_re, textures.buffer_c_im),
        _ => (textures.buffer_b_re, textures.buffer_b_im),
    };
    let world = app.world_mut();
    world.spawn(Readback::texture(out_re)).observe(
        |event: On<ReadbackComplete>, mut capture: ResMut<Capture>| {
            capture.re = event.data.clone();
            capture.serial += 1;
        },
    );
    world.spawn(Readback::texture(out_im)).observe(
        |event: On<ReadbackComplete>, mut capture: ResMut<Capture>| {
            capture.im = event.data.clone();
        },
    );
    world.flush();
    // Let the pipelines compile and a few transforms land.
    for _ in 0..30 {
        step(&mut app);
    }
    let serial = app.world().resource::<Capture>().serial;
    while app.world().resource::<Capture>().serial <= serial + 3 {
        step(&mut app);
        assert!(Instant::now() < deadline, "readback timed out");
    }
    let capture = app.world().resource::<Capture>();
    let texels = |bytes: &[u8]| -> Vec<[f32; 4]> {
        assert_eq!(
            bytes.len(),
            (n * n * 16) as usize,
            "unexpected readback size"
        );
        bytes
            .chunks_exact(16)
            .map(|t| {
                std::array::from_fn(|i| f32::from_le_bytes(t[i * 4..i * 4 + 4].try_into().unwrap()))
            })
            .collect()
    };
    (texels(&capture.re), texels(&capture.im))
}

/// 2D DFT of lane `lane`, `exp(-i 2π k x / n)` along each axis, in f64.
fn dft(n: usize, re: &[[f32; 4]], im: &[[f32; 4]], lane: usize) -> Vec<(f64, f64)> {
    let twiddle: Vec<(f64, f64)> = (0..n)
        .map(|k| {
            let t = -TAU * k as f64 / n as f64;
            (t.cos(), t.sin())
        })
        .collect();
    let mut a: Vec<(f64, f64)> = (0..n * n)
        .map(|i| (re[i][lane] as f64, im[i][lane] as f64))
        .collect();
    let mut line = vec![(0.0, 0.0); n];
    for axis in 0..2 {
        for l in 0..n {
            let at = |i: usize| if axis == 0 { l * n + i } else { i * n + l };
            for (k, out) in line.iter_mut().enumerate() {
                let mut sum = (0.0, 0.0);
                for x in 0..n {
                    let (vr, vi) = a[at(x)];
                    let (wr, wi) = twiddle[(k * x) % n];
                    sum.0 += vr * wr - vi * wi;
                    sum.1 += vr * wi + vi * wr;
                }
                *out = sum;
            }
            for (i, v) in line.iter().enumerate() {
                a[at(i)] = *v;
            }
        }
    }
    a
}

/// Largest error against `expected` as a share of its largest magnitude, over the compared lanes.
fn relative_error(
    n: usize,
    re: &[[f32; 4]],
    im: &[[f32; 4]],
    expected: impl Fn(usize, usize) -> (f64, f64),
) -> f64 {
    let (mut err, mut scale) = (0.0f64, 0.0f64);
    for lane in 0..LANES {
        for i in 0..n * n {
            let (er, ei) = expected(lane, i);
            let dr = re[i][lane] as f64 - er;
            let di = im[i][lane] as f64 - ei;
            err = err.max((dr * dr + di * di).sqrt());
            scale = scale.max((er * er + ei * ei).sqrt());
        }
    }
    err / scale.max(1e-12)
}

fn forward_matches_dft(n: u32) {
    let (re_img, re) = input(n, 0);
    let (im_img, im) = input(n, 1);
    let (out_re, out_im) = run(n, FftSchedule::Forward, re_img, im_img);
    let n = n as usize;
    let expected: Vec<Vec<(f64, f64)>> = (0..LANES).map(|lane| dft(n, &re, &im, lane)).collect();
    let error = relative_error(n, &out_re, &out_im, |lane, i| expected[lane][i]);
    assert!(
        error < 1e-4,
        "{n}x{n} forward FFT off the DFT by {error:e} of its largest term"
    );
}

fn round_trip_returns_the_input(n: u32) {
    let (re_img, re) = input(n, 2);
    let (im_img, im) = input(n, 3);
    let (out_re, out_im) = run(n, FftSchedule::ForwardThenInverse, re_img, im_img);
    let n = n as usize;
    let error = relative_error(n, &out_re, &out_im, |lane, i| {
        (re[i][lane] as f64, im[i][lane] as f64)
    });
    assert!(
        error < 1e-5,
        "{n}x{n} round trip off the input by {error:e}"
    );
}

#[test]
#[ignore = "requires a graphics adapter"]
fn forward_fft_matches_a_dft_in_workgroup_memory() {
    let _gpu = GPU.lock().unwrap_or_else(|e| e.into_inner());
    forward_matches_dft(64);
    forward_matches_dft(256);
}

#[test]
#[ignore = "requires a graphics adapter"]
fn forward_fft_matches_a_dft_stage_by_stage() {
    let _gpu = GPU.lock().unwrap_or_else(|e| e.into_inner());
    forward_matches_dft(1024);
}

#[test]
#[ignore = "requires a graphics adapter"]
fn forward_then_inverse_returns_the_input() {
    let _gpu = GPU.lock().unwrap_or_else(|e| e.into_inner());
    round_trip_returns_the_input(256);
    round_trip_returns_the_input(1024);
}
