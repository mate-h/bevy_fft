//! Floating bodies on a dispersive patch: obstruction, reflection and wakes. Run explicitly with a
//! graphics adapter.
use bevy::{
    app::PluginsState,
    prelude::*,
    render::gpu_readback::{Readback, ReadbackComplete},
    time::TimeUpdateStrategy,
    window::ExitCondition,
    winit::WinitPlugin,
};
use bevy_fft::{bodies::*, dispersive::*, fft::FftPlugin, water::*};
use std::{
    sync::Arc,
    thread,
    time::{Duration, Instant},
};

/// Two dispersive patches stepping at once in one process disturb each other's results, so the
/// tests here take turns.
static GPU: std::sync::Mutex<()> = std::sync::Mutex::new(());

const N: u32 = 128;
/// `init_basin` fills the patch to this depth over a flat bed.
const REST_DEPTH: f32 = 4.0;
const DT: f32 = 1.0 / 60.0;

#[derive(Resource, Default)]
struct Captures {
    state: Vec<u8>,
    obstruction: Vec<u8>,
    serial: u64,
}

fn step(app: &mut App) {
    app.update();
    assert!(app.should_exit().is_none(), "renderer must remain healthy");
    thread::sleep(Duration::from_millis(1));
}

/// A 128 m patch of 4 m water at the origin in deep mode, with body maps.
fn patch_app() -> App {
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
    .add_plugins((FftPlugin, FloatingBodyPlugin, DispersivePlugin))
    .insert_resource(WaterLodClear)
    .insert_resource(WaterLodSettings {
        resolution: 256,
        slice_count: 1,
        base_texel_size: 1.0,
    })
    .insert_resource(TimeUpdateStrategy::ManualDuration(Duration::from_secs_f32(
        DT,
    )))
    .insert_resource(bevy::render::error_handler::RenderErrorHandler(
        |error, _, _| {
            panic!("GPU validation failed: {error:?}");
        },
    ))
    .init_resource::<Captures>();
    while app.plugins_state() == PluginsState::Adding {
        bevy::tasks::tick_global_task_pools_on_main_thread();
        thread::sleep(Duration::from_millis(1));
    }
    app.finish();
    app.cleanup();
    let world = app.world_mut();
    let mut controller = world.resource_scope(|world, mut images: Mut<Assets<Image>>| {
        let mut controller = DispersiveController::spawn(&mut world.commands(), &mut images, N);
        controller.tile_world = N as f32;
        controller.deep_water_depth = Some(100.0);
        controller.placement = Some(DispersivePlacement {
            datum: 0.0,
            blend_cells: 4.0,
            ..default()
        });
        controller.enable_bodies(&mut images);
        controller
    });
    let state = controller.state.clone();
    let obstruction = controller.body_maps.as_ref().unwrap().obstruction.clone();
    controller.edge_trim_cells = 12.0;
    world.insert_resource(controller);
    world.spawn(Readback::texture(state)).observe(
        |event: On<ReadbackComplete>, mut captures: ResMut<Captures>| {
            captures.state = event.data.clone();
            captures.serial += 1;
        },
    );
    world.spawn(Readback::texture(obstruction)).observe(
        |event: On<ReadbackComplete>, mut captures: ResMut<Captures>| {
            captures.obstruction = event.data.clone();
        },
    );
    world.flush();
    for _ in 0..30 {
        step(&mut app);
    }
    app
}

fn settle(app: &mut App) {
    let deadline = Instant::now() + Duration::from_secs(60);
    let serial = app.world().resource::<Captures>().serial;
    while app.world().resource::<Captures>().serial <= serial + 1 {
        step(app);
        assert!(Instant::now() < deadline, "readback timed out");
    }
}

/// Surface deviation per cell, row by row.
fn eta(app: &App) -> Vec<f32> {
    let bytes = &app.world().resource::<Captures>().state;
    let texel = bytes.len() / (N * N) as usize;
    (0..(N * N) as usize)
        .map(|i| {
            f32::from_le_bytes(bytes[i * texel..i * texel + 4].try_into().unwrap()) - REST_DEPTH
        })
        .collect()
}

fn obstruction(app: &App) -> Vec<f32> {
    let bytes = &app.world().resource::<Captures>().obstruction;
    let texel = bytes.len() / (N * N) as usize;
    (0..(N * N) as usize)
        .map(|i| f32::from_le_bytes(bytes[i * texel..i * texel + 4].try_into().unwrap()))
        .collect()
}

/// Cell of world `xz` on the patch centred on the origin with 1 m cells.
fn cell(xz: Vec2) -> usize {
    let c = (xz + N as f32 * 0.5).floor().as_uvec2();
    (c.y * N + c.x) as usize
}

/// A 4 m by 2 m by 8 m barge, half under water.
fn barge() -> FloatingBody {
    let hull = Arc::new(voxelize(&Mesh::from(Cuboid::new(4.0, 2.0, 8.0)), 0.25).unwrap());
    let mut body = FloatingBody::new(hull, 8000.0);
    body.estimate_velocity = false;
    body
}

fn energy(eta: &[f32], cells: impl Iterator<Item = (u32, u32)>) -> f32 {
    cells.map(|(x, y)| eta[(y * N + x) as usize].powi(2)).sum()
}

/// Presses the brush down at `x` meters on the X axis for a quarter second.
fn splash(app: &mut App, x: f32) {
    for _ in 0..15 {
        let mut controller = app.world_mut().resource_mut::<DispersiveController>();
        let at = controller.world_to_cells(Vec2::new(x, 0.0));
        controller.pointer = at;
        controller.pointer_prev = at;
        controller.brush_radius = 3.0;
        // 1.4 m of head, so the splash's waves stand well above the float rounding of the dent.
        controller.brush_strength = 2.8;
        controller.brush_active = true;
        step(app);
    }
    app.world_mut()
        .resource_mut::<DispersiveController>()
        .brush_active = false;
}

#[test]
#[ignore = "requires a graphics adapter"]
fn a_hull_at_the_waterline_obstructs_its_cells() {
    let _gpu = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let mut app = patch_app();
    app.world_mut().spawn((barge(), Transform::IDENTITY));
    for _ in 0..60 {
        step(&mut app);
    }
    settle(&mut app);
    let o = obstruction(&app);
    assert!(
        o[cell(Vec2::ZERO)] < 0.05,
        "under the hull {}",
        o[cell(Vec2::ZERO)]
    );
    assert!(o[cell(Vec2::new(1.5, 3.5))] < 0.05);
    assert!(
        o[cell(Vec2::new(10.0, 0.0))] > 0.95,
        "open water {}",
        o[cell(Vec2::new(10.0, 0.0))]
    );
    assert!(o[cell(Vec2::new(0.0, 10.0))] > 0.95);
}

#[test]
#[ignore = "requires a graphics adapter"]
fn a_still_hull_holds_its_dent_without_radiating() {
    let _gpu = GPU.lock().unwrap_or_else(|e| e.into_inner());
    // Band-limiting the bare surface each step smeared the dent onto the open water next to the
    // hull, the obstruction put it back under the hull, and the waves around it grew past 1 m in
    // 20 s.
    let mut app = patch_app();
    app.world_mut().spawn((wall(), Transform::IDENTITY));
    for _ in 0..1200 {
        step(&mut app);
    }
    settle(&mut app);
    let e = eta(&app);
    assert!(
        (e[cell(Vec2::ZERO)] + 1.0).abs() < 0.02,
        "dent {}",
        e[cell(Vec2::ZERO)]
    );
    let waves = (0..N * N)
        .filter(|&i| ((i % N) as i32 - 64).abs() > 3 || ((i / N) as i32 - 64).abs() > 12)
        .map(|i| e[i as usize].abs())
        .fold(0.0f32, f32::max);
    assert!(waves < 0.05, "waves around the hull reach {waves} m");
}

/// A 4 m wide, 24 m long barge across the X axis, half under water.
fn wall() -> FloatingBody {
    let hull = Arc::new(voxelize(&Mesh::from(Cuboid::new(4.0, 2.0, 24.0)), 0.25).unwrap());
    let mut body = FloatingBody::new(hull, 24000.0);
    body.estimate_velocity = false;
    body
}

/// The surface 15 s after a splash at x = 25 m, or at the same time without one. Waves about 10 m
/// long, at about 2 m/s, take that long to pass the barge.
fn after_splash(with_barge: bool, with_splash: bool) -> Vec<f32> {
    let mut app = patch_app();
    if with_barge {
        app.world_mut().spawn((wall(), Transform::IDENTITY));
    }
    // Let the barge settle into its dent.
    for _ in 0..300 {
        step(&mut app);
    }
    if with_splash {
        splash(&mut app, 25.0);
    } else {
        for _ in 0..15 {
            step(&mut app);
        }
    }
    for _ in 0..900 {
        step(&mut app);
    }
    // Hold the patch while the readback catches up, so both runs stop at the same step and the
    // barge's own settling cancels exactly.
    app.world_mut()
        .resource_mut::<DispersiveController>()
        .paused = true;
    settle(&mut app);
    eta(&app)
}

/// Energy of the splash's waves behind the barge (x from -50 to -4 m) and beyond the splash (x
/// from 24 to 50 m), where the waves the barge reflects have arrived by then, along the band
/// |z| < 12 m it spans. Subtracting the run without the splash leaves out the barge's own
/// settling.
fn shadow_and_front(with_barge: bool) -> (f32, f32) {
    let with = after_splash(with_barge, true);
    let without = after_splash(with_barge, false);
    let waves: Vec<f32> = with.iter().zip(&without).map(|(a, b)| a - b).collect();
    let band = |x0: f32, x1: f32| {
        let (x0, x1) = ((x0 + 64.0) as u32, (x1 + 64.0) as u32);
        (x0..x1).flat_map(move |x| (52..76).map(move |y| (x, y)))
    };
    (
        energy(&waves, band(-50.0, -4.0)),
        energy(&waves, band(24.0, 50.0)),
    )
}

#[test]
#[ignore = "requires a graphics adapter"]
fn a_hull_reflects_waves_and_shadows_the_water_behind_it() {
    let _gpu = GPU.lock().unwrap_or_else(|e| e.into_inner());
    // A 4 m wide barge, 1 m deep, against waves 10 to 20 m long leaves about half the energy
    // behind it and sends about a quarter more back past the splash. The Airy step is global and
    // long waves bend around the barge's ends, so some energy still reaches the water behind it,
    // as in Gilligan's eWave.
    let (open_behind, open_front) = shadow_and_front(false);
    let (barge_behind, barge_front) = shadow_and_front(true);
    assert!(
        barge_behind < 0.7 * open_behind,
        "behind the barge {barge_behind} against {open_behind} without it, front {barge_front} against {open_front}"
    );
    assert!(
        barge_front > 1.1 * open_front,
        "in front of the barge {barge_front} against {open_front} without it"
    );
}

#[test]
#[ignore = "requires a graphics adapter"]
fn a_towed_body_trails_a_wake_in_deep_mode() {
    let _gpu = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let mut app = patch_app();
    let mut body = barge();
    body.estimate_velocity = true;
    let entity = app
        .world_mut()
        .spawn((body, Transform::from_xyz(-40.0, 0.0, 0.0)))
        .id();
    for _ in 0..60 {
        step(&mut app);
    }
    // Tow it 40 m along +X at 4 m/s.
    for _ in 0..600 {
        let mut transform = app.world_mut().get_mut::<Transform>(entity).unwrap();
        transform.translation.x += 4.0 * DT;
        step(&mut app);
    }
    settle(&mut app);
    let e = eta(&app);
    let at = app
        .world()
        .get::<Transform>(entity)
        .unwrap()
        .translation
        .xz();
    // Waves well away from the barge, behind it along its path.
    let wake = (0..N * N)
        .filter(|&i| {
            let xz = Vec2::new((i % N) as f32, (i / N) as f32) - N as f32 * 0.5 + 0.5;
            xz.distance(at) > 10.0 && xz.x < at.x && xz.x > -40.0
        })
        .map(|i| e[i as usize].abs())
        .fold(0.0f32, f32::max);
    assert!(wake > 0.05, "largest wake wave {wake} m");
    assert!(wake < 2.0, "wake wave {wake} m");
}

/// Half-float bits of `v`, for the swell written into the LOD data. Values here are small and
/// normal, so the conversion skips subnormals and infinities.
fn f16_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    if v == 0.0 || exponent <= 0 {
        return sign;
    }
    let mantissa = ((bits >> 13) & 0x3ff) as u16;
    sign | ((exponent as u16) << 10) | mantissa
}

/// Swell amplitude in meters and wavelength in meters of the ambient water in the LOD data.
const SWELL: f32 = 0.3;
const SWELL_LENGTH: f32 = 8.0;

/// A still swell along z, one period under the barge's 8 m length, written into the LOD displacement every frame in place of an ocean.
#[derive(Resource)]
struct Swell(bevy::render::render_resource::Buffer);

fn swell_bytes() -> Vec<u8> {
    let res = 256u32;
    let k = std::f32::consts::TAU / SWELL_LENGTH;
    let mut bytes = Vec::with_capacity((res * res * 8) as usize);
    for j in 0..res {
        let z = j as f32 + 0.5 - 0.5 * res as f32;
        for _ in 0..res {
            let lanes = [0.0, SWELL * k * (k * z).cos(), SWELL * (k * z).sin(), 0.0];
            for lane in lanes {
                bytes.extend_from_slice(&f16_bits(lane).to_le_bytes());
            }
        }
    }
    bytes
}

fn write_swell(
    mut ctx: bevy::render::renderer::RenderContext,
    device: Res<bevy::render::renderer::RenderDevice>,
    lod: Option<Res<WaterLodData>>,
    images: Res<bevy::render::render_asset::RenderAssets<bevy::render::texture::GpuImage>>,
    mut swell: Local<Option<Swell>>,
) {
    use bevy::render::render_resource::*;
    let (Some(lod), Some(target)) = (
        lod.as_ref(),
        lod.as_ref().and_then(|l| images.get(&l.displacement)),
    ) else {
        return;
    };
    let _ = lod;
    let swell = swell.get_or_insert_with(|| {
        Swell(device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("test_swell"),
            contents: &swell_bytes(),
            usage: BufferUsages::COPY_SRC,
        }))
    });
    ctx.command_encoder().copy_buffer_to_texture(
        TexelCopyBufferInfo {
            buffer: &swell.0,
            layout: TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(256 * 8),
                rows_per_image: Some(256),
            },
        },
        target.texture.as_image_copy(),
        Extent3d {
            width: 256,
            height: 256,
            depth_or_array_layers: 1,
        },
    );
}

#[derive(Resource, Default)]
struct Composite {
    bytes: Vec<u8>,
    serial: u64,
}

/// The composite LOD surface height over every cell a barge held still in the swell obstructs,
/// after a few seconds, with the swell the LOD data had there before the patch.
fn composite_under_hull() -> (Vec<f32>, Vec<f32>) {
    let mut app = patch_app();
    app.sub_app_mut(bevy::render::RenderApp).add_systems(
        bevy::render::renderer::RenderGraph,
        write_swell.in_set(WaterLodGraphSystems::Base),
    );
    app.init_resource::<Composite>();
    app.world_mut().spawn((barge(), Transform::IDENTITY));
    for _ in 0..240 {
        step(&mut app);
    }
    let displacement = app.world().resource::<WaterLodData>().displacement.clone();
    app.world_mut()
        .spawn(Readback::texture(displacement))
        .observe(|event: On<ReadbackComplete>, mut c: ResMut<Composite>| {
            c.bytes = event.data.clone();
            c.serial += 1;
        });
    settle(&mut app);
    let deadline = Instant::now() + Duration::from_secs(30);
    while app.world().resource::<Composite>().serial < 2 {
        step(&mut app);
        assert!(Instant::now() < deadline, "LOD readback timed out");
    }
    let o = obstruction(&app);
    let bytes = &app.world().resource::<Composite>().bytes;
    let row = bytes.len() / 256;
    let k = std::f32::consts::TAU / SWELL_LENGTH;
    let (mut composite, mut ambient) = (Vec::new(), Vec::new());
    for y in 0..N {
        for x in 0..N {
            if o[(y * N + x) as usize] > 0.05 {
                continue;
            }
            // The patch's 1 m cells line up with the LOD's 1 m texels, 64 texels in.
            let (tx, ty) = (x as usize + 64, y as usize + 64);
            let o = ty * row + tx * 8 + 4;
            let h = half_to_f32(u16::from_le_bytes([bytes[o], bytes[o + 1]]));
            composite.push(h);
            ambient.push(SWELL * (k * (y as f32 + 0.5 - 0.5 * N as f32)).sin());
        }
    }
    (composite, ambient)
}

fn half_to_f32(bits: u16) -> f32 {
    let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exponent = ((bits >> 10) & 0x1f) as i32;
    let mantissa = (bits & 0x3ff) as f32;
    if exponent == 0 {
        return sign * mantissa * 2f32.powi(-24);
    }
    sign * (1.0 + mantissa / 1024.0) * 2f32.powi(exponent - 15)
}

fn spread(values: &[f32]) -> f32 {
    let mean = values.iter().sum::<f32>() / values.len() as f32;
    (values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / values.len() as f32).sqrt()
}

#[test]
#[ignore = "requires a graphics adapter"]
fn ambient_swell_rests_against_a_hull_instead_of_running_through_it() {
    let _gpu = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let (composite, ambient) = composite_under_hull();
    assert!(
        composite.len() > 10,
        "only {} cells under the hull",
        composite.len()
    );
    // The swell runs under the hull with a spread of about 0.2 m. The composite surface there
    // rests against the hull's flat bottom instead: the footprint measures the hull's head against
    // the composite surface, swell included, so the patch takes the swell away under the hull
    // without a separate ambient term (Gilligan §3.4, eq. 77).
    let swell_spread = spread(&ambient);
    let surface_spread = spread(&composite);
    assert!(
        swell_spread > 0.15,
        "the swell under the hull spreads only {swell_spread} m"
    );
    assert!(
        surface_spread < 0.25 * swell_spread,
        "the surface under the hull spreads {surface_spread} m, the swell {swell_spread} m"
    );
    // The barge is held with its centre on the rest surface, so its bottom is 1 m down.
    let mean = composite.iter().sum::<f32>() / composite.len() as f32;
    assert!(
        (mean + 1.0).abs() < 0.15,
        "the surface under the hull sits at {mean} m, the hull's bottom at -1 m"
    );
}
