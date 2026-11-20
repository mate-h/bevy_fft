//! Moving shocks and eddies on flat beds, against exact solutions: Stoker's bore over a wet bed and
//! Ritter's front over a dry one, the shocks a river makes when a jam gives way or a flood runs
//! down a dry bed, and a vortex left to spin down, as behind a boulder. Run explicitly with a
//! graphics adapter.
//!
//! The dam breaks run in a 120 m channel of 0.25 m cells, four cells wide, with walls on every
//! edge and a dam at 30 m. The water starts at rest, 1 m deep upstream. Each test steps 6 s, reads
//! the depth back and finds where the bore or the front has reached.
//!
//! MacCormack advection (CMF10) is non-conservative and moves a bore at the wrong speed
//! (docs/river_solver_spike.md). On 2026-11-20 it ran Stoker's bore 24% short of its exact
//! distance, with the plateau behind it 0.331 m deep against 0.396 m, and the dry front with
//! Manning n 0.03 to 35% of Ritter's frictionless distance. The MacCormack tests record that
//! error and fail if it grows. Momentum-conserving advection
//! ([`ShallowWaterSettings::momentum_conserving`]) put the bore within a cell, the plateau within
//! 1% and the dry front to 47%, and keeps a vortex's energy about as well.
use bevy::{
    app::PluginsState,
    prelude::*,
    render::{
        Render, RenderApp, RenderSystems,
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        gpu_readback::{Readback, ReadbackComplete},
        render_resource::PollType,
        render_resource::*,
        renderer::{RenderContext, RenderDevice, RenderGraph, RenderGraphSystems, RenderQueue},
    },
    window::ExitCondition,
    winit::WinitPlugin,
};
use bevy_fft::shallow_water::*;
use std::{
    sync::{
        Arc,
        atomic::{AtomicU32, Ordering},
    },
    thread,
    time::{Duration, Instant},
};

const G: f32 = 9.81;
const DT: f32 = 0.005;
const STEPS_PER_FRAME: u32 = 40;
const FRAMES: u32 = 30;

/// The dam-break channel.
const NX: u32 = 480;
const NY: u32 = 4;
const DX: f32 = 0.25;
/// The dam, in meters from the left wall.
const DAM: f32 = 30.0;
const H_LEFT: f32 = 1.0;

/// A grid to run: its size, cell width, starting state and how to step it.
#[derive(Clone)]
struct Setup {
    cells: UVec2,
    dx: f32,
    /// Bed and depth per cell, row by row.
    bed_water: Vec<[f32; 2]>,
    /// Velocity per face across x, `cells.x + 1` per row, and across z, `cells.x` per row and
    /// `cells.y + 1` rows. Empty for still water.
    face_u: Vec<f32>,
    face_w: Vec<f32>,
    manning_n: f32,
    /// Ferguson's grain size, zero for Manning's friction.
    grain_d84: f32,
    momentum_conserving: bool,
}

/// What a run reads back: bed and depth per cell, and the cell-centred velocity.
struct Result {
    bed_water: Vec<Vec2>,
    velocity: Vec<Vec2>,
    /// Wall time of each frame that stepped the solver, waiting for the GPU, in ms.
    frame_ms: Vec<f32>,
}

#[derive(Resource, Clone, ExtractResource)]
#[extract_app(RenderApp)]
struct Grid {
    images: ShallowWaterImages,
    cells: UVec2,
    dx: f32,
    manning_n: f32,
    grain_d84: f32,
    momentum_conserving: bool,
    frames_done: Arc<AtomicU32>,
}

#[derive(Resource, Default)]
struct Capture {
    bed_water: Vec<u8>,
    velocity: Vec<u8>,
    serial: u64,
}

#[derive(Resource, Default)]
struct Driver {
    solver: Option<ShallowWaterSolver>,
    frame: u32,
}

fn prepare_driver(
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    images: Res<bevy::render::render_asset::RenderAssets<bevy::render::texture::GpuImage>>,
    pipelines: Res<ShallowWaterPipelines>,
    cache: Res<PipelineCache>,
    grid: Option<Res<Grid>>,
    mut driver: ResMut<Driver>,
) {
    let Some(grid) = grid else {
        return;
    };
    if driver.solver.is_some() {
        return;
    }
    let Some(textures) = ShallowWaterTextures::from_images(&grid.images, &images) else {
        return;
    };
    let Some(solver) = ShallowWaterSolver::new(
        &device,
        &pipelines,
        &cache,
        ShallowWaterSolverDescriptor {
            cells: grid.cells,
            textures,
            inputs: ShallowWaterInputs::default(),
            particle_count: 0,
        },
    ) else {
        return;
    };
    solver.write_settings(
        &queue,
        &ShallowWaterSettings {
            dt: DT,
            cell_size: grid.dx,
            gravity: G,
            manning_n: grid.manning_n,
            grain_d84: grid.grain_d84,
            overshoot_limiter: false,
            momentum_conserving: grid.momentum_conserving,
            ..default()
        },
    );
    driver.solver = Some(solver);
}

fn run_driver(mut ctx: RenderContext, mut driver: ResMut<Driver>, grid: Option<Res<Grid>>) {
    let Some(grid) = grid else {
        return;
    };
    if driver.frame >= FRAMES {
        return;
    }
    let Some(solver) = driver.solver.as_ref() else {
        return;
    };
    {
        let mut pass = ctx
            .command_encoder()
            .begin_compute_pass(&ComputePassDescriptor {
                label: Some("shallow_water_moving_shock_test"),
                timestamp_writes: None,
            });
        for _ in 0..STEPS_PER_FRAME {
            solver.encode_step(&mut pass);
        }
        solver.encode_reconstruct_velocity(&mut pass);
    }
    driver.frame += 1;
    grid.frames_done.store(driver.frame, Ordering::Relaxed);
}

fn step(app: &mut App) {
    app.update();
    assert!(app.should_exit().is_none(), "renderer must remain healthy");
    thread::sleep(Duration::from_millis(2));
}

/// Floats of a readback, `lanes` per texel, without the rows' padding.
fn texels(bytes: &[u8], width: u32, height: u32, lane: usize) -> Vec<f32> {
    let row = bytes.len() / height as usize;
    (0..height as usize)
        .flat_map(|y| (0..width as usize).map(move |x| (x, y)))
        .map(|(x, y)| {
            let o = y * row + x * 16 + lane * 4;
            f32::from_le_bytes(bytes[o..o + 4].try_into().unwrap())
        })
        .collect()
}

/// Runs `setup` for `FRAMES × STEPS_PER_FRAME` steps.
fn run(setup: Setup) -> Result {
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
    .add_plugins(ShallowWaterSolverPlugin)
    .add_plugins(ExtractResourcePlugin::<Grid>::default())
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
    let render_app = app.sub_app_mut(RenderApp);
    render_app
        .init_resource::<Driver>()
        .add_systems(
            Render,
            prepare_driver.in_set(RenderSystems::PrepareBindGroups),
        )
        .add_systems(RenderGraph, run_driver.in_set(RenderGraphSystems::Render));

    let (nx, ny) = (setup.cells.x, setup.cells.y);
    let frames_done = Arc::new(AtomicU32::new(0));
    let images = {
        let mut assets = app.world_mut().resource_mut::<Assets<Image>>();
        let images = ShallowWaterImages::new(&mut assets, nx, ny);
        // The starting state goes up with the images.
        let mut set = |handle: &Handle<Image>, texels: Vec<[f32; 4]>| {
            assets.get_mut(handle).unwrap().data = Some(bytemuck::cast_slice(&texels).to_vec());
        };
        set(
            &images.bed_water,
            setup
                .bed_water
                .iter()
                .map(|&[bed, h]| [bed, h, 0.0, 0.0])
                .collect(),
        );
        // The face textures are R32Float, one float per texel.
        let mut set_r = |handle: &Handle<Image>, values: &[f32]| {
            if !values.is_empty() {
                assets.get_mut(handle).unwrap().data = Some(bytemuck::cast_slice(values).to_vec());
            }
        };
        set_r(&images.flow_x, &setup.face_u);
        set_r(&images.flow_y, &setup.face_w);
        images
    };
    app.insert_resource(Grid {
        images,
        cells: setup.cells,
        dx: setup.dx,
        manning_n: setup.manning_n,
        grain_d84: setup.grain_d84,
        momentum_conserving: setup.momentum_conserving,
        frames_done: frames_done.clone(),
    });

    let deadline = Instant::now() + Duration::from_secs(60);
    let mut frame_ms = Vec::new();
    while frames_done.load(Ordering::Relaxed) < FRAMES {
        let before = frames_done.load(Ordering::Relaxed);
        let start = Instant::now();
        app.update();
        assert!(app.should_exit().is_none(), "renderer must remain healthy");
        let _ = app.world().resource::<RenderDevice>().poll(PollType::Wait {
            submission_index: None,
            timeout: Some(Duration::from_secs(5)),
        });
        if frames_done.load(Ordering::Relaxed) > before {
            frame_ms.push(start.elapsed().as_secs_f32() * 1000.0);
        }
        assert!(
            Instant::now() < deadline,
            "solver never finished its frames"
        );
    }
    let grid = app.world().resource::<Grid>().images.clone();
    app.world_mut()
        .spawn(Readback::texture(grid.bed_water.clone()))
        .observe(
            |event: On<ReadbackComplete>, mut capture: ResMut<Capture>| {
                capture.bed_water = event.data.clone();
                capture.serial += 1;
            },
        );
    app.world_mut()
        .spawn(Readback::texture(grid.velocity.clone()))
        .observe(
            |event: On<ReadbackComplete>, mut capture: ResMut<Capture>| {
                capture.velocity = event.data.clone();
                capture.serial += 1;
            },
        );
    while app.world().resource::<Capture>().serial < 2 {
        step(&mut app);
        assert!(Instant::now() < deadline, "readback timed out");
    }
    let capture = app.world().resource::<Capture>();
    let bed = texels(&capture.bed_water, nx, ny, 0);
    let depth = texels(&capture.bed_water, nx, ny, 1);
    let vx = texels(&capture.velocity, nx, ny, 0);
    let vz = texels(&capture.velocity, nx, ny, 1);
    Result {
        frame_ms,
        bed_water: bed
            .iter()
            .zip(&depth)
            .map(|(&b, &h)| Vec2::new(b, h))
            .collect(),
        velocity: vx.iter().zip(&vz).map(|(&x, &z)| Vec2::new(x, z)).collect(),
    }
}

fn elapsed() -> f32 {
    DT * (FRAMES * STEPS_PER_FRAME) as f32
}

/// The dam-break channel with `h_right` downstream.
fn dam_break(h_right: f32, manning_n: f32, momentum_conserving: bool) -> Setup {
    Setup {
        cells: UVec2::new(NX, NY),
        dx: DX,
        bed_water: (0..NX * NY)
            .map(|i| {
                let x = ((i % NX) as f32 + 0.5) * DX;
                [0.0, if x < DAM { H_LEFT } else { h_right }]
            })
            .collect(),
        face_u: Vec::new(),
        face_w: Vec::new(),
        manning_n,
        grain_d84: 0.0,
        momentum_conserving,
    }
}

/// Depth along the channel's centre row, and the volume of the whole grid in m³.
fn centre_and_volume(result: &Result) -> (Vec<f32>, f32) {
    let row = (NY / 2 * NX) as usize;
    let centre = result.bed_water[row..row + NX as usize]
        .iter()
        .map(|c| c.y)
        .collect();
    let volume = result.bed_water.iter().map(|c| c.y).sum::<f32>() * DX * DX;
    (centre, volume)
}

fn initial_volume(h_right: f32) -> f32 {
    let length = NX as f32 * DX;
    (DAM * H_LEFT + (length - DAM) * h_right) * NY as f32 * DX
}

/// Stoker's solution for a dam break onto still water `h_right` deep: the depth between the
/// rarefaction and the bore, and the bore's speed.
fn stoker(h_left: f32, h_right: f32) -> (f32, f32) {
    let (g, hl, hr) = (G as f64, h_left as f64, h_right as f64);
    // The rarefaction's speed behind the bore must match the bore's speed of the water it moves.
    let mismatch = |hm: f64| {
        let rarefaction = 2.0 * ((g * hl).sqrt() - (g * hm).sqrt());
        let bore = (hm - hr) * (g * (hm + hr) / (2.0 * hm * hr)).sqrt();
        rarefaction - bore
    };
    let (mut lo, mut hi) = (hr, hl);
    for _ in 0..100 {
        let mid = 0.5 * (lo + hi);
        if mismatch(mid) > 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let hm = 0.5 * (lo + hi);
    let um = 2.0 * ((g * hl).sqrt() - (g * hm).sqrt());
    (hm as f32, (hm * um / (hm - hr)) as f32)
}

/// The furthest cell centre downstream of the dam where the depth exceeds `level`.
fn reach(depth: &[f32], level: f32) -> f32 {
    depth
        .iter()
        .rposition(|&h| h > level)
        .map(|i| (i as f32 + 0.5) * DX)
        .unwrap_or(0.0)
}

/// Stoker: 1 m upstream, 0.1 m downstream, no friction. Returns the bore's position, its exact
/// position (about 48.6 m after 6 s), the plateau depth behind it and the exact plateau depth.
fn stoker_run(momentum_conserving: bool) -> (f32, f32, f32, f32) {
    let h_right = 0.1;
    let result = run(dam_break(h_right, 0.0, momentum_conserving));
    let (depth, volume) = centre_and_volume(&result);
    assert!(
        depth.iter().all(|h| h.is_finite() && *h >= 0.0),
        "depth went negative or non-finite"
    );
    let start = initial_volume(h_right);
    assert!(
        (volume - start).abs() < 1e-4 * start,
        "volume {volume} m³ against {start} m³ at the start"
    );
    let (hm, speed) = stoker(H_LEFT, h_right);
    let exact = DAM + speed * elapsed();
    let found = reach(&depth, 0.5 * (hm + h_right));
    // The plateau between the rarefaction and the bore, away from both ends.
    let plateau =
        &depth[((DAM + 0.6 * (exact - DAM)) / DX) as usize..((exact - 2.0) / DX) as usize];
    let plateau_depth = plateau.iter().sum::<f32>() / plateau.len().max(1) as f32;
    println!(
        "Stoker bore at {found:.2} m against {exact:.2} m ({:+.1}% of its run), plateau \
         {plateau_depth:.3} m against {hm:.3} m",
        100.0 * ((found - DAM) / (exact - DAM) - 1.0)
    );
    (found, exact, plateau_depth, hm)
}

/// Ritter with Manning n 0.03: the front's share of Ritter's frictionless run 2√(g h) t.
fn ritter_run(momentum_conserving: bool) -> f32 {
    let result = run(dam_break(0.0, 0.03, momentum_conserving));
    let (depth, volume) = centre_and_volume(&result);
    assert!(
        depth.iter().all(|h| h.is_finite() && *h >= 0.0),
        "depth went negative or non-finite"
    );
    let start = initial_volume(0.0);
    assert!(
        (volume - start).abs() < 1e-4 * start,
        "volume {volume} m³ against {start} m³ at the start"
    );
    let ritter = DAM + 2.0 * (G * H_LEFT).sqrt() * elapsed();
    let found = reach(&depth, 0.005);
    let share = (found - DAM) / (ritter - DAM);
    println!(
        "dry front at {found:.2} m, {:.0}% of Ritter's frictionless run to {ritter:.2} m",
        100.0 * share
    );
    share
}

#[test]
#[ignore = "requires a graphics adapter"]
fn maccormack_runs_a_dam_break_bore_a_quarter_slow() {
    let (found, exact, _, _) = stoker_run(false);
    let error = (found - DAM) / (exact - DAM) - 1.0;
    assert!(
        error.abs() < 0.3,
        "the bore ran {:+.1}% of its exact distance",
        100.0 * error
    );
}

#[test]
#[ignore = "requires a graphics adapter"]
fn momentum_conserving_advection_runs_a_dam_break_bore_at_stokers_speed() {
    let (found, exact, plateau, hm) = stoker_run(true);
    assert!(
        (found - exact).abs() <= 1.5 * DX,
        "the bore reached {found:.2} m against {exact:.2} m"
    );
    assert!(
        (plateau - hm).abs() < 0.02 * hm,
        "the plateau is {plateau:.3} m deep against {hm:.3} m"
    );
}

/// Friction and the scheme hold the front back from Ritter's frictionless run.
#[test]
#[ignore = "requires a graphics adapter"]
fn dam_breaks_onto_a_dry_bed_advance_their_fronts() {
    let maccormack = ritter_run(false);
    assert!(
        (0.3..1.0).contains(&maccormack),
        "MacCormack's front ran {:.0}% of Ritter's distance",
        100.0 * maccormack
    );
    let conserving = ritter_run(true);
    assert!(
        conserving > maccormack && conserving < 1.0,
        "the momentum-conserving front ran {:.0}% of Ritter's distance, MacCormack's {:.0}%",
        100.0 * conserving,
        100.0 * maccormack
    );
}

/// The vortex grid: 64 m square of 0.5 m cells, 1 m deep.
const VORTEX_CELLS: u32 = 128;
const VORTEX_DX: f32 = 0.5;

/// A Gaussian vortex, 2 m in radius and up to about 0.6 m/s, in still water 1 m deep, set on the
/// faces. Returns the setup and the kinetic energy per unit density, in m⁵/s².
fn vortex(momentum_conserving: bool) -> (Setup, f32) {
    let n = VORTEX_CELLS;
    let centre = 0.5 * n as f32 * VORTEX_DX;
    let (radius, spin) = (2.0f32, 1.0f32);
    // Velocity of the vortex at a point: v = spin · r × e^(−r²/R²), turning counterclockwise.
    let velocity = |x: f32, z: f32| {
        let (dx, dz) = (x - centre, z - centre);
        let decay = (-(dx * dx + dz * dz) / (radius * radius)).exp();
        Vec2::new(-dz, dx) * spin * decay
    };
    let face_u = (0..n * (n + 1))
        .map(|i| {
            velocity(
                (i % (n + 1)) as f32 * VORTEX_DX,
                ((i / (n + 1)) as f32 + 0.5) * VORTEX_DX,
            )
            .x
        })
        .collect();
    let face_w = (0..n * (n + 1))
        .map(|i| {
            velocity(
                ((i % n) as f32 + 0.5) * VORTEX_DX,
                (i / n) as f32 * VORTEX_DX,
            )
            .y
        })
        .collect();
    let energy = (0..n * n)
        .map(|i| {
            let v = velocity(
                ((i % n) as f32 + 0.5) * VORTEX_DX,
                ((i / n) as f32 + 0.5) * VORTEX_DX,
            );
            0.5 * v.length_squared()
        })
        .sum::<f32>()
        * VORTEX_DX
        * VORTEX_DX;
    let setup = Setup {
        cells: UVec2::splat(n),
        dx: VORTEX_DX,
        bed_water: vec![[0.0, 1.0]; (n * n) as usize],
        face_u,
        face_w,
        manning_n: 0.0,
        grain_d84: 0.0,
        momentum_conserving,
    };
    (setup, energy)
}

/// Kinetic energy left in the vortex, as a share of its start, after 6 s.
fn vortex_energy_kept(momentum_conserving: bool) -> f32 {
    let (setup, start) = vortex(momentum_conserving);
    let result = run(setup);
    let energy = result
        .bed_water
        .iter()
        .zip(&result.velocity)
        .map(|(c, v)| 0.5 * c.y * v.length_squared())
        .sum::<f32>()
        * VORTEX_DX
        * VORTEX_DX;
    energy / start
}

/// Eddies behind boulders and along eddy lines need advection that doesn't damp them much
/// faster than MacCormack's.
#[test]
#[ignore = "requires a graphics adapter"]
fn momentum_conserving_advection_keeps_a_vortex_spinning() {
    let maccormack = vortex_energy_kept(false);
    let conserving = vortex_energy_kept(true);
    println!(
        "vortex energy kept after {:.0} s: MacCormack {:.1}%, momentum-conserving {:.1}%",
        elapsed(),
        100.0 * maccormack,
        100.0 * conserving
    );
    assert!(
        1.0 - conserving <= 1.5 * (1.0 - maccormack) + 0.02,
        "momentum-conserving advection lost {:.1}% of the vortex's energy, MacCormack {:.1}%",
        100.0 * (1.0 - conserving),
        100.0 * (1.0 - maccormack)
    );
}

/// Wall time per solver step on a 1024² grid of wet water, momentum-conserving advection against
/// MacCormack, alternated three times each. The GPU may be shared, so only the ratio of the two
/// medians is meaningful, and only loosely.
#[test]
#[ignore = "requires a graphics adapter; a benchmark"]
fn momentum_conserving_advection_costs_about_as_much_as_maccormack() {
    let n = 1024u32;
    let setup = |momentum_conserving| Setup {
        cells: UVec2::splat(n),
        dx: 0.5,
        bed_water: (0..n * n)
            .map(|i| {
                let (x, z) = ((i % n) as f32 - 512.0, (i / n) as f32 - 512.0);
                [0.0, 1.0 + 0.3 * (-(x * x + z * z) / 400.0).exp()]
            })
            .collect(),
        face_u: Vec::new(),
        face_w: Vec::new(),
        manning_n: 0.03,
        grain_d84: 0.0,
        momentum_conserving,
    };
    let median = |mut v: Vec<f32>| {
        // Leave out the first frames, which wait on pipelines and uploads.
        v.drain(..v.len().min(5));
        v.sort_by(f32::total_cmp);
        v[v.len() / 2]
    };
    let (mut mac, mut sd) = (Vec::new(), Vec::new());
    for _ in 0..3 {
        mac.push(median(run(setup(false)).frame_ms));
        sd.push(median(run(setup(true)).frame_ms));
    }
    let per_step = |v: &[f32]| {
        let mut v = v.to_vec();
        v.sort_by(f32::total_cmp);
        v[1] / STEPS_PER_FRAME as f32
    };
    let (mac, sd) = (per_step(&mac), per_step(&sd));
    println!(
        "1024² step: MacCormack {mac:.3} ms, momentum-conserving {sd:.3} ms ({:.2}x)",
        sd / mac
    );
    assert!(
        sd < 1.25 * mac,
        "momentum-conserving step {sd:.3} ms against {mac:.3} ms"
    );
}

/// Ferguson's (2007) `U/u*` for relative depth `r = h / D84`, his D84 fit.
fn ferguson_ratio(r: f32) -> f32 {
    let (a1, a2) = (6.5f32, 2.5f32);
    a1 * a2 * r / (a1 * a1 + a2 * a2 * r.powf(5.0 / 3.0)).sqrt()
}

/// Speed left after 6 s in a 120 m channel of still-level water `h` deep that starts moving at
/// 1 m/s everywhere, over grains `d84`. The walls send waves in from the ends, which reach no
/// further than 10 m in that time, so the middle only feels bed friction.
fn coasting_speed(h: f32, d84: f32) -> f32 {
    let u0 = 1.0;
    let mut face_u = vec![u0; ((NX + 1) * NY) as usize];
    for row in face_u.chunks_mut((NX + 1) as usize) {
        row[0] = 0.0;
        row[NX as usize] = 0.0;
    }
    let setup = Setup {
        cells: UVec2::new(NX, NY),
        dx: DX,
        bed_water: vec![[0.0, h]; (NX * NY) as usize],
        face_u,
        face_w: Vec::new(),
        manning_n: 0.0,
        grain_d84: d84,
        momentum_conserving: true,
    };
    let result = run(setup);
    let middle = (NY / 2 * NX + NX / 2) as usize;
    result.velocity[middle].x
}

/// Water coasting over a cobble bed slows as `du/dt = -Cf u² / h`, so `u = u0 / (1 + Cf u0 t / h)`
/// with Ferguson's `Cf = (u*/U)²`. Checked where the water is as deep as the grains, where
/// Manning's n would be far off, and ten times deeper.
#[test]
#[ignore = "requires a graphics adapter"]
fn ferguson_friction_slows_coasting_water_as_his_resistance_law_says() {
    let h = 0.2;
    for d84 in [0.2f32, 0.02] {
        let cf = ferguson_ratio(h / d84).powi(-2);
        let expected = 1.0 / (1.0 + cf * elapsed() / h);
        let got = coasting_speed(h, d84);
        println!(
            "h/D84 {:.0}: Cf {cf:.4}, speed after {:.0} s {got:.3} m/s, expected {expected:.3}",
            h / d84,
            elapsed()
        );
        assert!(
            (got - expected).abs() < 0.03 * expected + 0.005,
            "h/D84 {}: water coasted at {got:.3} m/s, Ferguson says {expected:.3}",
            h / d84
        );
    }
}
