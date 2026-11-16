//! Drives the shallow water solver core directly, without the plugin's presets or per-frame loop.
//! Run explicitly with a graphics adapter.
//!
//! Each frame the test writes a new bed into its own `R32Float` texture, copies it into the grid
//! with `syncBed`, and runs several steps with a source texture. A Gaussian bump travels across
//! the bed while sources add water. With walls on every edge and the overshoot limiter off, the
//! solver conserves volume, so the water on the grid must equal what the sources added. A second
//! run drains a strip with sink weight 1 and checks that it ends dry.
use bevy::{
    app::PluginsState,
    prelude::*,
    render::{
        Render, RenderApp, RenderSystems,
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        gpu_readback::{Readback, ReadbackComplete},
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

const N: u32 = 64;
const FRAMES: u32 = 60;
const STEPS_PER_FRAME: u32 = 4;
const DT: f32 = 0.25;
/// Depth added per second inside the source block.
const RATE: f32 = 0.05;
/// Source block, in cells.
const SOURCE: std::ops::Range<u32> = 8..16;
/// Columns with sink weight 1 in the drained run.
const SINK: std::ops::Range<u32> = 56..64;

#[derive(Clone, Copy)]
struct Scenario {
    sink: bool,
    evaporation: f32,
}

#[derive(Resource, Clone, ExtractResource)]
#[extract_app(RenderApp)]
struct Grid {
    images: ShallowWaterImages,
    scenario_sink: bool,
    evaporation: f32,
    frames_done: Arc<AtomicU32>,
}

#[derive(Resource, Default)]
struct Capture {
    bytes: Vec<u8>,
    serial: u64,
}

/// Render-world driver: the solver, the bed and source textures it owns, and the frame count.
#[derive(Resource, Default)]
struct Driver {
    solver: Option<ShallowWaterSolver>,
    bed: Option<Texture>,
    frame: u32,
}

/// Bed height at cell `(x, y)` on frame `frame`: a gentle slope and a bump moving along +x.
fn bed_height(x: u32, y: u32, frame: u32) -> f32 {
    let cx = 20.0 + 0.4 * frame as f32;
    let (px, py) = (x as f32 + 0.5, y as f32 + 0.5);
    let r2 = (px - cx).powi(2) + (py - 32.0).powi(2);
    0.002 * px + 0.3 * (-r2 / 30.0).exp()
}

fn source_texel(x: u32, y: u32, sink: bool) -> [f32; 4] {
    let add = if SOURCE.contains(&x) && SOURCE.contains(&y) {
        RATE
    } else {
        0.0
    };
    let drain = if sink && SINK.contains(&x) { 1.0 } else { 0.0 };
    [add, drain, 0.0, 0.0]
}

fn upload(queue: &RenderQueue, texture: &Texture, bytes: &[u8], texel_size: u32) {
    queue.write_texture(
        texture.as_image_copy(),
        bytes,
        TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(N * texel_size),
            rows_per_image: None,
        },
        Extent3d {
            width: N,
            height: N,
            depth_or_array_layers: 1,
        },
    );
}

fn input_texture(device: &RenderDevice, label: &'static str, format: TextureFormat) -> Texture {
    device.create_texture(&TextureDescriptor {
        label: Some(label),
        size: Extent3d {
            width: N,
            height: N,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format,
        usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
        view_formats: &[],
    })
}

#[allow(clippy::too_many_arguments)]
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
    if driver.frame >= FRAMES {
        return;
    }
    if driver.solver.is_none() {
        let Some(textures) = ShallowWaterTextures::from_images(&grid.images, &images) else {
            return;
        };
        let bed = input_texture(&device, "test_bed", TextureFormat::R32Float);
        let source = input_texture(&device, "test_source", TextureFormat::Rgba32Float);
        let texels: Vec<f32> = (0..N * N)
            .flat_map(|i| source_texel(i % N, i / N, grid.scenario_sink))
            .collect();
        upload(&queue, &source, bytemuck::cast_slice(&texels), 16);
        let Some(solver) = ShallowWaterSolver::new(
            &device,
            &pipelines,
            &cache,
            ShallowWaterSolverDescriptor {
                cells: UVec2::splat(N),
                textures,
                inputs: ShallowWaterInputs {
                    bed: Some(bed.create_view(&TextureViewDescriptor::default())),
                    source: Some(source.create_view(&TextureViewDescriptor::default())),
                },
                particle_count: 0,
            },
        ) else {
            return;
        };
        solver.write_settings(
            &queue,
            &ShallowWaterSettings {
                dt: DT,
                cell_size: 1.0,
                gravity: 1.0,
                manning_n: 0.03,
                evaporation: grid.evaporation,
                overshoot_limiter: false,
                ..default()
            },
        );
        driver.solver = Some(solver);
        driver.bed = Some(bed);
    }
    let frame = driver.frame;
    let heights: Vec<f32> = (0..N * N)
        .map(|i| bed_height(i % N, i / N, frame))
        .collect();
    upload(
        &queue,
        driver.bed.as_ref().unwrap(),
        bytemuck::cast_slice(&heights),
        4,
    );
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
                label: Some("shallow_water_solver_test"),
                timestamp_writes: None,
            });
        solver.encode_sync_bed(&mut pass);
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

fn solver_app(scenario: Scenario) -> (App, Arc<AtomicU32>) {
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

    let frames_done = Arc::new(AtomicU32::new(0));
    let images = {
        let mut assets = app.world_mut().resource_mut::<Assets<Image>>();
        ShallowWaterImages::new(&mut assets, N, N)
    };
    app.insert_resource(Grid {
        images,
        scenario_sink: scenario.sink,
        evaporation: scenario.evaporation,
        frames_done: frames_done.clone(),
    });
    (app, frames_done)
}

/// Bed and depth per cell after the driver's last frame.
fn run(scenario: Scenario) -> Vec<Vec2> {
    let (mut app, frames_done) = solver_app(scenario);
    let deadline = Instant::now() + Duration::from_secs(60);
    while frames_done.load(Ordering::Relaxed) < FRAMES {
        step(&mut app);
        assert!(
            Instant::now() < deadline,
            "solver never finished its frames"
        );
    }
    let bed_water = app.world().resource::<Grid>().images.bed_water.clone();
    app.world_mut().spawn(Readback::texture(bed_water)).observe(
        |event: On<ReadbackComplete>, mut capture: ResMut<Capture>| {
            capture.bytes = event.data.clone();
            capture.serial += 1;
        },
    );
    while app.world().resource::<Capture>().serial == 0 {
        step(&mut app);
        assert!(Instant::now() < deadline, "bed_water readback timed out");
    }
    let bytes = &app.world().resource::<Capture>().bytes;
    let row = bytes.len() / N as usize;
    let lane = |o: usize| f32::from_le_bytes(bytes[o..o + 4].try_into().unwrap());
    (0..(N * N) as usize)
        .map(|i| {
            let o = (i / N as usize) * row + (i % N as usize) * 16;
            Vec2::new(lane(o), lane(o + 4))
        })
        .collect()
}

#[test]
#[ignore = "requires a graphics adapter"]
fn core_solver_steps_a_moving_bed_with_sources_and_sinks() {
    let cells = run(Scenario {
        sink: false,
        evaporation: 0.0,
    });
    let last = FRAMES - 1;
    for (i, c) in cells.iter().enumerate() {
        let (x, y) = (i as u32 % N, i as u32 / N);
        assert!(c.is_finite() && c.y >= 0.0, "cell ({x}, {y}) holds {c}");
        assert_eq!(c.x, bed_height(x, y, last), "bed at ({x}, {y})");
    }
    let volume: f32 = cells.iter().map(|c| c.y).sum();
    let block = (SOURCE.len() * SOURCE.len()) as f32;
    let injected = RATE * DT * (FRAMES * STEPS_PER_FRAME) as f32 * block;
    assert!(
        (volume - injected).abs() < 1e-3 * injected,
        "volume {volume} against {injected} added"
    );
    let outside: f32 = cells
        .iter()
        .enumerate()
        .filter(|(i, _)| {
            let (x, y) = (*i as u32 % N, *i as u32 / N);
            !(SOURCE.contains(&x) && SOURCE.contains(&y))
        })
        .map(|(_, c)| c.y)
        .sum();
    assert!(
        outside > 0.2 * injected,
        "only {outside} of {injected} left the source block"
    );

    let cells = run(Scenario {
        sink: true,
        evaporation: 0.01,
    });
    let mut drained = 0.0;
    for (i, c) in cells.iter().enumerate() {
        let x = i as u32 % N;
        assert!(c.is_finite() && c.y >= 0.0, "cell {i} holds {c}");
        if SINK.contains(&x) {
            assert_eq!(c.y, 0.0, "sink cell {i} still holds water");
        }
        drained += c.y;
    }
    assert!(
        drained < volume,
        "sink and evaporation left {drained}, at least the closed grid's {volume}"
    );
}
