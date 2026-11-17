//! Spray from a waterfall in a shallow water river. Run explicitly with a graphics adapter.
use bevy::{
    app::PluginsState,
    prelude::*,
    render::gpu_readback::{Readback, ReadbackComplete},
    time::TimeUpdateStrategy,
    window::ExitCondition,
    winit::WinitPlugin,
};
use bevy_fft::{particles::*, shallow_water::*, water::*};
use std::{
    thread,
    time::{Duration, Instant},
};

const NX: u32 = 48;
const NY: u32 = 8;
const SLOPE: f32 = 0.002;
/// Discharge per meter of channel width, m²/s.
const Q: f32 = 0.25;
const DT: f32 = 0.08;
/// Particles per cubic meter over the lip.
const RATE: f32 = 40.0;

#[derive(Resource, Default)]
struct Counters {
    spawned: u32,
    serial: u64,
}

fn step(app: &mut App) {
    app.update();
    assert!(app.should_exit().is_none(), "renderer must remain healthy");
    thread::sleep(Duration::from_millis(1));
}

/// A placed channel of 1 m cells falling along +x, walled along its sides, fed over its left edge
/// and draining over its right edge, with a bed that drops by `drop` meters halfway along.
fn channel_app(drop: f32) -> App {
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
    .insert_resource(WaterLodClear)
    .insert_resource(WaterLodSettings {
        resolution: 64,
        slice_count: 1,
        base_texel_size: 1.0,
    })
    .add_plugins((ShallowWaterPlugin, WaterParticlesPlugin))
    .insert_resource(TimeUpdateStrategy::ManualDuration(Duration::from_secs_f32(
        1.0 / 60.0,
    )))
    .insert_resource(bevy::render::error_handler::RenderErrorHandler(
        |error, _, _| {
            panic!("GPU validation failed: {error:?}");
        },
    ))
    .init_resource::<Counters>();
    let mut images = app.world_mut().resource_mut::<Assets<Image>>();
    let mut controller = ShallowWaterController::new(&mut images, NX, NY);
    let bed: Vec<f32> = (0..NX * NY)
        .map(|i| {
            let x = (i % NX) as f32;
            SLOPE * (NX as f32 - x - 0.5) + if x < (NX / 2) as f32 { drop } else { 0.0 }
        })
        .collect();
    controller.bed_source = images.add(bed_source_image(NX, NY, &bed));
    controller.preset_index = SHALLOW_WATER_PRESET_BED_SOURCE;
    controller.fill_level = 0.0;
    controller.gravity = 9.81;
    controller.dt = DT;
    controller.manning_n = 0.03;
    controller.placement = Some(ShallowWaterPlacement {
        center: Vec2::ZERO,
        cell_size: 1.0,
        datum: 0.0,
        blend_cells: 0.0,
    });
    controller.waterfalls = Some(ShallowWaterWaterfalls {
        rate: RATE,
        ..default()
    });
    let mut profile = ShallowWaterBorderProfile::new(UVec2::new(NX, NY));
    for j in 0..NY {
        profile.set(
            GridEdge::Left,
            j,
            ShallowWaterBorderFace::Inflow {
                discharge_per_width: Q,
            },
        );
        profile.set(
            GridEdge::Right,
            j,
            ShallowWaterBorderFace::Outflow { slope: SLOPE },
        );
    }
    controller.border_profile = Some(images.add(profile.image()));
    app.insert_resource(controller);
    while app.plugins_state() == PluginsState::Adding {
        bevy::tasks::tick_global_task_pools_on_main_thread();
        thread::sleep(Duration::from_millis(1));
    }
    app.finish();
    app.cleanup();
    step(&mut app);
    let state = app.world().resource::<WaterParticles>().state.clone();
    app.world_mut().spawn(Readback::buffer(state)).observe(
        |event: On<ReadbackComplete>, mut counters: ResMut<Counters>| {
            counters.spawned = bytemuck::cast_slice::<u8, u32>(&event.data)[1];
            counters.serial += 1;
        },
    );
    app
}

/// Steps `frames` frames, then until a fresh readback lands, and returns the particles spawned in
/// all.
fn spawned_after(app: &mut App, frames: u32, deadline: Instant) -> u32 {
    for _ in 0..frames {
        step(app);
        assert!(Instant::now() < deadline, "river run timed out");
    }
    app.world_mut()
        .resource_mut::<ShallowWaterController>()
        .paused = true;
    let serial = app.world().resource::<Counters>().serial;
    while app.world().resource::<Counters>().serial <= serial + 2 {
        step(app);
        assert!(Instant::now() < deadline, "readback timed out");
    }
    app.world_mut()
        .resource_mut::<ShallowWaterController>()
        .paused = false;
    app.world().resource::<Counters>().spawned
}

#[test]
#[ignore = "requires a graphics adapter"]
fn a_waterfall_sprays_in_proportion_to_its_discharge() {
    let deadline = Instant::now() + Duration::from_secs(240);
    // A 3 m drop, three cell widths: past the 1.5 cap.
    let mut app = channel_app(3.0);
    let before = spawned_after(&mut app, 1500, deadline);
    let frames = 250;
    let after = spawned_after(&mut app, frames, deadline);
    // Everything that flows in pours over the lip once the river has settled.
    let expected = RATE * Q * NY as f32 * DT * frames as f32;
    let thrown = (after - before) as f32;
    assert!(
        (thrown - expected).abs() < 0.25 * expected,
        "the waterfall threw {thrown} particles over {frames} frames, expected about {expected}"
    );

    // The same river without the drop throws none.
    let mut app = channel_app(0.0);
    let flat = spawned_after(&mut app, 1500, deadline);
    assert_eq!(
        flat, 0,
        "a river without a waterfall threw {flat} particles"
    );
}
