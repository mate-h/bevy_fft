//! Checks a baked terrain's river banks against the river's settled water.
//!
//! ```sh
//! cargo run --release --example river_banks -- <terrain> [--cell 2] [--seconds 600]
//! cargo run --release --example river_banks -- --demo
//! ```
//!
//! `<terrain>` is a folder under `assets/terrains` that `tools/bake_topobathy.py` wrote, with
//! `height.r32` and the rivers in `manifest.json`. Each river with a `discharge` flows into a
//! shallow water grid over the whole square where its spline enters, and out at normal depth where
//! it leaves. The grid starts dry and runs headless until the river settles. Then
//! `check_river_banks` reports water that left each channel and stretches of bank lower than the
//! settled surface, with where they are along the river and in the world. `--demo` runs a small
//! valley with one notched bank instead.

use bevy::{
    app::PluginsState,
    prelude::*,
    render::gpu_readback::{Readback, ReadbackComplete},
    time::TimeUpdateStrategy,
    window::ExitCondition,
    winit::WinitPlugin,
};
use bevy_fft::shallow_water::*;
use std::{path::PathBuf, thread, time::Duration};

const G: f32 = 9.81;
const MANNING_N: f32 = 0.035;

/// The terrain and rivers to check, with heights in meters at cell centres, rows along +Z.
struct Site {
    cells: u32,
    cell_size: f32,
    bed: Vec<f32>,
    rivers: Vec<RiverSpline>,
    seconds: f32,
}

fn arg(args: &[String], name: &str) -> Option<f32> {
    let at = args.iter().position(|a| a == name)?;
    args.get(at + 1)?.parse().ok()
}

fn load(name: &str, args: &[String]) -> Site {
    let folder = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("assets/terrains")
        .join(name);
    let manifest: serde_json::Value = serde_json::from_slice(
        &std::fs::read(folder.join("manifest.json")).expect("no manifest.json; bake the terrain"),
    )
    .expect("manifest.json is not JSON");
    let n = manifest["size"][0].as_u64().unwrap() as usize;
    let side = manifest["side_m"].as_f64().unwrap() as f32;
    let raw = std::fs::read(folder.join("height.r32"))
        .expect("no height.r32; bake the terrain again with the current tool");
    let heights: Vec<f32> = raw
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    assert_eq!(
        heights.len(),
        n * n,
        "height.r32 does not match the manifest"
    );
    let texel = side / n as f32;
    let height = |p: Vec2| {
        let f = (p + 0.5 * side) / texel - 0.5;
        let i = f.floor();
        let t = f - i;
        let at = |di: f32, dj: f32| {
            let c = (i + Vec2::new(di, dj)).clamp(Vec2::ZERO, Vec2::splat((n - 1) as f32));
            heights[c.y as usize * n + c.x as usize]
        };
        let a = at(0.0, 0.0) + (at(1.0, 0.0) - at(0.0, 0.0)) * t.x;
        let b = at(0.0, 1.0) + (at(1.0, 1.0) - at(0.0, 1.0)) * t.x;
        a + (b - a) * t.y
    };
    let cell_size = arg(args, "--cell").unwrap_or(2.0);
    let cells = (side / cell_size).round() as u32;
    let bed = (0..cells * cells)
        .map(|i| {
            let c = Vec2::new((i % cells) as f32, (i / cells) as f32) + 0.5;
            height((c - 0.5 * cells as f32) * cell_size)
        })
        .collect();
    let rivers = manifest["rivers"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|river| {
            let discharge = river["discharge"].as_f64()? as f32;
            let points = river["points"]
                .as_array()?
                .iter()
                .map(|p| Vec2::new(p[0].as_f64().unwrap() as f32, p[1].as_f64().unwrap() as f32))
                .collect();
            Some(RiverSpline {
                points,
                width: river["width"].as_f64()? as f32,
                discharge,
                outflow_slope: river["outflow_slope"].as_f64().unwrap_or(0.001) as f32,
            })
        })
        .collect::<Vec<_>>();
    assert!(
        !rivers.is_empty(),
        "no river in manifest.json has a discharge to check"
    );
    Site {
        cells,
        cell_size,
        bed,
        rivers,
        seconds: arg(args, "--seconds").unwrap_or(600.0),
    }
}

/// A 96 m valley falling along +x with an 8 m channel, and the left bank cut low for 8 m.
fn demo() -> Site {
    let cells = 96;
    let bed = (0..cells * cells)
        .map(|i| {
            let (x, z) = ((i % cells) as f32 - 47.5, (i / cells) as f32 - 47.5);
            let floor = 0.002 * (48.0 - x);
            if z.abs() < 4.0 {
                floor - 1.0
            } else if z.abs() < 7.0 {
                if z > 0.0 && (-8.0..0.0).contains(&x) {
                    floor - 0.8
                } else {
                    floor + 0.5
                }
            } else {
                floor - 0.7
            }
        })
        .collect();
    Site {
        cells,
        cell_size: 1.0,
        bed,
        rivers: vec![RiverSpline {
            points: vec![Vec2::new(-60.0, 0.0), Vec2::new(60.0, 0.0)],
            width: 8.0,
            discharge: 5.0,
            outflow_slope: 0.002,
        }],
        seconds: 240.0,
    }
}

#[derive(Resource, Default)]
struct Capture {
    bytes: Vec<u8>,
    serial: u64,
}

fn step(app: &mut App) {
    app.update();
    assert!(app.should_exit().is_none(), "the renderer failed");
    thread::sleep(Duration::from_millis(1));
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let site = match args.first().map(String::as_str) {
        Some("--demo") => demo(),
        Some(name) if !name.starts_with("--") => load(name, &args),
        _ => {
            eprintln!("usage: river_banks <terrain> [--cell 2] [--seconds 600] | --demo");
            std::process::exit(2);
        }
    };
    // Solver heights stay positive.
    let lowest = site.bed.iter().copied().fold(f32::INFINITY, f32::min);
    let offset = 1.0 - lowest;
    let deepest = site.bed.iter().copied().fold(f32::NEG_INFINITY, f32::max) - lowest;
    // A Courant number near 0.3 for water as deep as the relief, capped for shallow sites.
    let dt = (0.3 * site.cell_size / (G * deepest.clamp(1.0, 10.0)).sqrt()).min(0.1);
    let frames = (site.seconds / dt).ceil() as u32;

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
    .add_plugins(ShallowWaterPlugin)
    .insert_resource(TimeUpdateStrategy::ManualDuration(Duration::from_secs_f32(
        1.0 / 60.0,
    )))
    .init_resource::<Capture>();
    let placement = ShallowWaterPlacement {
        center: Vec2::ZERO,
        cell_size: site.cell_size,
        datum: -offset,
        blend_cells: 0.0,
    };
    let mut images = app.world_mut().resource_mut::<Assets<Image>>();
    let mut controller = ShallowWaterController::new(&mut images, site.cells, site.cells);
    let bed: Vec<f32> = site.bed.iter().map(|h| h + offset).collect();
    controller.bed_source = images.add(bed_source_image(site.cells, site.cells, &bed));
    controller.preset_index = SHALLOW_WATER_PRESET_BED_SOURCE;
    controller.fill_level = 0.0;
    controller.gravity = G;
    controller.dt = dt;
    controller.manning_n = MANNING_N;
    controller.placement = Some(placement);
    let mut profile = ShallowWaterBorderProfile::new(UVec2::splat(site.cells));
    for river in &site.rivers {
        profile.add_river(river, &placement);
    }
    controller.border_profile = Some(images.add(profile.image()));
    let bed_water = controller.bed_water.clone();
    app.insert_resource(controller);
    while app.plugins_state() == PluginsState::Adding {
        bevy::tasks::tick_global_task_pools_on_main_thread();
        thread::sleep(Duration::from_millis(1));
    }
    app.finish();
    app.cleanup();
    app.world_mut().spawn(Readback::texture(bed_water)).observe(
        |event: On<ReadbackComplete>, mut capture: ResMut<Capture>| {
            capture.bytes = event.data.clone();
            capture.serial += 1;
        },
    );

    println!(
        "{} cells of {} m, {} s of river in {frames} steps of {dt:.3} s",
        site.cells, site.cell_size, site.seconds
    );
    for frame in 0..frames {
        step(&mut app);
        if frame % 1000 == 999 {
            println!("  {:.0} s", (frame + 1) as f32 * dt);
        }
    }
    app.world_mut()
        .resource_mut::<ShallowWaterController>()
        .paused = true;
    let serial = app.world().resource::<Capture>().serial;
    while app.world().resource::<Capture>().serial <= serial + 1 {
        step(&mut app);
    }
    let bytes = &app.world().resource::<Capture>().bytes;
    let row = bytes.len() / site.cells as usize;
    let lane = |o: usize| f32::from_le_bytes(bytes[o..o + 4].try_into().unwrap());
    let (bed, depth): (Vec<f32>, Vec<f32>) = (0..(site.cells * site.cells) as usize)
        .map(|i| {
            let o = (i / site.cells as usize) * row + (i % site.cells as usize) * 16;
            (lane(o) - offset, lane(o + 4))
        })
        .unzip();
    let grid = SettledGrid {
        cells: UVec2::splat(site.cells),
        cell_size: site.cell_size,
        corner: Vec2::splat(-0.5 * site.cells as f32 * site.cell_size),
        bed: &bed,
        depth: &depth,
    };

    let mut problems = 0;
    for (index, river) in site.rivers.iter().enumerate() {
        let length: f32 = river
            .sample(32)
            .windows(2)
            .map(|p| (p[1] - p[0]).length())
            .sum();
        // Skip a few widths at each end, where the river enters and leaves.
        let end = 3.0 * river.width;
        let report = check_river_banks(
            river,
            &grid,
            &RiverBankCheck {
                along: (end, length - end),
                ..default()
            },
        );
        println!(
            "river {index}: {} m wide, {} m³/s, {length:.0} m long",
            river.width, river.discharge
        );
        if report.is_clean() {
            println!("  banks hold");
        }
        for leak in &report.leaks {
            println!(
                "  leak meeting the channel from {:.0} to {:.0} m along: {:.0} m² wet outside it, \
                 up to {:.2} m deep, deepest at world ({:.1}, {:.1})",
                leak.along.0,
                leak.along.1,
                leak.area,
                leak.max_depth,
                leak.deepest.x,
                leak.deepest.y
            );
        }
        for low in &report.low_banks {
            println!(
                "  low {} bank from {:.0} to {:.0} m along: crest {:.2} m under a surface at \
                 {:.2} m, at world ({:.1}, {:.1})",
                if low.side > 0.0 { "left" } else { "right" },
                low.along.0,
                low.along.1,
                low.crest,
                low.surface,
                low.at.x,
                low.at.y
            );
        }
        problems += report.leaks.len() + report.low_banks.len();
    }
    std::process::exit(i32::from(problems > 0));
}
