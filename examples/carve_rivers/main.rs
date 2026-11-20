//! Carves rivers into a bevy_terrain height tile set with `river_geometry`, where the elevation
//! model recorded their water surface instead of their bed. No window, no GPU.
//!
//! ```text
//! cargo run --release --example carve_rivers -- <terrain folder> <rivers.json>
//! cargo run --release --example carve_rivers -- assets/terrains/kauai_shoreline examples/beach/rivers.json
//! ```
//!
//! On the first run it copies the tile set's `height` folder to `height_uncarved` and from then
//! on always routes over that copy, so running it again gives the same tiles. Delete
//! `height_uncarved` after re-baking the terrain.
//!
//! Each river in the JSON file:
//!
//! ```text
//! { "source": [x, z],          where it comes onto the map, in metres from the centre
//!   "bankfull": 10.0,          discharge (m³/s) that sizes the channel (ChannelModel)
//!   "breach_depth": 1.5,       rises up to this high (m) are a recorded water surface and get cut
//!   "sea_level": 0.0,          the route ends at the first texel below this
//!   "smooth": 7.0,             centreline smoothing radius (m)
//!   "detail": 1.0,             steps, pools and point bars, 0 to 1
//!   "bank_width": 4.0,         graded banks beside the channel (m)
//!   "seed": 1 }
//! ```
//!
//! It routes and profiles each river on the tile set's `work_lod` mosaic (the `--lod` flag,
//! default 2, about 1 m on the beach), rasterizes and carves a window around it, and turns the
//! result into a height change per texel. Every tile of every LOD, borders included, then takes
//! that change: sampled between texels on finer LODs, averaged over each texel's footprint on
//! coarser ones. Texels away from the rivers keep their bytes.
use river_geometry::{
    ChannelModel, ChannelShape, NO_RIVER, River, bilinear, carve, long_profile, rasterize, route,
    smooth_path,
};
use std::path::{Path, PathBuf};

const BORDER: usize = 1;

struct Tiles {
    lod: u32,
    /// Tiles per edge.
    count: usize,
    /// Texels per tile edge, borders included.
    size: usize,
    /// Row-major tiles, row-major texels, rows along +Z.
    data: Vec<Vec<f32>>,
}

fn tile_path(root: &Path, lod: u32, x: usize, y: usize) -> PathBuf {
    root.join(format!("{lod}/0_0/0_{lod}_{x}_{y}.tif"))
}

fn read_lod(root: &Path, lod: u32) -> Tiles {
    let count = 1usize << lod;
    let mut data = Vec::new();
    let mut size = 0;
    for y in 0..count {
        for x in 0..count {
            let path = tile_path(root, lod, x, y);
            let file = std::fs::File::open(&path).unwrap_or_else(|e| panic!("{path:?}: {e}"));
            let mut decoder = tiff::decoder::Decoder::new(std::io::BufReader::new(file)).unwrap();
            size = decoder.dimensions().unwrap().0 as usize;
            let tiff::decoder::DecodingResult::F32(tile) = decoder.read_image().unwrap() else {
                panic!("{path:?} is not f32");
            };
            data.push(tile);
        }
    }
    Tiles {
        lod,
        count,
        size,
        data,
    }
}

fn write_tile(path: &Path, size: usize, data: &[f32]) {
    let file = std::fs::File::create(path).unwrap();
    let mut encoder = tiff::encoder::TiffEncoder::new(std::io::BufWriter::new(file)).unwrap();
    encoder
        .write_image::<tiff::encoder::colortype::Gray32Float>(size as u32, size as u32, data)
        .unwrap();
}

/// The tiles' centre texels as one row-major grid, `count × (size - 2)` texels on an edge.
fn mosaic(tiles: &Tiles) -> (Vec<f32>, usize) {
    let centre = tiles.size - 2 * BORDER;
    let n = tiles.count * centre;
    let mut out = vec![0.0; n * n];
    for (t, tile) in tiles.data.iter().enumerate() {
        let (tx, ty) = (t % tiles.count, t / tiles.count);
        for y in 0..centre {
            for x in 0..centre {
                out[(ty * centre + y) * n + tx * centre + x] =
                    tile[(y + BORDER) * tiles.size + x + BORDER];
            }
        }
    }
    (out, n)
}

#[derive(Clone)]
struct Spec {
    source: [f32; 2],
    bankfull: f32,
    breach_depth: f32,
    sea_level: f32,
    smooth: f32,
    detail: f32,
    bank_width: f32,
    seed: u32,
}

fn specs(path: &Path) -> Vec<Spec> {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{path:?}: {e}"));
    let json: serde_json::Value = serde_json::from_str(&text).unwrap();
    let num = |v: &serde_json::Value, key: &str, default: f64| {
        v.get(key).and_then(|x| x.as_f64()).unwrap_or(default) as f32
    };
    json.as_array()
        .expect("a JSON list of rivers")
        .iter()
        .map(|v| {
            let source = v["source"].as_array().expect("source: [x, z]");
            Spec {
                source: [
                    source[0].as_f64().unwrap() as f32,
                    source[1].as_f64().unwrap() as f32,
                ],
                bankfull: num(v, "bankfull", 10.0),
                breach_depth: num(v, "breach_depth", 1.5),
                sea_level: num(v, "sea_level", 0.0),
                smooth: num(v, "smooth", 7.0),
                detail: num(v, "detail", 1.0),
                bank_width: num(v, "bank_width", 4.0),
                seed: num(v, "seed", 1.0) as u32,
            }
        })
        .collect()
}

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut lod = 2u32;
    if let Some(i) = args.iter().position(|a| a == "--lod") {
        lod = args[i + 1].parse().unwrap();
        args.drain(i..=i + 1);
    }
    let [root, rivers] = &args[..] else {
        panic!("usage: carve_rivers <terrain folder> <rivers.json> [--lod n]");
    };
    let root = Path::new(root);
    let (carved, uncarved) = (root.join("height"), root.join("height_uncarved"));
    if !uncarved.exists() {
        let status = std::process::Command::new("cp")
            .arg("-R")
            .arg(&carved)
            .arg(&uncarved)
            .status()
            .unwrap();
        assert!(status.success(), "copying {carved:?} to {uncarved:?}");
        println!("kept the uncarved tiles in {uncarved:?}");
    }
    let config = std::fs::read_to_string(root.join("config.tc.ron")).unwrap();
    let side: f32 = config
        .split("side_length:")
        .nth(1)
        .and_then(|s| s.trim_start().split(|c: char| c == ',' || c == ')').next())
        .and_then(|s| s.trim().parse().ok())
        .expect("side_length in config.tc.ron");
    let lods: u32 = config
        .split("lod_count:")
        .nth(1)
        .and_then(|s| s.trim_start().split(',').next())
        .and_then(|s| s.trim().parse().ok())
        .expect("lod_count in config.tc.ron");

    let work = read_lod(&uncarved, lod);
    let (ground, n) = mosaic(&work);
    let mpt = side / n as f32;
    let to_texel = |p: [f32; 2]| [p[0] / mpt + 0.5 * n as f32, p[1] / mpt + 0.5 * n as f32];
    let mut delta = vec![0.0f32; n * n];
    let mut bed = ground.clone();
    for (k, spec) in specs(Path::new(rivers)).iter().enumerate() {
        let shape = ChannelShape {
            model: ChannelModel::default(),
            metres_per_texel: mpt,
            cells_per_height: 1.0 / mpt,
            profile: 2.0,
            max_depth: 10.0,
            bank_width: spec.bank_width / mpt,
            bank_slope: 0.0,
        };
        let cells = route(&bed, [n as u32; 2], to_texel(spec.source), |i| {
            bed[i] < spec.sea_level
        });
        let path = smooth_path(&cells, (spec.smooth / mpt).round() as usize);
        if path.len() < 4 {
            println!("river {k}: goes nowhere from {:?}", spec.source);
            continue;
        }
        let along: Vec<f32> = path
            .iter()
            .map(|&p| bilinear(&bed, [n as u32; 2], p))
            .collect();
        let profile = long_profile(&along, spec.breach_depth);
        let cut = along
            .iter()
            .zip(&profile)
            .map(|(a, p)| a - p)
            .fold(0.0, f32::max);
        // Rasterize a window around the river.
        let (w, h) = shape.section(spec.bankfull);
        let margin = (2.0 * w + shape.bank_width + 4.0).ceil() as usize;
        let lo = path
            .iter()
            .fold([f32::MAX; 2], |m, p| [m[0].min(p[0]), m[1].min(p[1])]);
        let hi = path
            .iter()
            .fold([f32::MIN; 2], |m, p| [m[0].max(p[0]), m[1].max(p[1])]);
        let x0 = (lo[0] as usize).saturating_sub(margin);
        let y0 = (lo[1] as usize).saturating_sub(margin);
        let x1 = (hi[0] as usize + margin).min(n - 1);
        let y1 = (hi[1] as usize + margin).min(n - 1);
        let (ww, wh) = (x1 - x0 + 1, y1 - y0 + 1);
        let river = River {
            path: path
                .iter()
                .map(|p| [p[0] - x0 as f32, p[1] - y0 as f32])
                .collect(),
            flow: vec![spec.bankfull; path.len()],
            profile,
            seed: spec.seed,
        };
        let window: Vec<f32> = (0..wh)
            .flat_map(|y| (0..ww).map(move |x| (x, y)))
            .map(|(x, y)| bed[(y0 + y) * n + x0 + x])
            .collect();
        let cells = rasterize(&[river], &shape, spec.detail, 0.0, [ww as u32, wh as u32]);
        let carved_window = carve(&window, &cells);
        let mut lowered = 0;
        for y in 0..wh {
            for x in 0..ww {
                let i = (y0 + y) * n + x0 + x;
                let new = carved_window[y * ww + x];
                if cells[y * ww + x][0] < NO_RIVER && new < bed[i] {
                    bed[i] = new;
                    delta[i] = bed[i] - ground[i];
                    lowered += 1;
                }
            }
        }
        let first = path[0];
        let last = path[path.len() - 1];
        let at = |p: [f32; 2]| [(p[0] - 0.5 * n as f32) * mpt, (p[1] - 0.5 * n as f32) * mpt];
        println!(
            "river {k}: {} m from {:?} to {:?}, channel {:.1} m x {:.2} m, profile cuts up to {cut:.2} m, {lowered} texels lowered",
            (path.len() as f32 * mpt).round(),
            at(first).map(|v| v.round()),
            at(last).map(|v| v.round()),
            2.0 * w * mpt,
            h,
        );
    }

    // Every LOD takes the change. Texel centres of tile (tx, ty) at LOD l sit at
    // ((t + (i + 0.5 - BORDER) / centre) / count - 0.5) × side.
    let field = [n as u32; 2];
    for l in 0..lods {
        let tiles = read_lod(&uncarved, l);
        let centre = (tiles.size - 2 * BORDER) as f32;
        let span = side / tiles.count as f32;
        let texel = span / centre;
        let ratio = texel / mpt;
        let mut written = 0;
        for (t, tile) in tiles.data.iter().enumerate() {
            let (tx, ty) = (t % tiles.count, t / tiles.count);
            let mut out = tile.clone();
            let mut changed = false;
            for y in 0..tiles.size {
                for x in 0..tiles.size {
                    let world = |i: usize, t: usize| {
                        (t as f32 + (i as f32 + 0.5 - BORDER as f32) / centre) * span - 0.5 * side
                    };
                    let p = to_texel([world(x, tx), world(y, ty)]);
                    let d = if ratio <= 1.0 {
                        bilinear(&delta, field, p)
                    } else {
                        // Mean over the footprint of the coarse texel.
                        let r = 0.5 * ratio;
                        let (ax, bx) = ((p[0] - r).max(0.0) as usize, ((p[0] + r) as usize).min(n));
                        let (ay, by) = ((p[1] - r).max(0.0) as usize, ((p[1] + r) as usize).min(n));
                        let mut sum = 0.0;
                        for yy in ay..by {
                            for xx in ax..bx {
                                sum += delta[yy * n + xx];
                            }
                        }
                        sum / ((bx - ax) * (by - ay)).max(1) as f32
                    };
                    if d < 0.0 {
                        out[y * tiles.size + x] += d;
                        changed = true;
                    }
                }
            }
            let path = tile_path(&carved, tiles.lod, tx, ty);
            if changed {
                write_tile(&path, tiles.size, &out);
                written += 1;
            } else {
                // Back to the uncarved bytes, in case an earlier run carved it.
                std::fs::copy(tile_path(&uncarved, tiles.lod, tx, ty), &path).unwrap();
            }
        }
        println!(
            "LOD {l}: {written} of {} tiles carved",
            tiles.count * tiles.count
        );
    }
}
