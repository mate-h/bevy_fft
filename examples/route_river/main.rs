//! Routes the beach's river with `river_geometry` over the LiDAR tiles and compares the route and
//! its long profile with the hand-traced `CHANNEL` in the beach example. No window, no GPU.
//!
//! ```text
//! cargo run --release --example route_river -- [tile folder] [breach depth m]
//! ```
//!
//! The tile folder defaults to the beach's LOD 2 heights. Point it at an uncarved copy to see
//! what the route makes of the raw LiDAR. The route starts where the river comes onto the map and
//! ends at the first texel below sea level; the profile cuts rises up to the breach depth
//! (default 1.5 m).
use river_geometry::{bilinear, long_profile, route, smooth_path};

/// Edge of the terrain square (m), its tiles per edge at LOD 2, and the grid the route runs on.
const SIDE: f32 = 2048.0;
const TILES: usize = 4;
const CELL: f32 = 2.0;
/// Where the river comes onto the map (map metres, x east, z as the beach's grid rows).
const SOURCE: [f32; 2] = [-905.4, -971.6];
/// The beach's hand-traced thalweg through the reach the LiDAR flattened (x, z, bed above sea).
const CHANNEL: [[f32; 3]; 15] = [
    [-854.1, -829.3, 1.45],
    [-849.0, -815.0, 1.41],
    [-837.6, -803.0, 1.36],
    [-837.9, -787.0, 1.31],
    [-837.6, -771.0, 1.27],
    [-834.7, -755.0, 1.22],
    [-830.4, -739.0, 1.17],
    [-828.4, -723.0, 1.13],
    [-824.1, -707.0, 1.08],
    [-821.3, -691.0, 1.03],
    [-819.6, -675.0, 0.99],
    [-822.1, -659.0, 0.94],
    [-821.6, -643.0, 0.90],
    [-813.0, -629.0, 0.85],
    [-804.7, -620.7, -0.70],
];

/// Height above sea at map `xz`, from 4×4 tiles with a 1-texel border, as the beach reads them.
fn height(tiles: &[Vec<f32>], size: usize, xz: [f32; 2]) -> f32 {
    let centre = (size - 2) as f32;
    let u = |v: f32| ((v / SIDE + 0.5).clamp(0.0, 1.0 - 1e-6)) * TILES as f32;
    let (u, v) = (u(xz[0]), u(xz[1]));
    let (tx, ty) = (u.floor(), v.floor());
    let data = &tiles[ty as usize * TILES + tx as usize];
    let (px, py) = ((u - tx) * centre - 0.5 + 1.0, (v - ty) * centre - 0.5 + 1.0);
    let (ix, iy) = (px.floor(), py.floor());
    let (fx, fy) = (px - ix, py - iy);
    let at = |dx: usize, dy: usize| {
        let x = (ix as usize + dx).min(size - 1);
        let y = (iy as usize + dy).min(size - 1);
        data[y * size + x]
    };
    let top = at(0, 0) + (at(1, 0) - at(0, 0)) * fx;
    let bottom = at(0, 1) + (at(1, 1) - at(0, 1)) * fx;
    top + (bottom - top) * fy
}

fn main() {
    let mut args = std::env::args().skip(1);
    let folder = args
        .next()
        .unwrap_or_else(|| "assets/terrains/kauai_shoreline/height/2".into());
    let breach: f32 = args.next().and_then(|v| v.parse().ok()).unwrap_or(1.5);
    let mut tiles = Vec::new();
    let mut size = 0;
    for y in 0..TILES {
        for x in 0..TILES {
            let path = format!("{folder}/0_0/0_2_{x}_{y}.tif");
            let file = std::fs::File::open(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
            let mut decoder = tiff::decoder::Decoder::new(std::io::BufReader::new(file)).unwrap();
            size = decoder.dimensions().unwrap().0 as usize;
            let tiff::decoder::DecodingResult::F32(data) = decoder.read_image().unwrap() else {
                panic!("{path} is not f32");
            };
            tiles.push(data);
        }
    }
    let n = (SIDE / CELL) as u32;
    let to_map = |p: [f32; 2]| {
        [
            (p[0] - 0.5 * n as f32) * CELL,
            (p[1] - 0.5 * n as f32) * CELL,
        ]
    };
    let heights: Vec<f32> = (0..n * n)
        .map(|i| {
            let p = [(i % n) as f32 + 0.5, (i / n) as f32 + 0.5];
            height(&tiles, size, to_map(p))
        })
        .collect();
    let source = [
        SOURCE[0] / CELL + 0.5 * n as f32,
        SOURCE[1] / CELL + 0.5 * n as f32,
    ];
    let cells = route(&heights, [n, n], source, |i| heights[i] < 0.0);
    // Smooth over about a channel width (14 m).
    let path = smooth_path(&cells, (7.0 / CELL).round() as usize);
    let ground: Vec<f32> = path
        .iter()
        .map(|&p| bilinear(&heights, [n, n], p))
        .collect();
    let profile = long_profile(&ground, breach);
    let map: Vec<[f32; 2]> = path.iter().map(|&p| to_map(p)).collect();
    let end = map.last().unwrap();
    println!(
        "route: {} cells, {} samples ({:.0} m), from ({:.0}, {:.0}) to ({:.0}, {:.0}), ground {:.2} to {:.2} m",
        cells.len(),
        path.len(),
        path.len() as f32 * CELL,
        map[0][0],
        map[0][1],
        end[0],
        end[1],
        ground[0],
        ground.last().unwrap()
    );
    let cut: f32 = ground
        .iter()
        .zip(&profile)
        .map(|(g, p)| g - p)
        .fold(0.0, f32::max);
    println!("profile: breach depth {breach} m cuts up to {cut:.2} m below the ground");
    println!("   x       z     | route offset | route profile | hand bed | ground");
    for c in CHANNEL {
        let (i, d) = map
            .iter()
            .enumerate()
            .map(|(i, p)| (i, ((p[0] - c[0]).powi(2) + (p[1] - c[1]).powi(2)).sqrt()))
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .unwrap();
        println!(
            "{:7.1} {:7.1} | {:9.1} m  | {:10.2} m  | {:6.2} m | {:5.2} m",
            c[0], c[1], d, profile[i], c[2], ground[i]
        );
    }
}
