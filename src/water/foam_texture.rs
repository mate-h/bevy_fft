//! Tiling foam texture that the water surface thresholds by coverage.
//!
//! The texture holds values whose histogram is flat on [0, 1], so a pixel is foam where the value
//! exceeds `1 - coverage` and the share of foam pixels equals the coverage the foam sim reports.
//! The pattern is two octaves of Worley cell edges, which keeps foam along the boundaries between
//! bubble cells. As coverage drops, the threshold rises and only the brightest edges stay, so
//! thinning foam breaks into lace, as with Crest's black-point fade.
//!
//! The green and blue channels hold the slopes along X and Z of a tiling field of short ripples,
//! the flow detail the water surface lays over flowing water, encoded as
//! `0.5 + slope / (2 RIPPLE_SLOPE_RANGE)` for slopes scaled to a root mean square of one.

use bevy::{
    asset::RenderAssetUsages,
    image::{ImageAddressMode, ImageFilterMode, ImageSampler, ImageSamplerDescriptor},
    prelude::*,
    render::render_resource::{Extent3d, TextureDimension, TextureFormat},
};

/// Texels along each edge of the foam texture.
pub const WATER_FOAM_TEXTURE_SIZE: u32 = 256;

/// Handle of the generated foam texture, shared by every water surface material.
#[derive(Resource, Clone)]
pub struct WaterFoamTexture(pub Handle<Image>);

pub(super) fn create_water_foam_texture(mut commands: Commands, mut images: ResMut<Assets<Image>>) {
    let image = foam_texture_image(WATER_FOAM_TEXTURE_SIZE);
    commands.insert_resource(WaterFoamTexture(images.add(image)));
}

/// 32-bit integer hash, from Wellons' `lowbias32`.
fn hash(mut x: u32) -> u32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb_352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846c_a68b);
    x ^ (x >> 16)
}

/// Feature point of periodic cell `(x, y)` in a grid of `cells` per edge, in cell units within
/// the cell.
fn feature(x: i32, y: i32, cells: i32, seed: u32) -> Vec2 {
    let (x, y) = (x.rem_euclid(cells) as u32, y.rem_euclid(cells) as u32);
    let h = hash(x ^ hash(y ^ hash(seed)));
    let h2 = hash(h);
    Vec2::new(h as f32, h2 as f32) / u32::MAX as f32
}

/// Distance from `p` to the nearest cell boundary of a periodic Worley grid, as the gap between
/// the nearest and second nearest feature points. `p` is in cell units.
fn cell_edge_gap(p: Vec2, cells: i32, seed: u32) -> f32 {
    let base = p.floor();
    let (mut f1, mut f2) = (f32::MAX, f32::MAX);
    for dy in -1..=1 {
        for dx in -1..=1 {
            let cell = base + Vec2::new(dx as f32, dy as f32);
            let point = cell + feature(cell.x as i32, cell.y as i32, cells, seed);
            let d = p.distance(point);
            if d < f1 {
                f2 = f1;
                f1 = d;
            } else if d < f2 {
                f2 = d;
            }
        }
    }
    f2 - f1
}

/// Pattern values before their histogram is flattened. Larger values sit on cell edges.
fn foam_pattern(size: u32) -> Vec<f32> {
    const LARGE_CELLS: i32 = 12;
    const SMALL_CELLS: i32 = 40;
    (0..size * size)
        .map(|i| {
            let uv = (Vec2::new((i % size) as f32, (i / size) as f32) + 0.5) / size as f32;
            let large = cell_edge_gap(uv * LARGE_CELLS as f32, LARGE_CELLS, 1);
            let small = cell_edge_gap(uv * SMALL_CELLS as f32, SMALL_CELLS, 2);
            -(0.65 * large + 0.35 * small)
        })
        .collect()
}

/// Shortest and longest ripple across the foam texture's tile, as a share of the tile. With the
/// water surface's 6 m tile they are 12 cm and 60 cm, capillary-gravity ripples a few texels or more
/// across.
const RIPPLE_SHORTEST: f32 = 0.02;
const RIPPLE_LONGEST: f32 = 0.1;
/// Slopes the encoding spans either side of zero, in root mean squares. Matches
/// `FLOW_DETAIL_SLOPE_RANGE` in `surface.wesl`.
pub const RIPPLE_SLOPE_RANGE: f32 = 4.0;

/// Slopes along X and Z of a tiling ripple field at `size` texels per edge, scaled to a root mean
/// square slope of one. A sum of waves with whole numbers of cycles across the tile, random
/// directions and phases, and heights falling as the square of the wavenumber, so each band
/// carries about the same slope.
fn ripple_slopes(size: u32) -> Vec<Vec2> {
    let (low, high) = (1.0 / RIPPLE_LONGEST, 1.0 / RIPPLE_SHORTEST);
    let mut waves = Vec::new();
    let reach = high.ceil() as i32;
    for n in -reach..=reach {
        for m in 0..=reach {
            let k = Vec2::new(m as f32, n as f32);
            let length = k.length();
            if (m == 0 && n <= 0) || length < low || length > high {
                continue;
            }
            // Keep about one in five waves, so the field has no lattice look.
            let h = hash((m as u32) ^ hash((n + 1000) as u32 ^ hash(7)));
            if !h.is_multiple_of(5) {
                continue;
            }
            let phase = hash(h) as f32 / u32::MAX as f32 * std::f32::consts::TAU;
            waves.push((k, phase, 1.0 / (length * length)));
        }
    }
    let mut slopes: Vec<Vec2> = (0..size * size)
        .map(|i| {
            let uv = Vec2::new((i % size) as f32, (i / size) as f32) / size as f32;
            waves.iter().fold(Vec2::ZERO, |sum, (k, phase, amplitude)| {
                let angle = std::f32::consts::TAU * k.dot(uv) + phase;
                sum - *k * (*amplitude * angle.sin())
            })
        })
        .collect();
    let rms = (slopes.iter().map(|s| s.length_squared()).sum::<f32>() / slopes.len() as f32 / 2.0)
        .sqrt()
        .max(1e-12);
    for slope in &mut slopes {
        *slope /= rms;
    }
    slopes
}

/// Replaces each value by its rank, so the result is evenly spread over [0, 1].
fn flatten_histogram(values: &[f32]) -> Vec<f32> {
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|a, b| values[*a].total_cmp(&values[*b]));
    let mut ranks = vec![0.0; values.len()];
    let last = (values.len().max(2) - 1) as f32;
    for (rank, index) in order.into_iter().enumerate() {
        ranks[index] = rank as f32 / last;
    }
    ranks
}

/// Foam texture of `size` texels per edge: the foam pattern in red and the ripple slopes in green
/// and blue, 8 bits each, repeating and linearly filtered.
pub fn foam_texture_image(size: u32) -> Image {
    let values = flatten_histogram(&foam_pattern(size));
    let slopes = ripple_slopes(size);
    let encode =
        |s: f32| ((0.5 + s / (2.0 * RIPPLE_SLOPE_RANGE)).clamp(0.0, 1.0) * 255.0).round() as u8;
    let data: Vec<u8> = values
        .iter()
        .zip(&slopes)
        .flat_map(|(v, s)| [(v * 255.0).round() as u8, encode(s.x), encode(s.y), 255])
        .collect();
    let mut image = Image::new(
        Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        data,
        TextureFormat::Rgba8Unorm,
        RenderAssetUsages::RENDER_WORLD,
    );
    image.sampler = ImageSampler::Descriptor(ImageSamplerDescriptor {
        address_mode_u: ImageAddressMode::Repeat,
        address_mode_v: ImageAddressMode::Repeat,
        mag_filter: ImageFilterMode::Linear,
        min_filter: ImageFilterMode::Linear,
        ..default()
    });
    image
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn histogram_is_flat() {
        let image = foam_texture_image(64);
        let data: Vec<u8> = image.data.unwrap().iter().step_by(4).copied().collect();
        let mut bins = [0usize; 4];
        for v in &data {
            bins[(*v as usize * 4 / 256).min(3)] += 1;
        }
        for count in bins {
            assert!(
                (count as f32 / data.len() as f32 - 0.25).abs() < 0.01,
                "{bins:?}"
            );
        }
    }

    #[test]
    fn threshold_share_matches_coverage() {
        let values = flatten_histogram(&foam_pattern(64));
        for coverage in [0.05f32, 0.3, 0.8] {
            let share =
                values.iter().filter(|v| **v > 1.0 - coverage).count() as f32 / values.len() as f32;
            assert!((share - coverage).abs() < 0.01, "{coverage} {share}");
        }
    }

    #[test]
    fn ripples_tile_with_unit_slope() {
        let size = 64;
        let slopes = ripple_slopes(size);
        let mean = slopes.iter().sum::<Vec2>() / slopes.len() as f32;
        let rms =
            (slopes.iter().map(|s| s.length_squared()).sum::<f32>() / slopes.len() as f32 / 2.0)
                .sqrt();
        assert!(
            mean.length() < 0.05 && (rms - 1.0).abs() < 1e-3,
            "{mean} {rms}"
        );
        // Almost no slope falls outside the encoded range.
        let clipped = slopes
            .iter()
            .filter(|s| s.abs().max_element() > RIPPLE_SLOPE_RANGE)
            .count();
        assert!(clipped * 1000 < slopes.len(), "{clipped} clipped");
        // Opposite edges continue each other: the step across the seam is like any other step.
        let at = |x: u32, y: u32| slopes[(y * size + x) as usize];
        let seam: f32 = (0..size).map(|y| at(0, y).distance(at(size - 1, y))).sum();
        let inside: f32 = (0..size).map(|y| at(1, y).distance(at(0, y))).sum();
        assert!(seam < 1.5 * inside, "{seam} {inside}");
    }

    #[test]
    fn pattern_tiles() {
        // Feature points repeat with the grid, so opposite edges see the same cells.
        assert_eq!(feature(-1, 3, 12, 1), feature(11, 3, 12, 1));
        assert_eq!(feature(5, 12, 12, 1), feature(5, 0, 12, 1));
    }
}
