//! Tiling foam texture that the water surface thresholds by coverage.
//!
//! The texture holds values whose histogram is flat on [0, 1], so a pixel is foam where the value
//! exceeds `1 - coverage` and the share of foam pixels equals the coverage the foam sim reports.
//! The pattern is two octaves of Worley cell edges, which keeps foam along the boundaries between
//! bubble cells. As coverage drops, the threshold rises and only the brightest edges stay, so
//! thinning foam breaks into lace, as with Crest's black-point fade.

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

/// Foam texture of `size` texels per edge in an 8-bit red channel, repeating and linearly filtered.
pub fn foam_texture_image(size: u32) -> Image {
    let values = flatten_histogram(&foam_pattern(size));
    let data: Vec<u8> = values.iter().map(|v| (v * 255.0).round() as u8).collect();
    let mut image = Image::new(
        Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        data,
        TextureFormat::R8Unorm,
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
        let data = image.data.unwrap();
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
    fn pattern_tiles() {
        // Feature points repeat with the grid, so opposite edges see the same cells.
        assert_eq!(feature(-1, 3, 12, 1), feature(11, 3, 12, 1));
        assert_eq!(feature(5, 12, 12, 1), feature(5, 0, 12, 1));
    }
}
