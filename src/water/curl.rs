//! The breaker curl: the overhanging lip of a plunging wave, which a heightfield cannot hold.
//!
//! After Malan (2022), the curling part of a breaker looks like one cross-section translating along
//! the wavefront, so one animated cross-section serves every breaker. [`WaterBreakerCurl`] sweeps
//! it along each [`FrontLine`] of the [`WaterBreakingFronts`] on the same entity, forward along the
//! front's direction of travel, and plays it by the front's age: a rising face, a lip thrown
//! forward, and a curl that falls back onto the water ahead and flattens out. The front ids keep a
//! line's age while it breaks, so its curl plays through. Malan bakes the cross-section from a
//! Houdini simulation and then hand-animates it so it ends flat. Here it is hand-authored as a few
//! keyframes, ending flat the same way. The inner loop of the tube is kept, since the camera can
//! look into it, and the mesh is double-sided.
//!
//! The mesh is rebuilt on the CPU at each readback. It is a curtain over the water surface that
//! the waves draw anyway: its edges lie on the water, and its height scales with the depth at the
//! front, as a breaker's height is about 0.78 of the depth it breaks in.

use bevy::{
    asset::RenderAssetUsages,
    mesh::{Indices, PrimitiveTopology, VertexAttributeValues},
    prelude::*,
};

use super::{FrontLine, WaterBreakingFronts};

/// Samples along the cross-section, from the trough behind the front to the water ahead.
const PROFILE_SAMPLES: usize = 24;

/// The cross-section at a few animation times, as points from the back of the wave to the front,
/// in breaker heights: forward along the front's travel, and up from the water surface. Each
/// starts and ends on the water, so the curl joins the surface along its edges.
const KEYFRAMES: [(f32, [[f32; 2]; 9]); 4] = [
    // A steep hump.
    (
        0.0,
        [
            [-2.0, 0.0],
            [-1.4, 0.15],
            [-0.9, 0.45],
            [-0.5, 0.75],
            [-0.2, 0.9],
            [0.05, 0.8],
            [0.3, 0.45],
            [0.6, 0.15],
            [1.0, 0.0],
        ],
    ),
    // The face steepens and the lip is thrown forward over it.
    (
        0.4,
        [
            [-2.0, 0.0],
            [-1.3, 0.2],
            [-0.8, 0.6],
            [-0.35, 0.95],
            [0.05, 1.05],
            [0.45, 0.95],
            [0.6, 0.65],
            [0.35, 0.3],
            [1.2, 0.0],
        ],
    ),
    // The lip curls down onto the water ahead, around a tube.
    (
        0.7,
        [
            [-2.0, 0.0],
            [-1.2, 0.2],
            [-0.7, 0.55],
            [-0.25, 0.8],
            [0.2, 0.85],
            [0.75, 0.7],
            [1.0, 0.35],
            [0.95, 0.05],
            [1.4, 0.0],
        ],
    ),
    // Spent: a low bore that ends flat.
    (
        1.0,
        [
            [-2.0, 0.0],
            [-1.4, 0.03],
            [-0.9, 0.06],
            [-0.4, 0.08],
            [0.0, 0.08],
            [0.4, 0.07],
            [0.8, 0.04],
            [1.2, 0.01],
            [1.6, 0.0],
        ],
    ),
];

/// Draws the breaker curl along the [`WaterBreakingFronts`] of its entity. The plugin spawns a
/// mesh for it as a child, with a material of [`Self::color`].
#[derive(Component, Clone, Debug, Reflect)]
#[reflect(Component)]
pub struct WaterBreakerCurl {
    /// Seconds from a front's first breaking to its curl lying flat.
    pub duration: f32,
    /// Breaker height over the water depth at the front. 0.78 is the breaker index of
    /// McCowan (1894) for a solitary wave.
    pub height_over_depth: f32,
    /// Breaker heights in meters stay within this range.
    pub height_range: Vec2,
    /// Points at each end of a line over which the curl tapers to nothing, so it doesn't end in a
    /// wall.
    pub taper_points: usize,
    /// Colour of the water in the curl, and of its foam toward the lip as it breaks.
    pub color: Color,
    pub foam: Color,
}

impl Default for WaterBreakerCurl {
    fn default() -> Self {
        Self {
            duration: 2.0,
            height_over_depth: 0.78,
            height_range: Vec2::new(0.2, 3.0),
            taper_points: 4,
            color: Color::srgba(0.09, 0.32, 0.36, 0.85),
            foam: Color::srgba(0.92, 0.95, 0.96, 0.95),
        }
    }
}

/// The curl's mesh entity and the readback it was built from.
#[derive(Component)]
pub(super) struct BreakerCurlMesh {
    owner: Entity,
    frame: u32,
}

/// Point `i` of the cross-section at animation time `t` from 0 to 1, in breaker heights, with the
/// share of foam on it. The keyframes are blended with a smoothstep, and the points between
/// keyframe points with a Catmull-Rom spline.
pub fn curl_profile(t: f32, samples: usize) -> Vec<(Vec2, f32)> {
    let t = t.clamp(0.0, 1.0);
    let k = KEYFRAMES
        .windows(2)
        .position(|w| t <= w[1].0)
        .unwrap_or(KEYFRAMES.len() - 2);
    let (t0, a) = KEYFRAMES[k];
    let (t1, b) = KEYFRAMES[k + 1];
    let w = ((t - t0) / (t1 - t0)).clamp(0.0, 1.0);
    let w = w * w * (3.0 - 2.0 * w);
    let key: Vec<Vec2> = a
        .iter()
        .zip(&b)
        .map(|(a, b)| Vec2::from(*a).lerp(Vec2::from(*b), w))
        .collect();
    let last = key.len() - 1;
    // Foam grows over the lip as it breaks, and covers the spent bore.
    let breaking = ((t - 0.3) / 0.4).clamp(0.0, 1.0);
    (0..samples)
        .map(|i| {
            let s = i as f32 / (samples - 1) as f32 * last as f32;
            let j = (s.floor() as usize).min(last - 1);
            let f = s - j as f32;
            let p = |n: isize| key[(j as isize + n).clamp(0, last as isize) as usize];
            let (p0, p1, p2, p3) = (p(-1), p(0), p(1), p(2));
            let point = 0.5
                * (2.0 * p1
                    + (p2 - p0) * f
                    + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * f * f
                    + (3.0 * p1 - p0 - 3.0 * p2 + p3) * f * f * f);
            let toward_lip = s / last as f32;
            (point, breaking * toward_lip.powf(1.5))
        })
        .collect()
}

/// Passes of the smoothing along a line before the sweep.
const SMOOTHING_PASSES: usize = 4;

/// `values` smoothed along the line with a [1, 2, 1] / 4 kernel, `passes` times, wrapping around
/// for a closed line and keeping the end values of an open one.
fn smooth<T>(values: &mut [T], closed: bool, passes: usize)
where
    T: Copy + std::ops::Add<Output = T> + std::ops::Mul<f32, Output = T>,
{
    let n = values.len();
    if n < 3 {
        return;
    }
    for _ in 0..passes {
        let old = values.to_vec();
        for i in 0..n {
            let (a, b) = if closed {
                ((i + n - 1) % n, (i + 1) % n)
            } else if i == 0 || i == n - 1 {
                continue;
            } else {
                (i - 1, i + 1)
            };
            values[i] = old[a] * 0.25 + old[i] * 0.5 + old[b] * 0.25;
        }
    }
}

/// Appends the curl of `line` to the mesh buffers. Positions, directions and heights are smoothed
/// along the line first: the traced points step from cell to cell, and the curl would show the
/// staircase.
fn sweep(
    line: &FrontLine,
    settings: &WaterBreakerCurl,
    positions: &mut Vec<[f32; 3]>,
    colors: &mut Vec<[f32; 4]>,
    indices: &mut Vec<u32>,
) {
    let n = line.points.len();
    if n < 2 {
        return;
    }
    let profile = curl_profile(line.age / settings.duration.max(1e-3), PROFILE_SAMPLES);
    let water = LinearRgba::from(settings.color);
    let foam = LinearRgba::from(settings.foam);
    let base = positions.len() as u32;
    let taper = settings.taper_points.max(1) as f32;
    let mut centers: Vec<Vec3> = line.points.iter().map(|p| p.position).collect();
    let mut directions: Vec<Vec2> = line.points.iter().map(|p| p.direction).collect();
    let mut heights: Vec<f32> = line
        .points
        .iter()
        .map(|p| {
            (settings.height_over_depth * p.depth * p.intensity.clamp(0.0, 1.0))
                .clamp(settings.height_range.x, settings.height_range.y)
        })
        .collect();
    smooth(&mut centers, line.closed, SMOOTHING_PASSES);
    smooth(&mut directions, line.closed, SMOOTHING_PASSES);
    smooth(&mut heights, line.closed, SMOOTHING_PASSES);
    for i in 0..n {
        let from_end = i.min(n - 1 - i) as f32;
        let fade = if line.closed {
            1.0
        } else {
            (from_end / taper).min(1.0)
        };
        let height = heights[i] * fade;
        let forward = Vec3::new(directions[i].x, 0.0, directions[i].y).normalize_or(Vec3::X);
        for (point, foam_share) in &profile {
            let at = centers[i] + forward * point.x * height + Vec3::Y * point.y * height;
            positions.push(at.to_array());
            colors.push(water.mix(&foam, *foam_share).to_f32_array());
        }
    }
    let rows = if line.closed { n } else { n - 1 };
    let columns = PROFILE_SAMPLES as u32;
    for i in 0..rows {
        let a = base + i as u32 * columns;
        let b = base + ((i + 1) % n) as u32 * columns;
        for j in 0..columns - 1 {
            indices.extend_from_slice(&[a + j, b + j, a + j + 1, a + j + 1, b + j, b + j + 1]);
        }
    }
}

/// The curl mesh of all `lines`, or `None` when there is nothing to draw.
pub fn curl_mesh(lines: &[FrontLine], settings: &WaterBreakerCurl) -> Option<Mesh> {
    let (mut positions, mut colors, mut indices) = (Vec::new(), Vec::new(), Vec::new());
    for line in lines {
        sweep(line, settings, &mut positions, &mut colors, &mut indices);
    }
    if indices.is_empty() {
        return None;
    }
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::RENDER_WORLD,
    )
    .with_inserted_attribute(
        Mesh::ATTRIBUTE_POSITION,
        VertexAttributeValues::Float32x3(positions),
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_COLOR, colors)
    .with_inserted_indices(Indices::U32(indices));
    mesh.compute_smooth_normals();
    Some(mesh)
}

/// Rebuilds each curl mesh when its fronts have a new readback, spawning it the first time.
pub(super) fn update_breaker_curls(
    mut commands: Commands,
    owners: Query<(Entity, &WaterBreakingFronts, &WaterBreakerCurl)>,
    mut curls: Query<(&mut BreakerCurlMesh, &mut Mesh3d, &mut Visibility)>,
    children: Query<&Children>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    for (owner, fronts, settings) in &owners {
        let existing = children
            .get(owner)
            .ok()
            .and_then(|c| c.iter().find(|e| curls.contains(*e)));
        let mesh = curl_mesh(&fronts.lines, settings);
        match existing {
            Some(entity) => {
                let (mut curl, mut handle, mut visibility) = curls.get_mut(entity).unwrap();
                if curl.frame == fronts.frame {
                    continue;
                }
                curl.frame = fronts.frame;
                match mesh {
                    Some(mesh) => {
                        meshes.remove(&handle.0);
                        handle.0 = meshes.add(mesh);
                        *visibility = Visibility::Inherited;
                    }
                    None => *visibility = Visibility::Hidden,
                }
            }
            None => {
                let Some(mesh) = mesh else {
                    continue;
                };
                let material = materials.add(StandardMaterial {
                    base_color: Color::WHITE,
                    perceptual_roughness: 0.15,
                    reflectance: 0.35,
                    alpha_mode: AlphaMode::Blend,
                    double_sided: true,
                    cull_mode: None,
                    ..default()
                });
                let curl = commands
                    .spawn((
                        BreakerCurlMesh {
                            owner,
                            frame: fronts.frame,
                        },
                        Mesh3d(meshes.add(mesh)),
                        MeshMaterial3d(material),
                        Transform::default(),
                        bevy::light::NotShadowCaster,
                    ))
                    .id();
                commands.entity(owner).add_child(curl);
            }
        }
    }
}

/// Despawns curl meshes whose owner no longer has a [`WaterBreakerCurl`].
pub(super) fn remove_breaker_curls(
    mut commands: Commands,
    curls: Query<(Entity, &BreakerCurlMesh)>,
    owners: Query<(), With<WaterBreakerCurl>>,
) {
    for (entity, curl) in &curls {
        if !owners.contains(curl.owner) {
            commands.entity(entity).despawn();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::water::FrontPoint;

    #[test]
    fn the_profile_starts_and_ends_on_the_water() {
        for t in [0.0, 0.2, 0.4, 0.55, 0.7, 0.9, 1.0] {
            let p = curl_profile(t, PROFILE_SAMPLES);
            assert_eq!(p.len(), PROFILE_SAMPLES);
            assert!(
                p[0].0.y.abs() < 1e-5 && p[PROFILE_SAMPLES - 1].0.y.abs() < 1e-5,
                "{t}"
            );
        }
    }

    #[test]
    fn the_lip_overhangs_while_breaking_and_ends_flat() {
        // An overhang: further along the profile, a point lies behind one before it, above the
        // water, so a vertical line meets the surface more than once.
        let overhangs = |t: f32| {
            let p = curl_profile(t, PROFILE_SAMPLES);
            (1..p.len()).any(|i| p[i].0.x < p[i - 1].0.x - 1e-3 && p[i].0.y > 0.05)
        };
        assert!(!overhangs(0.0));
        assert!(overhangs(0.5));
        let spent = curl_profile(1.0, PROFILE_SAMPLES);
        assert!(spent.iter().all(|(p, _)| p.y < 0.1));
        assert!(spent.last().unwrap().1 > 0.9, "the spent bore is foam");
    }

    #[test]
    fn a_line_sweeps_one_profile_per_point_and_tapers_at_its_ends() {
        let line = FrontLine {
            id: 1,
            age: 0.8,
            closed: false,
            points: (0..10)
                .map(|z| FrontPoint {
                    position: Vec3::new(0.0, 0.0, z as f32),
                    direction: Vec2::X,
                    depth: 2.0,
                    intensity: 1.0,
                    ..default()
                })
                .collect(),
        };
        let settings = WaterBreakerCurl::default();
        let mesh = curl_mesh(std::slice::from_ref(&line), &settings).unwrap();
        let positions = mesh
            .attribute(Mesh::ATTRIBUTE_POSITION)
            .unwrap()
            .as_float3()
            .unwrap();
        assert_eq!(positions.len(), 10 * PROFILE_SAMPLES);
        let top = |row: usize| {
            positions[row * PROFILE_SAMPLES..(row + 1) * PROFILE_SAMPLES]
                .iter()
                .map(|p| p[1])
                .fold(0.0, f32::max)
        };
        assert_eq!(top(0), 0.0, "the end of the line lies on the water");
        // 0.78 of 2 m deep water, near the top of the profile.
        assert!((1.2..1.9).contains(&top(5)), "{}", top(5));
        assert_eq!(mesh.indices().unwrap().len(), 9 * (PROFILE_SAMPLES - 1) * 6);
    }
}
