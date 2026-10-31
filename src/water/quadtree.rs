//! View-dependent water mesh from a quadtree of equal-resolution tiles, after Strugar's CDLOD.
//!
//! Every frame the quadtree is refined around the [`WaterViewer`]. A node is split while the viewer
//! is closer than its level's range, so tile size grows with distance, including distance above
//! the water. Tiles outside the camera frustum are dropped. The surviving tiles go to the GPU as a
//! list, and the mesh is a fixed pool of tile grids that the water vertex shader places from that
//! list. Indirect draws visit only the selected slots. Without indirect drawing, unused slots
//! collapse outside the clip volume.
//!
//! Vertices near the far edge of a tile's range morph onto the grid of the next coarser level, so
//! neighbouring tiles of different levels meet without cracks and levels change without popping.
//! [`WaterQuadtree::lod_range`] is clamped so that this holds for every tile.
//!
//! Selection runs in compute by default. The CPU path remains available as a reference.
//! Both paths use Bevy's material pipeline for shadows, prepasses, and deferred shading.

use bevy::{
    asset::RenderAssetUsages,
    camera::{
        primitives::{Aabb, Frustum},
        visibility::NoFrustumCulling,
    },
    mesh::{Indices, PrimitiveTopology},
    prelude::*,
    render::storage::ShaderBuffer,
    render::{
        batching::NoAutomaticBatching,
        gpu_readback::{Readback, ReadbackComplete},
    },
};

use super::{WaterSurfaceMaterial, WaterSurfaceTag, WaterViewer};

/// Quadtree water mesh settings. Add it with a
/// [`WaterSurfaceMaterial`](super::WaterSurfaceMaterial). The plugin builds the tile pool
/// [`Mesh3d`], tags the entity with [`WaterSurfaceTag`], and refreshes the tile list each frame.
#[derive(Component, Clone, Copy, Debug, Reflect)]
#[require(
    Transform,
    Visibility,
    NoFrustumCulling,
    NoAutomaticBatching,
    WaterQuadtreeStats
)]
pub struct WaterQuadtree {
    /// Selects tiles in compute. Disable to use the CPU reference selector.
    pub gpu_selection: bool,
    /// Edge length of the smallest tiles in meters.
    pub leaf_size: f32,
    /// Grid cells along each tile edge. Rounded to an even number, since morphing pairs cells.
    pub tile_cells: u32,
    /// Number of levels. Root tiles have edge `leaf_size * 2^(levels - 1)`, and a 3×3 block of
    /// them around the viewer sets the reach of the mesh.
    pub levels: u32,
    /// Size of the tile pool. Selection stops adding tiles once the pool is full.
    pub max_tiles: u32,
    /// Viewing distance covered by leaf tiles, as a multiple of `leaf_size`. Each level doubles
    /// it. Values below the crack-free minimum are raised to it.
    pub lod_range: f32,
    /// Share of each level's distance band, from 0 to 1, that stays unmorphed. Higher values
    /// morph over a shorter band near the far edge.
    pub morph_start: f32,
    /// Lowest and highest water elevation in meters, for frustum culling.
    pub height_bounds: Vec2,
    /// Largest horizontal displacement in meters, for frustum culling.
    pub horizontal_margin: f32,
    /// Drops tiles outside the viewer's frustum.
    pub frustum_cull: bool,
}

impl Default for WaterQuadtree {
    fn default() -> Self {
        Self {
            gpu_selection: true,
            leaf_size: 2.0,
            tile_cells: 8,
            levels: 13,
            max_tiles: 1024,
            lod_range: 4.0,
            morph_start: 0.75,
            height_bounds: Vec2::new(-24.0, 24.0),
            horizontal_margin: 12.0,
            frustum_cull: true,
        }
    }
}

/// Most recent completed tile selection, for comparing against other meshes.
/// GPU results arrive asynchronously. Without indirect drawing, unused pool slots still run
/// the vertex shader and collapse outside the clip volume.
#[derive(Component, Clone, Copy, Debug, Default, Reflect)]
pub struct WaterQuadtreeStats {
    /// Tiles selected and inside the frustum.
    pub tiles: u32,
    /// Vertices of the selected tiles.
    pub vertices: u32,
    /// Tiles that did not fit in the pool.
    pub dropped: u32,
}

/// One selected tile.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaterQuadtreeTile {
    /// Corner with the smallest X and Z, in meters.
    pub min: Vec2,
    /// Edge length in meters.
    pub size: f32,
    pub level: u32,
}

impl WaterQuadtree {
    pub(super) fn cells(&self) -> u32 {
        (self.tile_cells.max(2) / 2) * 2
    }

    pub(super) fn morph_start_ratio(&self) -> f32 {
        self.morph_start.clamp(0.05, 0.95)
    }

    /// Edge length of tiles at `level` in meters.
    pub fn tile_size(&self, level: u32) -> f32 {
        self.leaf_size * (1u64 << level) as f32
    }

    /// Smallest `lod_range` for which neighbouring tiles always meet without cracks.
    ///
    /// A tile at level `l` lies inside a split parent, so its far edge is at most the parent's
    /// diagonal past `range(l)`. The coarser neighbour there must not have started morphing yet,
    /// which needs that diagonal to fit inside the unmorphed part of the next band.
    pub fn min_lod_range(&self) -> f32 {
        2.0 * std::f32::consts::SQRT_2 / self.morph_start_ratio() * 1.01
    }

    /// Distance in meters within which tiles at `level` are split into `level - 1`. Tiles at
    /// `level` are fully morphed to the next coarser grid at this distance.
    pub fn range(&self, level: u32) -> f32 {
        self.leaf_size * self.lod_range.max(self.min_lod_range()) * (1u64 << level) as f32
    }

    /// Distances in meters where morphing starts and ends for tiles at `level`.
    pub fn morph_band(&self, level: u32) -> (f32, f32) {
        let end = self.range(level);
        let start = 0.5 * end * (1.0 + self.morph_start_ratio());
        (start, end)
    }

    /// Selects the tiles to draw for a viewer at `eye`, keeping only tiles that `visible` accepts.
    /// Returns the tiles and the number dropped because the pool was full.
    pub fn select(
        &self,
        eye: Vec3,
        mut visible: impl FnMut(&Aabb) -> bool,
    ) -> (Vec<WaterQuadtreeTile>, u32) {
        let mut tiles = Vec::new();
        let mut dropped = 0;
        let top = self.levels.clamp(1, 24) - 1;
        let root = self.tile_size(top);
        let base = (eye.xz() / root).floor();
        for dz in -1..=1 {
            for dx in -1..=1 {
                let min = (base + Vec2::new(dx as f32, dz as f32)) * root;
                self.select_node(eye, min, top, &mut visible, &mut tiles, &mut dropped);
            }
        }
        (tiles, dropped)
    }

    fn select_node(
        &self,
        eye: Vec3,
        min: Vec2,
        level: u32,
        visible: &mut impl FnMut(&Aabb) -> bool,
        tiles: &mut Vec<WaterQuadtreeTile>,
        dropped: &mut u32,
    ) {
        let size = self.tile_size(level);
        if !visible(&self.cull_box(min, size)) {
            return;
        }
        if level > 0 && flat_distance(eye, min, size) < self.range(level - 1) {
            let half = size * 0.5;
            for (ox, oz) in [(0.0, 0.0), (half, 0.0), (0.0, half), (half, half)] {
                self.select_node(
                    eye,
                    min + Vec2::new(ox, oz),
                    level - 1,
                    visible,
                    tiles,
                    dropped,
                );
            }
            return;
        }
        if tiles.len() >= self.max_tiles as usize {
            *dropped += 1;
            return;
        }
        tiles.push(WaterQuadtreeTile { min, size, level });
    }

    fn cull_box(&self, min: Vec2, size: f32) -> Aabb {
        let m = self.horizontal_margin.max(0.0);
        let lo = Vec3::new(min.x - m, self.height_bounds.x, min.y - m);
        let hi = Vec3::new(min.x + size + m, self.height_bounds.y, min.y + size + m);
        Aabb::from_min_max(lo, hi)
    }

    /// Builds the tile pool. Vertex X and Z hold the grid coordinate inside the tile, from 0 to
    /// [`Self::tile_cells`], and Y holds the pool slot.
    pub fn build_mesh(&self) -> Mesh {
        let n = self.cells();
        let row = n + 1;
        let per_tile = row * row;
        let slots = self.max_tiles.max(1);
        let mut positions = Vec::with_capacity((slots * per_tile) as usize);
        let mut indices = Vec::with_capacity((slots * n * n * 6) as usize);
        for slot in 0..slots {
            let base = slot * per_tile;
            for z in 0..row {
                for x in 0..row {
                    positions.push([x as f32, slot as f32, z as f32]);
                }
            }
            for z in 0..n {
                for x in 0..n {
                    let a = base + z * row + x;
                    let (b, c, d) = (a + row, a + 1, a + row + 1);
                    // Both triangles wind so their normal points up.
                    indices.extend([a, b, c, c, b, d]);
                }
            }
        }
        let normals = vec![[0.0, 1.0, 0.0]; positions.len()];
        let uvs: Vec<[f32; 2]> = positions.iter().map(|p| [p[0], p[2]]).collect();
        Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::default(),
        )
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
        .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
        .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, uvs)
        .with_inserted_indices(Indices::U32(indices))
    }

    /// Packs the tile list for `water_quadtree` in `surface.wesl`.
    fn gpu_words(&self, eye: Vec3, tiles: &[WaterQuadtreeTile]) -> Vec<[f32; 4]> {
        let mut words = Vec::with_capacity(2 + tiles.len() * 2);
        words.push([eye.x, eye.y, eye.z, 0.0]);
        words.push([
            f32::from_bits(tiles.len() as u32),
            f32::from_bits(self.cells()),
            f32::from_bits(1),
            0.0,
        ]);
        for tile in tiles {
            let (start, end) = self.morph_band(tile.level);
            words.push([tile.min.x, tile.min.y, tile.size, tile.level as f32]);
            words.push([start, end, 0.0, 0.0]);
        }
        words
    }
}

/// Distance from `eye` to the square at sea level with corner `min` and edge `size`.
fn flat_distance(eye: Vec3, min: Vec2, size: f32) -> f32 {
    let nearest = eye.xz().clamp(min, min + Vec2::splat(size));
    Vec3::new(nearest.x, 0.0, nearest.y).distance(eye)
}

/// Morph factor for a vertex of a tile at `level`, as the surface shader computes it.
#[cfg(test)]
fn morph_factor(q: &WaterQuadtree, eye: Vec3, xz: Vec2, level: u32) -> f32 {
    let (start, end) = q.morph_band(level);
    let d = Vec3::new(xz.x, 0.0, xz.y).distance(eye);
    ((d - start) / (end - start)).clamp(0.0, 1.0)
}

/// GPU tile list for the quadtree entity. Materials without a quadtree bind
/// [`WaterQuadtreeFallback`] instead.
#[derive(Component, Clone)]
pub struct WaterQuadtreeBuffer(pub Handle<ShaderBuffer>);

/// Tile list bound by water surface materials that draw an ordinary mesh. Its header marks the
/// quadtree as off.
#[derive(Resource, Clone)]
pub struct WaterQuadtreeFallback(pub Handle<ShaderBuffer>);

pub(super) fn create_water_quadtree_fallback(
    mut commands: Commands,
    mut buffers: ResMut<Assets<ShaderBuffer>>,
) {
    let words: Vec<[f32; 4]> = vec![[0.0; 4]; 4];
    commands.insert_resource(WaterQuadtreeFallback(
        buffers.add(ShaderBuffer::new(words, RenderAssetUsages::default())),
    ));
}

/// Builds the tile pool for new or changed quadtrees.
pub(super) fn build_water_quadtrees(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut buffers: ResMut<Assets<ShaderBuffer>>,
    quadtrees: Query<
        (
            Entity,
            &WaterQuadtree,
            Option<&WaterQuadtreeBuffer>,
            Option<&MeshMaterial3d<WaterSurfaceMaterial>>,
        ),
        Changed<WaterQuadtree>,
    >,
) {
    for (entity, quadtree, buffer, material) in &quadtrees {
        let mut entity = commands.entity(entity);
        entity.insert((Mesh3d(meshes.add(quadtree.build_mesh())), WaterSurfaceTag));
        // Re-inserting the material marks it changed, so the renderer picks it up again for an
        // entity whose mesh was swapped out.
        if let Some(material) = material {
            entity.insert(material.clone());
        }
        let mut words = quadtree.gpu_words(Vec3::ZERO, &[]);
        words.resize(2 + quadtree.max_tiles.max(1) as usize * 2, [0.0; 4]);
        let handle = buffer.map(|b| b.0.clone()).unwrap_or_else(|| {
            buffers.add(ShaderBuffer::new(
                words.clone(),
                RenderAssetUsages::default(),
            ))
        });
        buffers
            .insert(
                handle.id(),
                ShaderBuffer::new(words, RenderAssetUsages::default()),
            )
            .unwrap();
        entity.insert(WaterQuadtreeBuffer(handle.clone()));
        if quadtree.gpu_selection {
            entity.insert(Readback::buffer_range(handle, 16, 16));
        } else {
            entity.remove::<Readback>();
        }
    }
}

/// Selects tiles for every quadtree around the viewer and uploads the lists.
#[allow(clippy::type_complexity)]
pub(super) fn select_water_quadtrees(
    viewers: Query<(&GlobalTransform, Option<&Frustum>), With<WaterViewer>>,
    cameras: Query<(&GlobalTransform, Option<&Frustum>), With<Camera3d>>,
    mut quadtrees: Query<(
        &WaterQuadtree,
        &WaterQuadtreeBuffer,
        &Mesh3d,
        &mut WaterQuadtreeStats,
    )>,
    mut buffers: ResMut<Assets<ShaderBuffer>>,
    mut frame: ResMut<super::quadtree_gpu::WaterQuadtreeFrame>,
) {
    frame.0.clear();
    let Some((camera, frustum)) = viewers.iter().next().or_else(|| cameras.iter().next()) else {
        return;
    };
    let eye = camera.translation();
    for (quadtree, buffer, mesh, mut stats) in &mut quadtrees {
        if quadtree.gpu_selection {
            frame.0.push(super::quadtree_gpu::FrameEntry::new(
                *quadtree,
                eye,
                frustum,
                buffer.0.clone(),
                mesh.0.id(),
            ));
            continue;
        }
        let (tiles, dropped) = quadtree.select(eye, |aabb| match frustum {
            Some(f) if quadtree.frustum_cull => f.intersects_obb_identity(aabb),
            _ => true,
        });
        let row = quadtree.cells() + 1;
        *stats = WaterQuadtreeStats {
            tiles: tiles.len() as u32,
            vertices: tiles.len() as u32 * row * row,
            dropped,
        };
        if let Some(mut gpu) = buffers.get_mut(&buffer.0) {
            *gpu = ShaderBuffer::new(
                quadtree.gpu_words(eye, &tiles),
                RenderAssetUsages::default(),
            );
        }
    }
}

pub(super) fn read_quadtree_stats(
    event: On<ReadbackComplete>,
    mut quadtrees: Query<(&WaterQuadtree, &mut WaterQuadtreeStats)>,
) {
    let Ok((q, mut stats)) = quadtrees.get_mut(event.entity) else {
        return;
    };
    if !q.gpu_selection || event.data.len() < 16 {
        return;
    }
    let word = |i: usize| u32::from_le_bytes(event.data[i..i + 4].try_into().unwrap());
    let tiles = word(0).min(q.max_tiles);
    *stats = WaterQuadtreeStats {
        tiles,
        vertices: tiles * (q.cells() + 1).pow(2),
        dropped: word(12),
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{HashMap, HashSet};

    fn select_all(q: &WaterQuadtree, eye: Vec3) -> Vec<WaterQuadtreeTile> {
        let (tiles, dropped) = q.select(eye, |_| true);
        assert_eq!(dropped, 0);
        tiles
    }

    fn eyes() -> Vec<Vec3> {
        let mut out = Vec::new();
        for (i, h) in [1.5f32, 7.0, 40.0, 300.0].iter().enumerate() {
            for j in 0..5 {
                let t = (i * 5 + j) as f32;
                out.push(Vec3::new(t * 13.37 - 40.0, *h, 21.1 - t * 7.9));
            }
        }
        out
    }

    /// Tiles cover each root without overlap.
    #[test]
    fn tiles_tile_the_roots() {
        let q = WaterQuadtree {
            max_tiles: 100_000,
            levels: 8,
            ..default()
        };
        for eye in eyes() {
            let tiles = select_all(&q, eye);
            let area: f32 = tiles.iter().map(|t| t.size * t.size).sum();
            let root = q.tile_size(7);
            assert!((area - 9.0 * root * root).abs() < 1e-3 * area);
        }
    }

    /// Grid point in leaf-cell units.
    type Point = (i64, i64);
    /// A tile edge: the grid line it lies on, its start and end along that line, and whether the
    /// line runs along X.
    type Edge = (i64, i64, i64, bool);

    /// Morphed vertex positions along each shared tile edge must be the same set seen from both
    /// sides, or the surface tears once heights are applied.
    #[test]
    fn neighbouring_tiles_meet_without_cracks() {
        let q = WaterQuadtree {
            max_tiles: 100_000,
            levels: 9,
            ..default()
        };
        let n = q.cells() as i32;
        let unit = q.leaf_size / n as f32;
        for eye in eyes() {
            let tiles = select_all(&q, eye);
            // Morphed edge vertices keyed by the edge segment they lie on, in leaf-grid units.
            let mut edges: HashMap<Edge, HashSet<Point>> = HashMap::new();
            for t in &tiles {
                let step = (t.size / n as f32 / unit).round() as i64;
                let ox = (t.min.x / unit).round() as i64;
                let oz = (t.min.y / unit).round() as i64;
                let len = step * n as i64;
                for side in 0..4 {
                    for i in 0..=n {
                        let (gx, gz) = match side {
                            0 => (i, 0),
                            1 => (i, n),
                            2 => (0, i),
                            _ => (n, i),
                        };
                        let xz = t.min + Vec2::new(gx as f32, gz as f32) * t.size / n as f32;
                        let k = morph_factor(&q, eye, xz, t.level);
                        let g = Vec2::new(gx as f32, gz as f32);
                        let odd = (g * 0.5).fract() * 2.0;
                        let m = g - odd * k;
                        let p = (
                            ((ox as f32) + m.x * step as f32).round() as i64,
                            ((oz as f32) + m.y * step as f32).round() as i64,
                        );
                        // Key each edge by the leaf-grid line it lies on.
                        let key = match side {
                            0 => (oz, ox, ox + len, true),
                            1 => (oz + len, ox, ox + len, true),
                            2 => (ox, oz, oz + len, false),
                            _ => (ox + len, oz, oz + len, false),
                        };
                        edges.entry(key).or_default().insert(p);
                    }
                }
            }
            // Gather vertices per line and compare where segments of two tiles overlap.
            let mut lines: HashMap<(i64, bool), Vec<(Point, &HashSet<Point>)>> = HashMap::new();
            for ((line, a, b, horizontal), set) in &edges {
                lines
                    .entry((*line, *horizontal))
                    .or_default()
                    .push(((*a, *b), set));
            }
            for ((_, horizontal), segs) in &lines {
                for (i, (ra, sa)) in segs.iter().enumerate() {
                    for (rb, sb) in segs.iter().skip(i + 1) {
                        let lo = ra.0.max(rb.0);
                        let hi = ra.1.min(rb.1);
                        if lo >= hi {
                            continue;
                        }
                        let along = |p: &(i64, i64)| if *horizontal { p.0 } else { p.1 };
                        let clip = |s: &HashSet<(i64, i64)>| -> HashSet<(i64, i64)> {
                            s.iter()
                                .filter(|p| (lo..=hi).contains(&along(p)))
                                .copied()
                                .collect()
                        };
                        assert_eq!(clip(sa), clip(sb), "crack near eye {eye}");
                    }
                }
            }
        }
    }

    #[test]
    fn far_tiles_are_coarse_and_counts_stay_bounded() {
        let q = WaterQuadtree::default();
        let (tiles, dropped) = q.select(Vec3::new(3.0, 2.0, -5.0), |_| true);
        assert_eq!(dropped, 0);
        assert!(tiles.len() < 1024, "{} tiles", tiles.len());
        let finest = tiles.iter().map(|t| t.level).min().unwrap();
        assert_eq!(finest, 0);
        let high = q.select(Vec3::new(3.0, 500.0, -5.0), |_| true).0;
        assert!(high.iter().map(|t| t.level).min().unwrap() > 3);
    }

    #[test]
    fn pool_mesh_faces_up() {
        let q = WaterQuadtree {
            max_tiles: 2,
            ..default()
        };
        let mesh = q.build_mesh();
        let Some(bevy::mesh::VertexAttributeValues::Float32x3(p)) =
            mesh.attribute(Mesh::ATTRIBUTE_POSITION)
        else {
            panic!("positions");
        };
        let Some(Indices::U32(idx)) = mesh.indices() else {
            panic!("indices");
        };
        for t in idx.chunks(3) {
            let a = Vec3::from(p[t[0] as usize]);
            let b = Vec3::from(p[t[1] as usize]);
            let c = Vec3::from(p[t[2] as usize]);
            assert_eq!(a.y, b.y);
            let (a, b, c) = (a.with_y(0.0), b.with_y(0.0), c.with_y(0.0));
            assert!((b - a).cross(c - a).y > 0.0);
        }
    }
}
