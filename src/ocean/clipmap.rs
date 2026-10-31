//! Camera-centred LOD mesh for the ocean surface.
//!
//! The mesh follows the [`WaterViewer`] camera, or the first [`Camera3d`] when none is marked.
//!
//! A single plane cannot cover a large level with enough detail near the viewer. [`OceanClipmap`]
//! builds concentric square rings whose vertex spacing doubles with each level, then keeps the
//! mesh under the camera. The surface material samples the ocean by world position, so the waves
//! stay anchored while the mesh moves.
//!
//! The mesh moves in steps of [`OceanClipmap::snap_spacing`]. The surface shader then shifts each
//! ring onto that ring's own world lattice, so a move smaller than the ring spacing leaves the
//! ring where it was. Adjacent rings do not share edge vertices. The outer ring starts one cell
//! outside the inner one, and a skirt fills that gutter. The gutter stays open while the camera
//! moves, so the rings never fold across each other.

use std::collections::HashMap;

use bevy::{
    asset::RenderAssetUsages,
    mesh::{Indices, PrimitiveTopology},
    prelude::*,
};

use crate::water::{WaterSurfaceMaterial, WaterSurfaceTag, WaterViewer};

/// Camera-following LOD mesh settings. Add it with a
/// [`WaterSurfaceMaterial`](crate::water::WaterSurfaceMaterial). The plugin builds the [`Mesh3d`],
/// tags it with [`WaterSurfaceTag`], and moves the entity each frame.
#[derive(Component, Clone, Copy, Debug, Reflect)]
#[require(Transform, Visibility)]
pub struct OceanClipmap {
    /// Vertex spacing of the innermost level in meters.
    pub base_spacing: f32,
    /// Half-width of the innermost level in cells. Even, and at least 4. Smaller values are
    /// raised to 4. Each ring is half this many cells wide at its own spacing.
    pub cells_per_level: u32,
    /// Number of levels including the innermost square.
    pub levels: u32,
    /// Level whose spacing sets the move step. Levels up to this one never swim.
    pub snap_level: u32,
}

impl Default for OceanClipmap {
    fn default() -> Self {
        Self {
            base_spacing: 0.25,
            cells_per_level: 64,
            levels: 9,
            snap_level: 2,
        }
    }
}

impl OceanClipmap {
    /// Distance in meters the mesh moves at a time.
    pub fn snap_spacing(&self) -> f32 {
        self.base_spacing * (1u32 << self.snap_level.min(self.levels.saturating_sub(1))) as f32
    }

    fn cells(&self) -> u32 {
        self.cells_per_level.max(4) / 2 * 2
    }

    /// Half-width in meters of the whole mesh.
    pub fn extent(&self) -> f32 {
        self.base_spacing * self.cells() as f32 * (1u32 << self.levels.saturating_sub(1)) as f32
    }

    /// Builds the ring mesh centred on the origin in the XZ plane.
    pub fn build_mesh(&self) -> Mesh {
        let n = self.cells() as i64;
        let mut builder = MeshBuilder::default();

        // Integer grid units are base-spacing cells, so every level shares exact coordinates.
        builder.grid_level(n, 1);
        for level in 1..self.levels.max(1) {
            builder.ring_level(level, n, 1i64 << level);
        }
        builder.into_mesh(self.base_spacing)
    }
}

#[derive(Default)]
struct MeshBuilder {
    /// Level, then integer X and Z. Levels keep their own copies of a shared edge.
    lookup: HashMap<(u32, i64, i64), u32>,
    positions: Vec<[f32; 3]>,
    /// Ring spacing in base cells. Every vertex of a ring carries that ring's spacing.
    spacing: Vec<i64>,
    indices: Vec<u32>,
}

impl MeshBuilder {
    fn vertex(&mut self, level: u32, x: i64, z: i64, spacing: i64) -> u32 {
        if let Some(&id) = self.lookup.get(&(level, x, z)) {
            return id;
        }
        let id = self.positions.len() as u32;
        self.lookup.insert((level, x, z), id);
        self.positions.push([x as f32, 0.0, z as f32]);
        self.spacing.push(spacing);
        id
    }

    fn must(&self, level: u32, x: i64, z: i64) -> u32 {
        self.lookup[&(level, x, z)]
    }

    /// Adds a triangle wound so its normal points up (+Y).
    fn tri(&mut self, level: u32, a: (i64, i64), b: (i64, i64), c: (i64, i64), spacing: i64) {
        let cross_y = (b.1 - a.1) * (c.0 - a.0) - (b.0 - a.0) * (c.1 - a.1);
        let (b, c) = if cross_y >= 0 { (b, c) } else { (c, b) };
        let ia = self.vertex(level, a.0, a.1, spacing);
        let ib = self.vertex(level, b.0, b.1, spacing);
        let ic = self.vertex(level, c.0, c.1, spacing);
        self.indices.extend([ia, ib, ic]);
    }

    fn quad(&mut self, level: u32, x0: i64, z0: i64, s: i64) {
        let (x1, z1) = (x0 + s, z0 + s);
        self.tri(level, (x0, z0), (x0, z1), (x1, z0), s);
        self.tri(level, (x1, z0), (x0, z1), (x1, z1), s);
    }

    /// Full square of `2n × 2n` cells with spacing `s`, centred on the origin.
    fn grid_level(&mut self, n: i64, s: i64) {
        for j in 0..2 * n {
            for i in 0..2 * n {
                self.quad(0, -n * s + i * s, -n * s + j * s, s);
            }
        }
    }

    /// Square ring of level `level` with spacing `s` around the previous level.
    ///
    /// The hole is one cell larger than the previous level. The two edges can each slide by
    /// half a cell and still not cross. [`Self::stitch`] fills the gutter.
    fn ring_level(&mut self, level: u32, n: i64, s: i64) {
        let half = n / 2;
        let inner0 = half - 1;
        let inner1 = 2 * n - half + 1;
        for j in 0..2 * n {
            for i in 0..2 * n {
                if (inner0..inner1).contains(&i) && (inner0..inner1).contains(&j) {
                    continue;
                }
                let x0 = -n * s + i * s;
                let z0 = -n * s + j * s;
                self.quad(level, x0, z0, s);
            }
        }
        self.stitch(level - 1, level, n, s);
    }

    /// Fills the one-cell gutter between the previous level's outer edge and this ring.
    fn stitch(&mut self, fine: u32, coarse: u32, n: i64, s: i64) {
        let edge = n * s / 2;
        let outer = edge + s;
        self.span_x(fine, coarse, -edge, edge, -edge, -outer, s);
        self.span_x(fine, coarse, -edge, edge, edge, outer, s);
        self.span_z(fine, coarse, -edge, -outer, -edge, edge, s);
        self.span_z(fine, coarse, edge, outer, -edge, edge, s);
        for (cx, cz) in [(edge, edge), (edge, -edge), (-edge, edge), (-edge, -edge)] {
            self.corner(fine, coarse, cx, cz, s);
        }
    }

    /// One row of the gutter along constant Z, from `x0` to `x1` in coarse steps.
    fn span_x(
        &mut self,
        fine: u32,
        coarse: u32,
        x0: i64,
        x1: i64,
        z_fine: i64,
        z_coarse: i64,
        s: i64,
    ) {
        let m = s / 2;
        let mut x = x0;
        while x < x1 {
            self.face(
                fine,
                (x, z_fine),
                fine,
                (x + m, z_fine),
                coarse,
                (x, z_coarse),
            );
            self.face(
                fine,
                (x + m, z_fine),
                coarse,
                (x, z_coarse),
                coarse,
                (x + s, z_coarse),
            );
            self.face(
                fine,
                (x + m, z_fine),
                coarse,
                (x + s, z_coarse),
                fine,
                (x + s, z_fine),
            );
            x += s;
        }
    }

    /// One row of the gutter along constant X, from `z0` to `z1` in coarse steps.
    fn span_z(
        &mut self,
        fine: u32,
        coarse: u32,
        x_fine: i64,
        x_coarse: i64,
        z0: i64,
        z1: i64,
        s: i64,
    ) {
        let m = s / 2;
        let mut z = z0;
        while z < z1 {
            self.face(
                fine,
                (x_fine, z),
                fine,
                (x_fine, z + m),
                coarse,
                (x_coarse, z),
            );
            self.face(
                fine,
                (x_fine, z + m),
                coarse,
                (x_coarse, z),
                coarse,
                (x_coarse, z + s),
            );
            self.face(
                fine,
                (x_fine, z + m),
                coarse,
                (x_coarse, z + s),
                fine,
                (x_fine, z + s),
            );
            z += s;
        }
    }

    /// The square where two gutter rows meet, outside the previous level's corner.
    fn corner(&mut self, fine: u32, coarse: u32, cx: i64, cz: i64, s: i64) {
        let ox = cx.signum() * s;
        let oz = cz.signum() * s;
        self.face(
            fine,
            (cx, cz),
            coarse,
            (cx, cz + oz),
            coarse,
            (cx + ox, cz + oz),
        );
        self.face(
            fine,
            (cx, cz),
            coarse,
            (cx + ox, cz + oz),
            coarse,
            (cx + ox, cz),
        );
    }

    /// One triangle, wound so its local normal points up.
    fn face(&mut self, la: u32, a: (i64, i64), lb: u32, b: (i64, i64), lc: u32, c: (i64, i64)) {
        let cross_y = (b.1 - a.1) * (c.0 - a.0) - (b.0 - a.0) * (c.1 - a.1);
        let ((lb, b), (lc, c)) = if cross_y >= 0 {
            ((lb, b), (lc, c))
        } else {
            ((lc, c), (lb, b))
        };
        let ia = self.must(la, a.0, a.1);
        let ib = self.must(lb, b.0, b.1);
        let ic = self.must(lc, c.0, c.1);
        self.indices.extend([ia, ib, ic]);
    }

    fn into_mesh(self, spacing: f32) -> Mesh {
        let positions: Vec<[f32; 3]> = self
            .positions
            .iter()
            .map(|p| [p[0] * spacing, 0.0, p[2] * spacing])
            .collect();
        let normals = vec![[0.0, 1.0, 0.0]; positions.len()];
        let uvs: Vec<[f32; 2]> = positions.iter().map(|p| [p[0], p[2]]).collect();
        // World spacing of this vertex's ring. The surface shader shifts the whole ring onto
        // that lattice.
        let spacing: Vec<[f32; 2]> = self
            .spacing
            .iter()
            .map(|&cells| [cells as f32 * spacing, 0.0])
            .collect();
        Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::default(),
        )
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
        .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
        .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, uvs)
        .with_inserted_attribute(Mesh::ATTRIBUTE_UV_1, spacing)
        .with_inserted_indices(Indices::U32(self.indices))
    }
}

/// Builds the mesh for new or changed clipmaps.
pub(super) fn build_ocean_clipmaps(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    clipmaps: Query<
        (
            Entity,
            &OceanClipmap,
            Option<&MeshMaterial3d<WaterSurfaceMaterial>>,
        ),
        Changed<OceanClipmap>,
    >,
) {
    for (entity, clipmap, material) in &clipmaps {
        let mut entity = commands.entity(entity);
        entity.insert((Mesh3d(meshes.add(clipmap.build_mesh())), WaterSurfaceTag));
        // Re-inserting the material marks it changed, so the renderer picks it up again for an
        // entity whose mesh was swapped out.
        if let Some(material) = material {
            entity.insert(material.clone());
        }
    }
}

/// Moves each clipmap under its camera in whole snap steps.
///
/// Propagation has already copied this frame's camera position into [`GlobalTransform`].
/// The clipmap global is updated here as well, so the mesh lines up with the water LOD
/// slices in the same frame.
pub(super) fn follow_ocean_clipmaps(
    targets: Query<&GlobalTransform, (With<WaterViewer>, Without<OceanClipmap>)>,
    cameras: Query<&GlobalTransform, (With<Camera3d>, Without<OceanClipmap>)>,
    mut clipmaps: Query<(Entity, &OceanClipmap, &mut Transform, &mut GlobalTransform)>,
    child_of: Query<&ChildOf>,
    ancestors: Query<&GlobalTransform, Without<OceanClipmap>>,
) {
    let Some(camera) = targets.iter().next().or_else(|| cameras.iter().next()) else {
        return;
    };
    let eye = camera.translation();
    for (entity, clipmap, mut transform, mut global) in &mut clipmaps {
        let step = clipmap.snap_spacing().max(f32::EPSILON);
        let x = (eye.x / step).round() * step;
        let z = (eye.z / step).round() * step;
        if transform.translation.x == x && transform.translation.z == z {
            continue;
        }
        transform.translation.x = x;
        transform.translation.z = z;
        let local = *transform;
        *global = match child_of.get(entity) {
            Ok(child_of) => ancestors
                .get(child_of.parent())
                .map(|parent| parent.mul_transform(local))
                .unwrap_or_else(|_| GlobalTransform::from(local)),
            Err(_) => GlobalTransform::from(local),
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::mesh::VertexAttributeValues;

    fn grid_mesh(levels: u32) -> (Vec<[f32; 3]>, Vec<u32>) {
        let clipmap = OceanClipmap {
            base_spacing: 1.0,
            cells_per_level: 8,
            levels,
            snap_level: 0,
        };
        let mesh = clipmap.build_mesh();
        let Some(VertexAttributeValues::Float32x3(p)) = mesh.attribute(Mesh::ATTRIBUTE_POSITION)
        else {
            panic!("positions");
        };
        let Some(Indices::U32(i)) = mesh.indices() else {
            panic!("indices");
        };
        (p.clone(), i.clone())
    }

    /// Every interior edge is shared by two triangles. The outer border is used once.
    /// A crack or a T-junction leaves an inner edge used once.
    #[test]
    fn rings_have_no_t_junctions() {
        let (p, idx) = grid_mesh(4);
        let mut edges: HashMap<(u32, u32), u32> = HashMap::new();
        for t in idx.chunks(3) {
            for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                *edges.entry((a.min(b), a.max(b))).or_default() += 1;
            }
        }
        let extent = 8.0 * 8.0;
        for ((a, b), count) in edges {
            let pa = p[a as usize];
            let pb = p[b as usize];
            let on_border = (pa[0] == pb[0] && pa[0].abs() == extent)
                || (pa[2] == pb[2] && pa[2].abs() == extent);
            if on_border {
                assert_eq!(count, 1, "outer edge {pa:?}-{pb:?}");
            } else {
                assert_eq!(count, 2, "inner edge {pa:?}-{pb:?}");
            }
        }
    }

    #[test]
    fn triangles_face_up() {
        let (p, idx) = grid_mesh(3);
        for t in idx.chunks(3) {
            let a = Vec3::from(p[t[0] as usize]);
            let b = Vec3::from(p[t[1] as usize]);
            let c = Vec3::from(p[t[2] as usize]);
            assert!((b - a).cross(c - a).y > 0.0);
        }
    }

    #[test]
    fn extent_and_snap_follow_levels() {
        let c = OceanClipmap::default();
        assert_eq!(c.extent(), 0.25 * 64.0 * 256.0);
        assert_eq!(c.snap_spacing(), 1.0);
    }

    #[test]
    fn inner_and_outer_rings_keep_their_own_spacing() {
        let clipmap = OceanClipmap {
            base_spacing: 1.0,
            cells_per_level: 8,
            levels: 3,
            snap_level: 0,
        };
        let mesh = clipmap.build_mesh();
        let Some(VertexAttributeValues::Float32x3(positions)) =
            mesh.attribute(Mesh::ATTRIBUTE_POSITION)
        else {
            panic!("positions");
        };
        let Some(VertexAttributeValues::Float32x2(spacing)) = mesh.attribute(Mesh::ATTRIBUTE_UV_1)
        else {
            panic!("spacing");
        };
        let mut fine = None;
        let mut outer = None;
        for (p, s) in positions.iter().zip(spacing) {
            if p[0] == 1.0 && p[2] == 0.0 {
                fine = Some(s[0]);
            }
            if p[0] == clipmap.extent() && p[2] == clipmap.extent() {
                outer = Some(s[0]);
            }
        }
        assert_eq!(fine, Some(1.0));
        assert_eq!(outer, Some(4.0));
    }

    #[test]
    fn each_ring_keeps_its_shape_when_the_origin_moves() {
        let clipmap = OceanClipmap {
            base_spacing: 1.0,
            cells_per_level: 8,
            levels: 3,
            snap_level: 0,
        };
        let mesh = clipmap.build_mesh();
        let Some(VertexAttributeValues::Float32x3(positions)) =
            mesh.attribute(Mesh::ATTRIBUTE_POSITION)
        else {
            panic!("positions");
        };
        let Some(VertexAttributeValues::Float32x2(spacing)) = mesh.attribute(Mesh::ATTRIBUTE_UV_1)
        else {
            panic!("spacing");
        };
        let Some(Indices::U32(indices)) = mesh.indices() else {
            panic!("indices");
        };
        for origin in [0.0_f32, 1.0, 3.0, 5.0] {
            for t in indices.chunks(3) {
                let s = spacing[t[0] as usize][0];
                if spacing[t[1] as usize][0] != s || spacing[t[2] as usize][0] != s {
                    continue;
                }
                let center = (origin / s).round() * s;
                let world = |i: u32| {
                    let p = positions[i as usize];
                    Vec3::new(center + p[0], 0.0, center + p[2])
                };
                let (a, b, c) = (world(t[0]), world(t[1]), world(t[2]));
                let (la, lb, lc) = (
                    Vec3::from(positions[t[0] as usize]),
                    Vec3::from(positions[t[1] as usize]),
                    Vec3::from(positions[t[2] as usize]),
                );
                assert!(((b - a).length() - (lb - la).length()).abs() < 1e-4);
                assert!(((c - a).length() - (lc - la).length()).abs() < 1e-4);
            }
        }
    }

    #[test]
    fn skirt_stays_up_and_does_not_cross_itself() {
        let clipmap = OceanClipmap {
            base_spacing: 1.0,
            cells_per_level: 8,
            levels: 3,
            snap_level: 0,
        };
        let mesh = clipmap.build_mesh();
        let Some(VertexAttributeValues::Float32x3(positions)) =
            mesh.attribute(Mesh::ATTRIBUTE_POSITION)
        else {
            panic!("positions");
        };
        let Some(VertexAttributeValues::Float32x2(spacing)) = mesh.attribute(Mesh::ATTRIBUTE_UV_1)
        else {
            panic!("spacing");
        };
        let Some(Indices::U32(indices)) = mesh.indices() else {
            panic!("indices");
        };

        let mut tris = Vec::new();
        for t in indices.chunks(3) {
            let s = [
                spacing[t[0] as usize][0],
                spacing[t[1] as usize][0],
                spacing[t[2] as usize][0],
            ];
            tris.push(([t[0], t[1], t[2]], s[0] != s[1] || s[1] != s[2]));
        }
        assert!(tris.iter().any(|(_, seam)| *seam));

        let place = |i: u32, origin: f32| {
            let p = positions[i as usize];
            let step = spacing[i as usize][0];
            let parked = (origin / step).round() * step;
            Vec2::new(p[0] + parked, p[2] + parked)
        };
        let orient =
            |a: Vec2, b: Vec2, c: Vec2| (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
        let crosses = |a: Vec2, b: Vec2, c: Vec2, d: Vec2| {
            let o1 = orient(a, b, c);
            let o2 = orient(a, b, d);
            let o3 = orient(c, d, a);
            let o4 = orient(c, d, b);
            o1 * o2 < -1e-3 && o3 * o4 < -1e-3
        };

        for origin in [0.0_f32, 0.5, 1.0, 1.5, 3.0] {
            let world: Vec<[Vec2; 3]> = tris
                .iter()
                .map(|(t, _)| {
                    [
                        place(t[0], origin),
                        place(t[1], origin),
                        place(t[2], origin),
                    ]
                })
                .collect();
            for tri in &world {
                let y = (Vec3::new(tri[1].x, 0.0, tri[1].y) - Vec3::new(tri[0].x, 0.0, tri[0].y))
                    .cross(Vec3::new(tri[2].x, 0.0, tri[2].y) - Vec3::new(tri[0].x, 0.0, tri[0].y))
                    .y;
                assert!(y > 1e-3, "triangle flipped at origin {origin}: {tri:?}");
            }
            for i in 0..world.len() {
                if !tris[i].1 {
                    continue;
                }
                for j in (i + 1)..world.len() {
                    let (a, b) = (&world[i], &world[j]);
                    let hit = [(0, 1), (1, 2), (2, 0)].iter().any(|&(u, v)| {
                        [(0, 1), (1, 2), (2, 0)]
                            .iter()
                            .any(|&(p, q)| crosses(a[u], a[v], b[p], b[q]))
                    });
                    assert!(!hit, "triangles cross at origin {origin}: {a:?} vs {b:?}");
                }
            }
        }
    }
}
