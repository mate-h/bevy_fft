//! Camera-centred LOD mesh for the ocean surface.
//!
//! A single plane cannot cover a large level with enough detail near the viewer. [`OceanClipmap`]
//! builds concentric square rings whose vertex spacing doubles with each level, then keeps the
//! mesh under the camera. The surface material samples the ocean by world position, so the waves
//! stay anchored while the mesh moves.
//!
//! The mesh moves in steps of [`OceanClipmap::snap_spacing`]. Levels at or finer than that step
//! keep the same world vertex positions after a move, so they never swim. Coarser levels shift
//! inside their cells, and the vertex shader fades out waves too short for their spacing (see
//! [`crate::ocean::OceanMaterialUniform::lod_base_spacing`]), which hides that shift.

use std::collections::HashMap;

use bevy::{
    asset::RenderAssetUsages,
    mesh::{Indices, PrimitiveTopology},
    prelude::*,
};

use crate::ocean::{OceanSurfaceMaterial, OceanSurfaceTag};

/// Camera-following LOD mesh settings. Add it with an [`OceanSurfaceMaterial`] and
/// [`OceanSurfaceTag`]; the plugin builds the [`Mesh3d`] and moves the entity each frame.
#[derive(Component, Clone, Copy, Debug, Reflect)]
#[require(Transform, Visibility)]
pub struct OceanClipmap {
    /// Vertex spacing of the innermost level in meters.
    pub base_spacing: f32,
    /// Half-width of the innermost level in cells. Must be even. Each ring is half this many
    /// cells wide at its own spacing.
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

    /// Half-width in meters of the whole mesh.
    pub fn extent(&self) -> f32 {
        self.base_spacing
            * self.cells_per_level as f32
            * (1u32 << self.levels.saturating_sub(1)) as f32
    }

    /// Builds the ring mesh centred on the origin in the XZ plane.
    pub fn build_mesh(&self) -> Mesh {
        let n = (self.cells_per_level.max(2) / 2 * 2) as i64;
        let mut builder = MeshBuilder::default();

        // Integer grid units are base-spacing cells, so every level shares exact coordinates.
        builder.grid_level(n, 1);
        for level in 1..self.levels.max(1) {
            builder.ring_level(n, 1i64 << level);
        }
        builder.into_mesh(self.base_spacing)
    }
}

/// Picks which camera an [`OceanClipmap`] follows. Without it, the first [`Camera3d`] is used.
#[derive(Component, Clone, Copy, Debug, Default, Reflect)]
pub struct OceanClipmapTarget;

#[derive(Default)]
struct MeshBuilder {
    lookup: HashMap<(i64, i64), u32>,
    positions: Vec<[f32; 3]>,
    indices: Vec<u32>,
}

impl MeshBuilder {
    fn vertex(&mut self, x: i64, z: i64) -> u32 {
        let next = self.positions.len() as u32;
        *self.lookup.entry((x, z)).or_insert_with(|| {
            self.positions.push([x as f32, 0.0, z as f32]);
            next
        })
    }

    /// Adds a triangle wound so its normal points up (+Y).
    fn tri(&mut self, a: (i64, i64), b: (i64, i64), c: (i64, i64)) {
        let cross_y = (b.1 - a.1) * (c.0 - a.0) - (b.0 - a.0) * (c.1 - a.1);
        let (b, c) = if cross_y >= 0 { (b, c) } else { (c, b) };
        let ia = self.vertex(a.0, a.1);
        let ib = self.vertex(b.0, b.1);
        let ic = self.vertex(c.0, c.1);
        self.indices.extend([ia, ib, ic]);
    }

    fn quad(&mut self, x0: i64, z0: i64, s: i64) {
        let (x1, z1) = (x0 + s, z0 + s);
        self.tri((x0, z0), (x0, z1), (x1, z0));
        self.tri((x1, z0), (x0, z1), (x1, z1));
    }

    /// Full square of `2n × 2n` cells with spacing `s`, centred on the origin.
    fn grid_level(&mut self, n: i64, s: i64) {
        for j in 0..2 * n {
            for i in 0..2 * n {
                self.quad(-n * s + i * s, -n * s + j * s, s);
            }
        }
    }

    /// Square ring with spacing `s` around the previous level, which has spacing `s / 2`.
    ///
    /// Quads touching the inner square along an edge are split at that edge's midpoint, which is
    /// a vertex of the finer level, so the two levels meet without T-junctions.
    fn ring_level(&mut self, n: i64, s: i64) {
        let half = n / 2;
        for j in 0..2 * n {
            for i in 0..2 * n {
                let inside_i = (half..3 * half).contains(&i);
                let inside_j = (half..3 * half).contains(&j);
                if inside_i && inside_j {
                    continue;
                }
                let x0 = -n * s + i * s;
                let z0 = -n * s + j * s;
                let (x1, z1) = (x0 + s, z0 + s);
                let m = s / 2;
                // Which edge, if any, lies on the inner square boundary.
                if inside_j && i == half - 1 {
                    // Inner edge at x = x1.
                    let mid = (x1, z0 + m);
                    self.tri((x1, z0), mid, (x0, z0));
                    self.tri(mid, (x1, z1), (x0, z1));
                    self.tri((x0, z0), mid, (x0, z1));
                } else if inside_j && i == 3 * half {
                    // Inner edge at x = x0.
                    let mid = (x0, z0 + m);
                    self.tri((x0, z0), mid, (x1, z0));
                    self.tri(mid, (x0, z1), (x1, z1));
                    self.tri((x1, z0), mid, (x1, z1));
                } else if inside_i && j == half - 1 {
                    // Inner edge at z = z1.
                    let mid = (x0 + m, z1);
                    self.tri((x0, z1), mid, (x0, z0));
                    self.tri(mid, (x1, z1), (x1, z0));
                    self.tri((x0, z0), mid, (x1, z0));
                } else if inside_i && j == 3 * half {
                    // Inner edge at z = z0.
                    let mid = (x0 + m, z0);
                    self.tri((x0, z0), mid, (x0, z1));
                    self.tri(mid, (x1, z0), (x1, z1));
                    self.tri((x0, z1), mid, (x1, z1));
                } else {
                    self.quad(x0, z0, s);
                }
            }
        }
    }

    fn into_mesh(self, spacing: f32) -> Mesh {
        let positions: Vec<[f32; 3]> = self
            .positions
            .iter()
            .map(|p| [p[0] * spacing, 0.0, p[2] * spacing])
            .collect();
        let normals = vec![[0.0, 1.0, 0.0]; positions.len()];
        let uvs: Vec<[f32; 2]> = positions.iter().map(|p| [p[0], p[2]]).collect();
        Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::default(),
        )
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
        .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
        .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, uvs)
        .with_inserted_indices(Indices::U32(self.indices))
    }
}

/// Builds the mesh for new or changed clipmaps and writes the LOD parameters into their material.
#[allow(clippy::type_complexity)]
pub(super) fn build_ocean_clipmaps(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<OceanSurfaceMaterial>>,
    clipmaps: Query<
        (
            Entity,
            &OceanClipmap,
            Option<&MeshMaterial3d<OceanSurfaceMaterial>>,
        ),
        Changed<OceanClipmap>,
    >,
) {
    for (entity, clipmap, material) in &clipmaps {
        commands
            .entity(entity)
            .insert((Mesh3d(meshes.add(clipmap.build_mesh())), OceanSurfaceTag));
        if let Some(mut mat) = material.and_then(|m| materials.get_mut(&m.0)) {
            mat.extension.settings.lod_base_spacing = clipmap.base_spacing;
            mat.extension.settings.lod_cells_per_level = clipmap.cells_per_level as f32;
        }
    }
}

/// Moves each clipmap under its camera in whole snap steps.
pub(super) fn follow_ocean_clipmaps(
    targets: Query<&GlobalTransform, With<OceanClipmapTarget>>,
    cameras: Query<&GlobalTransform, With<Camera3d>>,
    mut clipmaps: Query<(&OceanClipmap, &mut Transform)>,
) {
    let Some(camera) = targets.iter().next().or_else(|| cameras.iter().next()) else {
        return;
    };
    let eye = camera.translation();
    for (clipmap, mut transform) in &mut clipmaps {
        let step = clipmap.snap_spacing().max(f32::EPSILON);
        let x = (eye.x / step).round() * step;
        let z = (eye.z / step).round() * step;
        if transform.translation.x != x || transform.translation.z != z {
            transform.translation.x = x;
            transform.translation.z = z;
        }
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

    /// Every edge of a closed, crack-free surface is shared by two triangles, except the outer
    /// border. A T-junction would leave inner edges used once.
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
            let on_border = [a, b].iter().all(|&v| {
                let q = p[v as usize];
                q[0].abs() == extent || q[2].abs() == extent
            });
            if !on_border {
                assert_eq!(
                    count, 2,
                    "inner edge {:?}-{:?}",
                    p[a as usize], p[b as usize]
                );
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
}
