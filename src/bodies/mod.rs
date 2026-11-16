//! Floating bodies as voxels.
//!
//! A body's hull is voxelized once on the CPU with [`voxelize`]. Each [`Voxel`] keeps the volume
//! of the hull inside it, and voxels on the hull surface also keep the outward normal and area of
//! the surface they hold. A force pass can then evaluate buoyancy from the submerged volume and
//! drag and slamming from the surface voxels against the live water state, and sum them into one
//! force and torque per body.
//!
//! The volume comes from vertical rays through a grid of sample points in each voxel column. The
//! inside of a closed mesh is where a ray has crossed an odd number of triangles, so the share of
//! each voxel's height that lies inside is exact along the ray, and the samples average it over
//! the column. The surface comes from spreading each triangle's area and normal over the voxels it
//! passes through.

use bevy::{
    mesh::{PrimitiveTopology, VertexAttributeValues},
    prelude::*,
};

/// Sample rays per voxel column along X and along Z.
const RAYS_PER_AXIS: usize = 4;
/// Ray offsets are nudged by this share of a voxel, so rays miss the shared edges and vertices of
/// grid-aligned meshes, where a crossing would count twice.
const RAY_NUDGE: f32 = 1.0e-3;

/// One voxel of a hull, in the mesh's local space.
#[derive(Clone, Copy, Debug, Default, PartialEq, Reflect)]
pub struct Voxel {
    /// Centre of the voxel.
    pub center: Vec3,
    /// Volume of the hull inside the voxel in cubic meters, up to the voxel's full volume.
    pub volume: f32,
    /// Area-weighted mean outward normal of the hull surface in the voxel. Zero for voxels with no
    /// surface.
    pub normal: Vec3,
    /// Area of the hull surface in the voxel in square meters.
    pub area: f32,
}

impl Voxel {
    /// The voxel holds part of the hull surface.
    pub fn is_surface(&self) -> bool {
        self.area > 0.0
    }
}

/// A mesh turned into voxels by [`voxelize`].
#[derive(Clone, Debug, Default, PartialEq, Reflect)]
pub struct VoxelizedMesh {
    /// Edge length of a voxel in meters.
    pub voxel_size: f32,
    /// Voxels that hold some of the hull's volume or surface.
    pub voxels: Vec<Voxel>,
}

impl VoxelizedMesh {
    /// Volume of the hull in cubic meters.
    pub fn volume(&self) -> f32 {
        self.voxels.iter().map(|v| v.volume).sum()
    }

    /// Area of the hull surface in square meters.
    pub fn area(&self) -> f32 {
        self.voxels.iter().map(|v| v.area).sum()
    }

    /// Centroid of the hull's volume, its centre of buoyancy when fully submerged.
    pub fn centroid(&self) -> Vec3 {
        let volume = self.volume();
        if volume <= 0.0 {
            return Vec3::ZERO;
        }
        self.voxels
            .iter()
            .map(|v| v.center * v.volume)
            .sum::<Vec3>()
            / volume
    }
}

/// Why a mesh can't be voxelized.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VoxelizeError {
    /// Only triangle lists are supported.
    NotTriangleList,
    /// The mesh has no `Float32x3` positions.
    NoPositions,
    /// The voxel size isn't a positive finite number.
    BadVoxelSize,
}

impl std::fmt::Display for VoxelizeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::NotTriangleList => "the mesh is not a triangle list",
            Self::NoPositions => "the mesh has no Float32x3 positions",
            Self::BadVoxelSize => "the voxel size must be positive and finite",
        })
    }
}

impl std::error::Error for VoxelizeError {}

/// Voxelizes a closed triangle mesh in its local space, with voxels of `voxel_size` meters. The
/// grid's origin is the mesh's local origin, so voxel centres sit at `(i + 0.5) * voxel_size`.
///
/// Front faces must wind counter-clockwise seen from outside, as Bevy's own meshes do, for the
/// normals to point outward. The volume needs a closed mesh: an open one leaks, and rays through
/// the hole count it as inside or outside depending on what else they cross. The design notes
/// start at 0.25 m voxels for boats and logs.
pub fn voxelize(mesh: &Mesh, voxel_size: f32) -> Result<VoxelizedMesh, VoxelizeError> {
    if mesh.primitive_topology() != PrimitiveTopology::TriangleList {
        return Err(VoxelizeError::NotTriangleList);
    }
    let Some(VertexAttributeValues::Float32x3(positions)) =
        mesh.attribute(Mesh::ATTRIBUTE_POSITION)
    else {
        return Err(VoxelizeError::NoPositions);
    };
    let positions: Vec<Vec3> = positions.iter().map(|&p| Vec3::from(p)).collect();
    let triangles: Vec<[Vec3; 3]> = match mesh.indices() {
        Some(indices) => {
            let indices: Vec<usize> = indices.iter().collect();
            indices
                .chunks_exact(3)
                .map(|t| [positions[t[0]], positions[t[1]], positions[t[2]]])
                .collect()
        }
        None => positions
            .chunks_exact(3)
            .map(|t| [t[0], t[1], t[2]])
            .collect(),
    };
    voxelize_triangles(&triangles, voxel_size)
}

/// [`voxelize`] for a list of triangles.
pub fn voxelize_triangles(
    triangles: &[[Vec3; 3]],
    voxel_size: f32,
) -> Result<VoxelizedMesh, VoxelizeError> {
    if !(voxel_size.is_finite() && voxel_size > 0.0) {
        return Err(VoxelizeError::BadVoxelSize);
    }
    let s = voxel_size;
    if triangles.is_empty() {
        return Ok(VoxelizedMesh {
            voxel_size: s,
            voxels: Vec::new(),
        });
    }
    let (lo, hi) = triangles
        .iter()
        .flatten()
        .fold((Vec3::MAX, Vec3::MIN), |(lo, hi), &p| {
            (lo.min(p), hi.max(p))
        });
    let min = (lo / s).floor().as_ivec3();
    let max = (hi / s).floor().as_ivec3();
    let dims = (max - min + 1).as_uvec3();
    let cell = |c: IVec3| -> usize {
        let c = (c - min).as_uvec3();
        (c.x + dims.x * (c.y + dims.y * c.z)) as usize
    };
    let mut volume = vec![0.0f32; (dims.x * dims.y * dims.z) as usize];
    let mut surface = vec![(Vec3::ZERO, 0.0f32); volume.len()];

    // Triangles binned by the voxel columns their XZ bounds touch.
    let mut columns: Vec<Vec<usize>> = vec![Vec::new(); (dims.x * dims.z) as usize];
    for (t, tri) in triangles.iter().enumerate() {
        let tlo = tri[0].min(tri[1]).min(tri[2]);
        let thi = tri[0].max(tri[1]).max(tri[2]);
        let c0 = ((tlo / s).floor().as_ivec3() - min).max(IVec3::ZERO);
        let c1 = ((thi / s).floor().as_ivec3() - min).min(dims.as_ivec3() - 1);
        for z in c0.z..=c1.z {
            for x in c0.x..=c1.x {
                columns[(x as u32 + dims.x * z as u32) as usize].push(t);
            }
        }
    }

    // Volume: the inside intervals of vertical rays, averaged over each column.
    // Each ray stands for this much of the column's cross-section.
    let ray_area = s * s / (RAYS_PER_AXIS * RAYS_PER_AXIS) as f32;
    let mut hits: Vec<f32> = Vec::new();
    for z in 0..dims.z as i32 {
        for x in 0..dims.x as i32 {
            let candidates = &columns[(x as u32 + dims.x * z as u32) as usize];
            if candidates.is_empty() {
                continue;
            }
            for rz in 0..RAYS_PER_AXIS {
                for rx in 0..RAYS_PER_AXIS {
                    let offset =
                        |r: usize, nudge: f32| (r as f32 + 0.5 + nudge) / RAYS_PER_AXIS as f32;
                    let ray = Vec2::new(
                        (min.x + x) as f32 + offset(rx, RAY_NUDGE),
                        (min.z + z) as f32 + offset(rz, RAY_NUDGE * 1.618),
                    ) * s;
                    hits.clear();
                    hits.extend(
                        candidates
                            .iter()
                            .filter_map(|&t| ray_hit(&triangles[t], ray)),
                    );
                    hits.sort_by(f32::total_cmp);
                    for pair in hits.chunks_exact(2) {
                        add_interval(
                            &mut volume,
                            &cell,
                            IVec3::new(min.x + x, 0, min.z + z),
                            min.y,
                            max.y,
                            pair[0],
                            pair[1],
                            s,
                            ray_area,
                        );
                    }
                }
            }
        }
    }

    // Surface: each triangle's area and normal, spread over sample points about half a voxel apart.
    for tri in triangles {
        let cross = (tri[1] - tri[0]).cross(tri[2] - tri[0]);
        let area = 0.5 * cross.length();
        if area <= 0.0 {
            continue;
        }
        let normal = cross / (2.0 * area);
        let longest = (tri[1] - tri[0])
            .length()
            .max((tri[2] - tri[1]).length())
            .max((tri[0] - tri[2]).length());
        let n = ((2.0 * longest / s).ceil() as usize).max(1);
        // Barycentric sample points at the centres of the n² sub-triangles.
        let mut samples = Vec::with_capacity(n * n);
        for i in 0..n {
            for j in 0..n - i {
                samples.push((i as f32 + 1.0 / 3.0, j as f32 + 1.0 / 3.0));
                if i + j + 1 < n {
                    samples.push((i as f32 + 2.0 / 3.0, j as f32 + 2.0 / 3.0));
                }
            }
        }
        let share = area / samples.len() as f32;
        for (u, v) in samples {
            let (u, v) = (u / n as f32, v / n as f32);
            // Nudged inward, so a face on a voxel boundary lands in the voxel it bounds.
            let p = tri[0] + (tri[1] - tri[0]) * u + (tri[2] - tri[0]) * v - normal * (s * 1.0e-3);
            let c = (p / s).floor().as_ivec3().clamp(min, max);
            let entry = &mut surface[cell(c)];
            entry.0 += normal * share;
            entry.1 += share;
        }
    }

    let full = s * s * s;
    let mut voxels = Vec::new();
    for z in min.z..=max.z {
        for y in min.y..=max.y {
            for x in min.x..=max.x {
                let c = IVec3::new(x, y, z);
                let i = cell(c);
                let (normal, area) = surface[i];
                let v = volume[i].min(full);
                if v > full * 1.0e-6 || area > 0.0 {
                    voxels.push(Voxel {
                        center: (c.as_vec3() + 0.5) * s,
                        volume: v,
                        normal: normal.normalize_or_zero(),
                        area,
                    });
                }
            }
        }
    }
    Ok(VoxelizedMesh {
        voxel_size: s,
        voxels,
    })
}

/// Height where the vertical ray through `ray` (X, Z) crosses the triangle, if it does.
fn ray_hit(tri: &[Vec3; 3], ray: Vec2) -> Option<f32> {
    let (a, b, c) = (tri[0].xz(), tri[1].xz(), tri[2].xz());
    let d = (b - a).perp_dot(c - a);
    if d == 0.0 {
        return None;
    }
    let w1 = (ray - a).perp_dot(c - a) / d;
    let w2 = (b - a).perp_dot(ray - a) / d;
    let w0 = 1.0 - w1 - w2;
    if w0 < 0.0 || w1 < 0.0 || w2 < 0.0 {
        return None;
    }
    Some(w0 * tri[0].y + w1 * tri[1].y + w2 * tri[2].y)
}

/// Adds the part of the inside interval `y0..y1` of one ray to the voxels of its column.
#[allow(clippy::too_many_arguments)]
fn add_interval(
    volume: &mut [f32],
    cell: &impl Fn(IVec3) -> usize,
    column: IVec3,
    min_y: i32,
    max_y: i32,
    y0: f32,
    y1: f32,
    s: f32,
    ray_area: f32,
) {
    let first = ((y0 / s).floor() as i32).max(min_y);
    let last = ((y1 / s).floor() as i32).min(max_y);
    for y in first..=last {
        let bottom = y as f32 * s;
        let inside = (y1.min(bottom + s) - y0.max(bottom)).max(0.0);
        volume[cell(IVec3::new(column.x, y, column.z))] += ray_area * inside;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f32, b: f32, tolerance: f32) -> bool {
        (a - b).abs() <= tolerance * b.abs()
    }

    #[test]
    fn a_cube_fills_its_voxels_exactly() {
        let mesh = Mesh::from(Cuboid::new(1.0, 1.0, 1.0));
        let v = voxelize(&mesh, 0.25).unwrap();
        assert!(close(v.volume(), 1.0, 1e-3), "volume {}", v.volume());
        assert!(close(v.area(), 6.0, 1e-3), "area {}", v.area());
        assert!(v.centroid().length() < 1e-3, "centroid {:?}", v.centroid());
        // The cube's faces lie on voxel boundaries, so it covers exactly 4³ voxels.
        let filled = v.voxels.iter().filter(|x| x.volume > 0.0).count();
        assert_eq!(filled, 64);
        for x in &v.voxels {
            assert!(x.volume <= 0.25f32.powi(3) * 1.0001);
        }
    }

    /// Exact volume and area of a closed triangle mesh, from the divergence theorem.
    fn mesh_volume_area(mesh: &Mesh) -> (f32, f32) {
        let Some(VertexAttributeValues::Float32x3(p)) = mesh.attribute(Mesh::ATTRIBUTE_POSITION)
        else {
            panic!("positions");
        };
        let idx: Vec<usize> = mesh.indices().unwrap().iter().collect();
        idx.chunks_exact(3).fold((0.0, 0.0), |(v, a), t| {
            let [x, y, z] = [p[t[0]], p[t[1]], p[t[2]]].map(Vec3::from);
            (
                v + x.dot(y.cross(z)) / 6.0,
                a + 0.5 * (y - x).cross(z - x).length(),
            )
        })
    }

    #[test]
    fn a_sphere_matches_its_volume_and_area() {
        let mesh = Sphere::new(1.0).mesh().ico(5).unwrap();
        let v = voxelize(&mesh, 0.1).unwrap();
        let (volume, area) = mesh_volume_area(&mesh);
        assert!(
            close(v.volume(), volume, 0.002),
            "volume {} of {volume}",
            v.volume()
        );
        assert!(close(v.area(), area, 1e-4), "area {} of {area}", v.area());
        for x in v.voxels.iter().filter(|x| x.is_surface()) {
            assert!(x.normal.dot(x.center.normalize()) > 0.8, "{x:?}");
        }
        // Surface samples are half a voxel apart, so a voxel the hull barely clips can miss them.
        // Voxels without surface are full or nearly so.
        let interior = v.voxels.iter().filter(|x| !x.is_surface());
        for x in interior {
            assert!(x.volume >= 0.99e-3, "{x:?}");
        }
    }

    #[test]
    fn an_offset_rotated_box_keeps_its_volume_and_centroid() {
        let rotation = Quat::from_euler(EulerRot::XYZ, 0.3, 0.7, -0.4);
        let offset = Vec3::new(2.3, -1.1, 0.6);
        let mesh = Mesh::from(Cuboid::new(2.0, 0.6, 1.2))
            .rotated_by(rotation)
            .translated_by(offset);
        let v = voxelize(&mesh, 0.1).unwrap();
        assert!(close(v.volume(), 1.44, 0.005), "volume {}", v.volume());
        assert!(
            close(v.area(), 2.0 * (1.2 + 2.4 + 0.72), 0.005),
            "area {}",
            v.area()
        );
        assert!(
            v.centroid().distance(offset) < 0.01,
            "centroid {:?}",
            v.centroid()
        );
    }

    #[test]
    fn coarse_voxels_still_hold_the_whole_volume() {
        let mesh = Sphere::new(0.4).mesh().ico(4).unwrap();
        let v = voxelize(&mesh, 0.25).unwrap();
        let (volume, _) = mesh_volume_area(&mesh);
        assert!(
            close(v.volume(), volume, 0.01),
            "volume {} of {volume}",
            v.volume()
        );
    }

    #[test]
    fn bad_input_is_rejected() {
        let mesh = Mesh::from(Cuboid::default());
        assert_eq!(voxelize(&mesh, 0.0), Err(VoxelizeError::BadVoxelSize));
        let lines = Mesh::new(PrimitiveTopology::LineList, default());
        assert_eq!(voxelize(&lines, 0.25), Err(VoxelizeError::NotTriangleList));
    }
}
