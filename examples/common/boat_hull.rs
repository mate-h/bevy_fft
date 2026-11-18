//! A procedural motor boat hull for the boat example and its tests.
//!
//! The hull is lofted through cross sections from the transom to the stem. Each section is a
//! rounded V below the waterline and a flared topside above it, closed by a flat deck. The bow
//! narrows to a raked stem line, and the stern ends in a flat transom. Local axes: −Z is the bow,
//! +Y up, +X starboard, and the design waterline is at y = 0.
#![allow(dead_code)]

use bevy::{
    asset::RenderAssetUsages,
    mesh::{Indices, PrimitiveTopology},
    prelude::*,
};
use bevy_fft::bodies::{VoxelizedMesh, voxelize};

/// Dimensions of a hull in meters.
#[derive(Clone, Copy, Debug)]
pub struct HullShape {
    /// Overall length, stern to stem.
    pub length: f32,
    /// Greatest beam at the waterline.
    pub beam: f32,
    /// Keel depth below the design waterline at the deepest section.
    pub draft: f32,
    /// Deck height above the design waterline amidships.
    pub freeboard: f32,
    /// Sections along the length, and points on each half section below the waterline.
    pub stations: usize,
    pub section_points: usize,
}

impl Default for HullShape {
    fn default() -> Self {
        Self {
            length: 7.2,
            beam: 2.5,
            draft: 0.6,
            freeboard: 0.85,
            stations: 32,
            section_points: 8,
        }
    }
}

impl HullShape {
    /// Half beam at the waterline at `s`, from 0 at the transom to 1 at the stem.
    fn half_beam(&self, s: f32) -> f32 {
        let b = 0.5 * self.beam;
        if s < 0.4 {
            // The transom is a little narrower than the widest section.
            b * (1.0 - 0.12 * ((0.4 - s) / 0.4).powi(2))
        } else {
            let u = ((s - 0.4) / 0.6).clamp(0.0, 1.0);
            b * (1.0 - u * u).max(0.0).powf(0.7)
        }
    }

    /// Keel depth below the waterline at `s`. The forefoot rises toward the stem, and the
    /// transom is just immersed.
    fn keel_depth(&self, s: f32) -> f32 {
        let d = self.draft;
        if s < 0.5 {
            d * (0.75 + 0.25 * (s / 0.5))
        } else {
            let u = ((s - 0.5) / 0.5).clamp(0.0, 1.0);
            d * (1.0 - u.powf(1.6) * 1.05)
        }
    }

    /// Deck height at `s`, with sheer rising to the bow.
    fn deck_height(&self, s: f32) -> f32 {
        self.freeboard * (1.0 + 0.35 * s * s)
    }

    /// Z of section `s`.
    fn z(&self, s: f32) -> f32 {
        0.5 * self.length - s * self.length
    }

    /// Points of the starboard half section at `s`, from the keel up to the deck edge.
    fn half_section(&self, s: f32) -> Vec<Vec2> {
        let b = self.half_beam(s);
        let d = self.keel_depth(s);
        let n = self.section_points;
        let mut points = Vec::with_capacity(n + 2);
        for k in 0..=n {
            // A V from the keel to the waterline, rounded at the bilge by blending toward an
            // ellipse quarter.
            let t = k as f32 / n as f32;
            let theta = t * std::f32::consts::FRAC_PI_2;
            let v = Vec2::new(t, t - 1.0);
            let round = Vec2::new(theta.sin(), -theta.cos());
            let Vec2 { x, y } = v.lerp(round, 0.4) * Vec2::new(b, d);
            points.push(Vec2::new(x, y));
        }
        // Flared topside to the deck edge.
        points.push(Vec2::new(b * 1.06 + 0.02, self.deck_height(s)));
        points
    }
}

/// A hull: its closed render mesh, its voxels, and the mass and centre of mass that float it
/// level at the design waterline.
pub struct BoatHull {
    pub shape: HullShape,
    pub mesh: Mesh,
    /// Every hull vertex, for a convex collider.
    pub points: Vec<Vec3>,
    pub voxels: VoxelizedMesh,
    /// Volume below the design waterline from the voxels, in m³.
    pub displacement: f32,
    /// Mass in kg that floats the hull at its design waterline.
    pub mass: f32,
    /// Local centre of mass: above the centre of buoyancy, so the boat floats level.
    pub center_of_mass: Vec3,
    /// Principal moments of inertia about the centre of mass, in kg m².
    pub inertia: Vec3,
}

/// Height of the centre of mass above the design waterline, in meters. Engine and ballast sit low.
pub const CENTER_OF_MASS_Y: f32 = -0.05;

/// Principal moments of inertia per kilogram about the centre of mass, in m², from the usual
/// radii of gyration of small craft: about 0.38 of the beam in roll and a quarter of the length
/// in pitch and yaw.
pub fn inertia_per_kg(shape: &HullShape) -> Vec3 {
    let roll = 0.38 * shape.beam;
    let pitch = 0.25 * shape.length;
    Vec3::new(pitch * pitch, pitch * pitch + roll * roll, roll * roll)
}

/// Builds the hull mesh, voxelizes it at `voxel_size`, and sizes its mass for water of `density`.
pub fn build_hull(shape: HullShape, voxel_size: f32, density: f32) -> BoatHull {
    let (mesh, points) = hull_mesh(&shape);
    let voxels = voxelize(&mesh, voxel_size).expect("the hull mesh is a closed triangle list");
    // Voxel boundaries fall on multiples of the voxel size, so y = 0 is one of them.
    let below: Vec<_> = voxels.voxels.iter().filter(|v| v.center.y < 0.0).collect();
    let displacement: f32 = below.iter().map(|v| v.volume).sum();
    let buoyancy_center =
        below.iter().map(|v| v.center * v.volume).sum::<Vec3>() / displacement.max(1e-6);
    let mass = density * displacement;
    let center_of_mass = Vec3::new(0.0, CENTER_OF_MASS_Y, buoyancy_center.z);
    let inertia = mass * inertia_per_kg(&shape);
    BoatHull {
        shape,
        mesh,
        points,
        voxels,
        displacement,
        mass,
        center_of_mass,
        inertia,
    }
}

/// The closed hull mesh, with smooth normals on the hull and flat ones on the deck and transom,
/// and its vertex positions.
pub fn hull_mesh(shape: &HullShape) -> (Mesh, Vec<Vec3>) {
    let stations = shape.stations.max(2);
    // Full sections, port deck edge to starboard deck edge, as 3D points.
    let sections: Vec<Vec<Vec3>> = (0..=stations)
        .map(|i| {
            let s = i as f32 / stations as f32;
            let z = shape.z(s);
            let half = shape.half_section(s);
            // At the stem every section point collapses onto the centre line.
            let squash = if i == stations { 0.0 } else { 1.0 };
            let mut full: Vec<Vec3> = half
                .iter()
                .rev()
                .map(|p| Vec3::new(-p.x * squash, p.y, z))
                .collect();
            full.extend(half.iter().skip(1).map(|p| Vec3::new(p.x * squash, p.y, z)));
            full
        })
        .collect();
    let ring = sections[0].len();

    let mut positions: Vec<Vec3> = Vec::new();
    let mut normals: Vec<Vec3> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    // Hull skin: shared vertices, smooth normals.
    for section in &sections {
        positions.extend(section.iter().copied());
    }
    let skin_vertices = positions.len();
    normals.resize(skin_vertices, Vec3::ZERO);
    for i in 0..stations {
        for j in 0..ring - 1 {
            let a = (i * ring + j) as u32;
            let b = a + 1;
            let c = a + ring as u32;
            let d = c + 1;
            indices.extend([a, b, c, b, d, c]);
        }
    }

    // Deck: the top edge of each section, flat.
    let deck_start = positions.len() as u32;
    for section in &sections {
        positions.push(section[0]);
        positions.push(section[ring - 1]);
        normals.push(Vec3::Y);
        normals.push(Vec3::Y);
    }
    for i in 0..stations as u32 {
        let a = deck_start + 2 * i;
        indices.extend([a, a + 2, a + 1, a + 1, a + 2, a + 3]);
    }

    // Transom: a fan over the stern section, flat.
    let transom_start = positions.len() as u32;
    let centre = sections[0].iter().copied().sum::<Vec3>() / ring as f32;
    positions.push(centre);
    normals.push(Vec3::Z);
    for &p in &sections[0] {
        positions.push(p);
        normals.push(Vec3::Z);
    }
    for j in 0..ring as u32 {
        let a = transom_start + 1 + j;
        let b = transom_start + 1 + (j + 1) % ring as u32;
        // Same winding sense as the skin and deck.
        indices.extend([transom_start, b, a]);
    }

    // Wind every triangle outward: flip all of them if the enclosed volume comes out negative.
    let signed_volume: f32 = indices
        .chunks_exact(3)
        .map(|t| {
            let (a, b, c) = (
                positions[t[0] as usize],
                positions[t[1] as usize],
                positions[t[2] as usize],
            );
            a.dot(b.cross(c)) / 6.0
        })
        .sum();
    if signed_volume < 0.0 {
        for t in indices.chunks_exact_mut(3) {
            t.swap(1, 2);
        }
    }

    // Smooth skin normals from the area-weighted face normals around each vertex.
    for t in indices.chunks_exact(3) {
        if (t[0] as usize) < skin_vertices {
            let (a, b, c) = (
                positions[t[0] as usize],
                positions[t[1] as usize],
                positions[t[2] as usize],
            );
            let n = (b - a).cross(c - a);
            for &v in t {
                normals[v as usize] += n;
            }
        }
    }
    for n in &mut normals[..skin_vertices] {
        *n = n.normalize_or(Vec3::Y);
    }
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    );
    mesh.insert_attribute(
        Mesh::ATTRIBUTE_POSITION,
        positions.iter().map(|p| p.to_array()).collect::<Vec<_>>(),
    );
    mesh.insert_attribute(
        Mesh::ATTRIBUTE_NORMAL,
        normals.iter().map(|n| n.to_array()).collect::<Vec<_>>(),
    );
    mesh.insert_indices(Indices::U32(indices));
    (mesh, positions)
}
