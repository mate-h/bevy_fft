//! The boat example's procedural hull.
#[path = "../examples/common/boat_hull.rs"]
mod boat_hull;

use bevy::prelude::*;
use boat_hull::*;

const SEA_WATER: f32 = 1025.0;

/// Volume of a closed triangle mesh from the divergence theorem.
fn mesh_volume(mesh: &Mesh) -> f32 {
    let Some(bevy::mesh::VertexAttributeValues::Float32x3(p)) =
        mesh.attribute(Mesh::ATTRIBUTE_POSITION)
    else {
        panic!("no positions");
    };
    let p: Vec<Vec3> = p.iter().map(|&v| Vec3::from(v)).collect();
    let idx: Vec<usize> = mesh.indices().unwrap().iter().collect();
    idx.chunks_exact(3)
        .map(|t| p[t[0]].dot(p[t[1]].cross(p[t[2]])) / 6.0)
        .sum()
}

#[test]
fn the_hull_is_closed_and_wound_outward() {
    let hull = build_hull(HullShape::default(), 0.2, SEA_WATER);
    let exact = mesh_volume(&hull.mesh);
    assert!(exact > 0.0, "inward winding: {exact} m³");
    // The voxels hold the same volume as the mesh, so no ray leaks through a gap.
    let voxels = hull.voxels.volume();
    assert!(
        (voxels - exact).abs() < 0.01 * exact,
        "voxels {voxels} m³ vs mesh {exact} m³"
    );
}

#[test]
fn the_hull_floats_level_at_its_design_draft() {
    let shape = HullShape::default();
    let hull = build_hull(shape, 0.2, SEA_WATER);
    // A displacement hull of this size weighs 3.5-4.5 t.
    let block = hull.displacement / (shape.length * shape.beam * shape.draft);
    assert!((0.35..0.5).contains(&block), "block coefficient {block}");
    assert!(
        (3500.0..4500.0).contains(&hull.mass),
        "mass {} kg",
        hull.mass
    );
    // Its weight equals the buoyancy of the water it displaces at the waterline.
    assert!((hull.mass - SEA_WATER * hull.displacement).abs() < 1.0);
    // The centre of mass sits on the centre line, over the centre of buoyancy, and below the
    // hull's centroid.
    let below: Vec<_> = hull
        .voxels
        .voxels
        .iter()
        .filter(|v| v.center.y < 0.0)
        .collect();
    let buoyancy = below.iter().map(|v| v.center * v.volume).sum::<Vec3>() / hull.displacement;
    assert!(hull.center_of_mass.x.abs() < 1e-6);
    assert!((hull.center_of_mass.z - buoyancy.z).abs() < 1e-4);
    assert!(hull.center_of_mass.y < hull.voxels.centroid().y);
}

#[test]
fn the_hull_is_symmetric_and_points_its_bow_forward() {
    let hull = build_hull(HullShape::default(), 0.2, SEA_WATER);
    let c = hull.voxels.centroid();
    assert!(c.x.abs() < 0.01, "centroid off the centre line: {c}");
    // The bow is fine and the stern full, so the volume sits aft of midships, toward +Z.
    assert!(c.z > 0.0, "centroid {c}");
    let lo = hull.points.iter().fold(Vec3::MAX, |a, &p| a.min(p));
    let hi = hull.points.iter().fold(Vec3::MIN, |a, &p| a.max(p));
    assert!((hi.z - lo.z - 7.2).abs() < 1e-4);
    assert!((hi.x - lo.x - 2.5 * 1.06 - 0.04).abs() < 0.05, "{lo} {hi}");
}
