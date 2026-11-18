//! The boat example's physics, shared with its GPU test: the water forces with the added mass and
//! damping the hull needs, the propeller, rudder and skeg, and the wake patch that follows it.
//!
//! The including file declares `boat_drive` and `boat_hull` next to this module.
#![allow(dead_code)]

use std::sync::Arc;

use bevy::prelude::*;
use bevy_fft::prelude::*;
use bevy_rapier3d::prelude::*;

use super::{boat_drive, boat_hull};

/// Edge of a hull voxel in meters.
pub const VOXEL: f32 = 0.2;
/// Sea water density in kg/m³.
pub const SEA_WATER: f32 = 1025.0;
/// Readbacks of the water forces to wait for before the boat is let go.
pub const SETTLE_READBACKS: u32 = 30;
/// Cells per edge of the wake patch, and their width in meters. Half-meter cells resolve the
/// boat's 2.5 m beam in five cells and the 6 m transverse waves at 3 m/s in twelve. With 1 m cells
/// the wake behind a boat at 3.5 m/s was a seventh as high.
pub const WAKE_CELLS: u32 = 256;
pub const WAKE_CELL: f32 = 0.5;
/// The wake patch moves to recentre on the boat once it is this many cells off centre.
pub const WAKE_FOLLOW_MARGIN: f32 = 4.0;
/// Depth of the open sea under the patch in meters, and the speed whose Kelvin wavelength picks
/// the patch's mode.
pub const OPEN_SEA_DEPTH: f32 = 100.0;
pub const WAKE_DESIGN_SPEED: f32 = 3.5;
/// Half width in meters of the foam the stern leaves, about the boat's half beam.
pub const WAKE_FOAM_HALF_WIDTH: f32 = 1.4;

/// Added mass in heave and added inertia in pitch and roll, as shares of the boat's own.
///
/// A hull that heaves or pitches has to move water with it. Strip theory gives about twice the
/// boat's mass in heave for this hull, and the same order in pitch; roll moves much less water.
pub const ADDED_MASS: Vec3 = Vec3::new(1.0, 1.5, 0.25);

/// Damping rates of heave, roll and pitch in 1/s, from about 15–25% of critical damping for this
/// hull.
pub const HEAVE_DAMPING: f32 = 1.5;
pub const ROLL_DAMPING: f32 = 1.5;
pub const PITCH_DAMPING: f32 = 2.0;

/// The boat the camera follows and the helm steers.
#[derive(Component)]
pub struct Boat;

/// The boat's helm and drive. Set [`Self::lever`] and [`Self::rudder`], and the helm moves toward
/// them at its rates.
#[derive(Component)]
pub struct BoatDrive {
    pub helm: boat_drive::Helm,
    pub params: boat_drive::DriveParams,
    /// Throttle lever position, from −[`boat_drive::DriveParams::reverse`] to 1.
    pub lever: f32,
    /// Rudder angle asked for, in radians.
    pub rudder: f32,
}

/// Runs the boat's forces each frame: [`apply_water_forces`], then [`apply_drive`], and keeps a
/// [`DispersiveController`] wake patch on the boat if there is one.
pub struct BoatPhysicsPlugin;

impl Plugin for BoatPhysicsPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            Update,
            (
                apply_water_forces,
                apply_drive.after(apply_water_forces),
                follow_boat_with_wake
                    .after(apply_water_forces)
                    .run_if(resource_exists::<DispersiveController>),
            ),
        );
    }
}

/// The boat's physics components for `hull`, at rest on its design waterline.
pub fn boat_physics(hull: &boat_hull::BoatHull) -> impl Bundle {
    let shape = hull.shape;
    let collider = Collider::convex_hull(&hull.points).expect("the hull has volume");
    let mut body = FloatingBody::new(Arc::new(hull.voxels.clone()), hull.mass);
    body.center_of_mass = hull.center_of_mass;
    // Rapier's velocities drive the drag.
    body.estimate_velocity = false;
    (
        Boat,
        BoatDrive {
            helm: boat_drive::Helm {
                throttle: 0.0,
                rudder: 0.0,
            },
            params: boat_drive::DriveParams::for_hull(0.5 * shape.length, -shape.draft),
            lever: 0.0,
            rudder: 0.0,
        },
        body,
        // Held until the forces arrive; see `apply_water_forces`.
        RigidBody::Fixed,
        collider,
        ColliderMassProperties::MassProperties(MassProperties {
            local_center_of_mass: hull.center_of_mass,
            mass: hull.mass,
            principal_inertia_local_frame: Quat::IDENTITY,
            principal_inertia: hull.inertia,
        }),
        Velocity::default(),
        ExternalForce::default(),
        Sleeping::disabled(),
    )
}

/// A deep-water wake patch for the boat, with cells `cell` meters wide: the hull presses it, and
/// its brush, with no strength, marks the foam the stern leaves.
pub fn wake_patch(
    commands: &mut Commands,
    images: &mut Assets<Image>,
    cell: f32,
) -> DispersiveController {
    let mut wake = DispersiveController::spawn(commands, images, WAKE_CELLS);
    wake.tile_world = WAKE_CELLS as f32 * cell;
    wake.placement = Some(DispersivePlacement::default());
    wake.enable_bodies(images);
    wake.brush_strength = 0.0;
    wake.brush_radius = WAKE_FOAM_HALF_WIDTH / cell;
    wake.choose_mode(OPEN_SEA_DEPTH, kelvin_wavelength(WAKE_DESIGN_SPEED, wake.g));
    wake
}

/// Hands the latest water forces to Rapier, with the added mass and damping the hull needs, and
/// Rapier's velocities to the force pass.
///
/// It holds the boat until the force pass has caught up, so it doesn't sink while the pipelines
/// compile, and meanwhile moves it up or down to where it displaces its own weight, so it isn't let
/// go in a trough or under a crest.
#[allow(clippy::type_complexity)]
pub fn apply_water_forces(
    settings: Res<FloatingBodySettings>,
    time: Res<Time>,
    mut bodies: Query<
        (
            &mut FloatingBody,
            &mut ExternalForce,
            &Velocity,
            &GlobalTransform,
            &mut RigidBody,
            &mut Transform,
        ),
        With<Boat>,
    >,
    mut last_frame: Local<u32>,
) {
    for (mut body, mut external, velocity, transform, mut rigid, mut held) in &mut bodies {
        let ready = body.frame > SETTLE_READBACKS;
        if !ready && body.frame > 0 && body.waterplane_area > 0.0 && body.frame != *last_frame {
            *last_frame = body.frame;
            let excess = body.submerged_volume - body.mass / SEA_WATER;
            held.translation.y += (excess / body.waterplane_area).clamp(-0.05, 0.05);
        }
        let kind = if ready {
            RigidBody::Dynamic
        } else {
            RigidBody::Fixed
        };
        if *rigid != kind {
            *rigid = kind;
        }
        body.linear_velocity = velocity.linear;
        body.angular_velocity = velocity.angular;
        let center = transform.transform_point(body.center_of_mass);
        let (force, torque) = body.compensated(
            center,
            velocity.linear,
            velocity.angular,
            time.delta_secs(),
            &settings,
        );
        let (damp_force, damp_torque) = radiation_damping(
            &body,
            transform.rotation(),
            velocity.linear,
            velocity.angular,
        );
        let (force, torque) = with_added_mass(
            &body,
            &settings,
            transform.rotation(),
            force + damp_force,
            torque + damp_torque,
        );
        external.force = force;
        external.torque = torque;
    }
}

/// Share of the boat's weight its submerged volume floats, from 0 to 1.
fn wet_share(body: &FloatingBody) -> f32 {
    (body.submerged_volume * SEA_WATER / body.mass.max(1.0)).clamp(0.0, 1.0)
}

/// Scales the water's net vertical force and its pitch and roll torques to act on the boat plus
/// the water it carries. Rapier knows only the boat's own mass, and gravity acts on that alone,
/// so the net vertical force, buoyancy and drag minus weight, is divided by the total mass
/// share. Surge, sway and yaw move little water and keep their forces.
///
/// Without it the boat heaves and pitches several times too fast, it bounces off waves, and the
/// forces, which arrive a few frames late, get out of step with its motion.
pub fn with_added_mass(
    body: &FloatingBody,
    settings: &FloatingBodySettings,
    rotation: Quat,
    force: Vec3,
    torque: Vec3,
) -> (Vec3, Vec3) {
    let share = Vec3::ONE / (Vec3::ONE + ADDED_MASS * wet_share(body));
    let weight = body.mass * settings.gravity;
    let force = force.with_y(weight + (force.y - weight) * share.y);
    let local = rotation.inverse() * torque;
    let local = Vec3::new(local.x * share.x, local.y, local.z * share.z);
    (force, rotation * local)
}

/// Damping of heave, roll and pitch that stands in for wave radiation.
///
/// A real hull that heaves, rolls or pitches makes waves, and the energy they carry off damps
/// the motion. The voxel forces have no such term, and their restoring torque arrives a few frames
/// late, which pumps energy into the motion instead. Without this a boat at rest in calm water
/// starts pitching within seconds and soon flips.
pub fn radiation_damping(
    body: &FloatingBody,
    rotation: Quat,
    linear: Vec3,
    angular: Vec3,
) -> (Vec3, Vec3) {
    let submerged = wet_share(body);
    let force = -Vec3::Y * body.mass * HEAVE_DAMPING * linear.y * submerged;
    // Roll and pitch rates in the boat's frame, from the hull's moments of inertia.
    let local = rotation.inverse() * angular;
    let inertia = body.mass * boat_hull::inertia_per_kg(&boat_hull::HullShape::default());
    let torque_local = -Vec3::new(
        inertia.x * PITCH_DAMPING * local.x,
        0.0,
        inertia.z * ROLL_DAMPING * local.z,
    ) * submerged;
    (force, rotation * torque_local)
}

/// Moves the helm toward the lever and rudder, and adds the propeller, rudder and skeg forces to
/// the water's. They're computed on the CPU from the boat's current motion, so unlike the water
/// forces they arrive without delay.
pub fn apply_drive(
    time: Res<Time>,
    mut boat: Query<(
        &mut BoatDrive,
        &FloatingBody,
        &GlobalTransform,
        &Velocity,
        &mut ExternalForce,
    )>,
) {
    for (mut drive, body, transform, velocity, mut external) in &mut boat {
        let (params, lever, rudder) = (drive.params, drive.lever, drive.rudder);
        boat_drive::steer(&mut drive.helm, &params, lever, rudder, time.delta_secs());
        let rotation = transform.rotation();
        let to_local = rotation.inverse();
        let linear = to_local * (velocity.linear - body.water_velocity);
        let angular = to_local * velocity.angular;
        // The stern gear works fully until half the hull is out of the water.
        let immersion =
            (body.submerged_volume * SEA_WATER / body.mass.max(1.0) / 0.5).clamp(0.0, 1.0);
        let forces = boat_drive::drive_forces(&params, &drive.helm, linear, angular, immersion);
        let (force, torque) = boat_drive::total(&forces, body.center_of_mass);
        external.force += rotation * force;
        external.torque += rotation * torque;
    }
}

/// Keeps the wake patch on the boat, a few whole cells at a time, and steps it by the frame's
/// time. The patch otherwise steps a fixed 1/60 s per frame, and at any other frame rate its waves
/// would run faster or slower than the boat that makes them.
pub fn follow_boat_with_wake(
    time: Res<Time>,
    boat: Query<(&Transform, &BoatDrive), With<Boat>>,
    mut wake: ResMut<DispersiveController>,
) {
    let Ok((boat, drive)) = boat.single() else {
        return;
    };
    wake.follow(boat.translation.xz(), WAKE_FOLLOW_MARGIN);
    // Foam behind the transom, from the patch's hull foam, which follows the brush pointer.
    let stern = boat
        .transform_point(drive.params.rudder_point.with_y(0.0))
        .xz();
    let at = wake.world_to_cells(stern);
    wake.pointer_prev = if wake.brush_active { wake.pointer } else { at };
    wake.pointer = at;
    wake.brush_active = true;
    let dt = time.delta_secs();
    if dt > 0.0 {
        wake.dt = dt.clamp(1.0 / 240.0, 1.0 / 30.0);
    }
}
