//! Propeller, rudder and skeg forces for the boat example and its tests.
//!
//! Everything here is in the boat's own frame: −Z is ahead, +Y up, +X starboard. The rudder and
//! the skeg are lifting surfaces. Each sees the water flowing past it from the boat's linear and
//! angular velocity, and the rudder also sees the propeller's wash. The lift is
//! `½ ρ A C_L(α) v²` across the flow, linear in the angle of attack up to the stall, and the drag
//! along it adds the induced drag. So the rudder steers even at low speed with the propeller
//! turning, and both surfaces resist a yaw rate or a skid like a keel.
#![allow(dead_code)]

use bevy::prelude::*;

/// Helm settings and how fast they move.
#[derive(Clone, Copy, Debug)]
pub struct Helm {
    /// Share of full thrust, from −[`DriveParams::reverse`] to 1.
    pub throttle: f32,
    /// Rudder angle in radians. Positive turns to starboard.
    pub rudder: f32,
}

/// Where the forces act and how strong they are.
#[derive(Clone, Copy, Debug)]
pub struct DriveParams {
    /// Thrust at full throttle in newtons.
    pub max_thrust: f32,
    /// Share of full thrust available astern.
    pub reverse: f32,
    /// Propeller disc area in m², for the speed of its wash.
    pub disc_area: f32,
    /// Share of the wash speed the rudder sees.
    pub wash_share: f32,
    /// Largest rudder angle in radians.
    pub max_rudder: f32,
    /// Throttle change per second, and rudder slew in radians per second.
    pub throttle_rate: f32,
    pub rudder_rate: f32,
    /// Local points of the propeller, the rudder's centre of pressure and the skeg's.
    pub propeller: Vec3,
    pub rudder_point: Vec3,
    pub skeg_point: Vec3,
    /// Lateral areas in m².
    pub rudder_area: f32,
    pub skeg_area: f32,
    /// Aspect ratio of both surfaces, for the lift slope and the induced drag.
    pub aspect_ratio: f32,
    /// Stall angle in radians.
    pub stall: f32,
    /// Water density in kg/m³.
    pub density: f32,
}

impl DriveParams {
    /// Settings for the example's 7.2 m hull, its stern at `stern_z` and keel at `keel_y`.
    pub fn for_hull(stern_z: f32, keel_y: f32) -> Self {
        Self {
            max_thrust: 4500.0,
            reverse: 0.4,
            disc_area: 0.12,
            wash_share: 0.5,
            max_rudder: 35f32.to_radians(),
            throttle_rate: 0.5,
            rudder_rate: 40f32.to_radians(),
            propeller: Vec3::new(0.0, keel_y + 0.1, stern_z - 0.4),
            rudder_point: Vec3::new(0.0, keel_y + 0.05, stern_z - 0.1),
            skeg_point: Vec3::new(0.0, keel_y - 0.1, stern_z - 1.6),
            rudder_area: 0.15,
            skeg_area: 0.35,
            aspect_ratio: 1.5,
            stall: 25f32.to_radians(),
            density: 1025.0,
        }
    }

    /// Lift slope per radian of a low aspect ratio foil.
    fn lift_slope(&self) -> f32 {
        std::f32::consts::TAU * self.aspect_ratio / (self.aspect_ratio + 2.0)
    }

    /// Lift coefficient at angle of attack `alpha`: linear up to the stall, then holding at 70%
    /// of its peak.
    pub fn lift_coefficient(&self, alpha: f32) -> f32 {
        let a = alpha.clamp(-std::f32::consts::FRAC_PI_2, std::f32::consts::FRAC_PI_2);
        if a.abs() <= self.stall {
            self.lift_slope() * a
        } else {
            a.signum() * self.lift_slope() * self.stall * 0.7
        }
    }

    /// Speed of the propeller's wash in m/s at `thrust` newtons, from momentum theory.
    pub fn wash_speed(&self, thrust: f32) -> f32 {
        (2.0 * thrust.max(0.0) / (self.density * self.disc_area)).sqrt()
    }
}

/// Moves the helm toward the requested throttle and rudder at the helm's rates.
pub fn steer(helm: &mut Helm, params: &DriveParams, throttle: f32, rudder: f32, dt: f32) {
    let throttle = throttle.clamp(-params.reverse, 1.0);
    let rudder = rudder.clamp(-params.max_rudder, params.max_rudder);
    let step = |from: f32, to: f32, rate: f32| from + (to - from).clamp(-rate * dt, rate * dt);
    helm.throttle = step(helm.throttle, throttle, params.throttle_rate);
    helm.rudder = step(helm.rudder, rudder, params.rudder_rate);
}

/// Force of one lifting surface at `point`, set `angle` radians off the centre line, in water
/// flowing past it at `flow` (local, m/s, the water's velocity relative to the surface).
pub fn foil_force(params: &DriveParams, area: f32, angle: f32, flow: Vec3) -> Vec3 {
    let flow = flow.with_y(0.0);
    let speed_sq = flow.length_squared();
    if speed_sq < 1e-6 {
        return Vec3::ZERO;
    }
    let dir = flow / speed_sq.sqrt();
    // Flow from ahead runs along +Z. A flow reversed by going astern still lifts, the foil
    // working back to front.
    let (dir_fwd, sign) = if dir.z >= 0.0 {
        (dir, 1.0)
    } else {
        (-dir, -1.0)
    };
    let beta = dir_fwd.x.atan2(dir_fwd.z);
    let alpha = beta - angle;
    let cl = params.lift_coefficient(alpha);
    let cd = 0.02 + cl * cl / (std::f32::consts::PI * params.aspect_ratio * 0.9);
    let q = 0.5 * params.density * area * speed_sq;
    // Across the flow, toward +X for flow along +Z: a flow from starboard (β < 0) pushes the
    // surface to port, back into line.
    let across = Vec3::new(dir_fwd.z, 0.0, -dir_fwd.x);
    across * q * cl * sign + dir * q * cd
}

/// Drive forces in the boat's frame: the force, and the points it acts at with their own forces,
/// for a boat moving at `linear` and turning at `angular` (local, relative to the water).
/// `immersion` from 0 to 1 fades the thrust and the foils as they leave the water.
pub fn drive_forces(
    params: &DriveParams,
    helm: &Helm,
    linear: Vec3,
    angular: Vec3,
    immersion: f32,
) -> [(Vec3, Vec3); 3] {
    let immersion = immersion.clamp(0.0, 1.0);
    let thrust = helm.throttle * params.max_thrust * immersion;
    let propeller = (params.propeller, Vec3::NEG_Z * thrust);
    let flow_at = |p: Vec3| -(linear + angular.cross(p));
    let wash = Vec3::Z * params.wash_speed(thrust) * params.wash_share;
    let rudder = (
        params.rudder_point,
        foil_force(
            params,
            params.rudder_area,
            helm.rudder,
            flow_at(params.rudder_point) + wash,
        ) * immersion,
    );
    let skeg = (
        params.skeg_point,
        foil_force(params, params.skeg_area, 0.0, flow_at(params.skeg_point)) * immersion,
    );
    [propeller, rudder, skeg]
}

/// Sum of `forces` and their torque about `center`, all local.
pub fn total(forces: &[(Vec3, Vec3)], center: Vec3) -> (Vec3, Vec3) {
    forces
        .iter()
        .fold((Vec3::ZERO, Vec3::ZERO), |(f, t), &(p, force)| {
            (f + force, t + (p - center).cross(force))
        })
}
