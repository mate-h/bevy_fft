//! Drive forces of the boat example.
#[path = "../examples/common/boat_drive.rs"]
mod boat_drive;

use bevy::prelude::*;
use boat_drive::*;

fn params() -> DriveParams {
    DriveParams::for_hull(3.6, -0.45)
}

/// Centre of mass a little forward of the stern gear.
const CENTER: Vec3 = Vec3::new(0.0, -0.05, 0.4);

#[test]
fn a_centred_rudder_going_straight_gives_no_yaw() {
    let p = params();
    let helm = Helm {
        throttle: 1.0,
        rudder: 0.0,
    };
    let forces = drive_forces(&p, &helm, Vec3::NEG_Z * 3.0, Vec3::ZERO, 1.0);
    let (force, torque) = total(&forces, CENTER);
    assert!(torque.y.abs() < 1e-3, "yaw torque {torque}");
    assert!(force.x.abs() < 1e-3, "side force {force}");
    // Full thrust ahead, less the foils' drag.
    assert!(force.z < -7500.0 && force.z > -8000.0, "{force}");
}

#[test]
fn starboard_rudder_turns_the_bow_to_starboard() {
    let p = params();
    for (throttle, speed) in [(1.0, 3.0), (1.0, 0.0), (0.0, 2.0)] {
        let helm = Helm {
            throttle,
            rudder: 20f32.to_radians(),
        };
        let forces = drive_forces(&p, &helm, Vec3::NEG_Z * speed, Vec3::ZERO, 1.0);
        let (_, torque) = total(&forces, CENTER);
        // Turning from −Z toward +X is a negative rotation about +Y.
        assert!(
            torque.y < -50.0,
            "throttle {throttle} at {speed} m/s: yaw torque {torque}"
        );
    }
}

#[test]
fn going_astern_reverses_the_rudder() {
    let p = params();
    let helm = Helm {
        throttle: 0.0,
        rudder: 20f32.to_radians(),
    };
    let ahead = total(
        &drive_forces(&p, &helm, Vec3::NEG_Z * 2.0, Vec3::ZERO, 1.0),
        CENTER,
    )
    .1;
    let astern = total(
        &drive_forces(&p, &helm, Vec3::Z * 2.0, Vec3::ZERO, 1.0),
        CENTER,
    )
    .1;
    assert!(ahead.y * astern.y < 0.0, "{ahead} {astern}");
}

#[test]
fn the_skeg_opposes_a_yaw_rate_and_a_skid() {
    let p = params();
    let helm = Helm {
        throttle: 0.0,
        rudder: 0.0,
    };
    // Swinging to port while going ahead.
    let (_, torque) = total(
        &drive_forces(&p, &helm, Vec3::NEG_Z * 3.0, Vec3::Y * 0.3, 1.0),
        CENTER,
    );
    assert!(torque.y < 0.0, "yaw torque {torque}");
    // Skidding to starboard: pushed back to port.
    let (force, _) = total(
        &drive_forces(&p, &helm, Vec3::new(0.5, 0.0, -3.0), Vec3::ZERO, 1.0),
        CENTER,
    );
    assert!(force.x < 0.0, "side force {force}");
}

#[test]
fn the_foils_stall_and_leave_the_water() {
    let p = params();
    let peak = p.lift_coefficient(p.stall);
    assert!(p.lift_coefficient(40f32.to_radians()) < peak);
    assert!(p.lift_coefficient(-p.stall) == -peak);
    let helm = Helm {
        throttle: 1.0,
        rudder: 0.3,
    };
    for (_, f) in drive_forces(&p, &helm, Vec3::NEG_Z * 3.0, Vec3::ZERO, 0.0) {
        assert_eq!(f, Vec3::ZERO);
    }
}

#[test]
fn the_helm_moves_at_its_rates() {
    let p = params();
    let mut helm = Helm {
        throttle: 0.0,
        rudder: 0.0,
    };
    steer(&mut helm, &p, 1.0, 1.0, 0.5);
    assert!((helm.throttle - 0.25).abs() < 1e-6);
    assert!((helm.rudder - 20f32.to_radians()).abs() < 1e-6);
    for _ in 0..20 {
        steer(&mut helm, &p, -1.0, 1.0, 0.5);
    }
    assert_eq!(helm.throttle, -p.reverse);
    assert_eq!(helm.rudder, p.max_rudder);
}
