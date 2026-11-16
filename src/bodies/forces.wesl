// Water forces on voxelized floating bodies. One workgroup per body loops over its voxels, samples
// the water LOD data at each one, and sums buoyancy, friction, pressure drag and slamming into
// one force and torque about the body's centre of mass.
//
// The force model follows Kerner's boat model (2015, parts 1 and 2), with relative velocities
// measured against the local water velocity instead of still water:
// - buoyancy is ρ g times the submerged part of each voxel, applied at its centroid;
// - friction on wetted hull surface uses the ITTC 1957 line, C_f = 0.075 / (log10 Re - 2)², with
//   the Reynolds number of the local slip speed over the hull length;
// - pressure drag pushes back on surface moving into the water, and suction pulls on surface
//   moving away, each linear plus quadratic in speed and falling off with the angle of attack;
// - slamming stops surface that enters the water fast, scaled by how fast its wetted area times
//   normal speed grows.

import bevy_fft::water::lod::{WaterLodSlices, sample_lod_finest};

/// Fixed-point steps that undo horizontal displacement, as in the ocean queries.
const INVERT_STEPS: u32 = 4u;
const WORKGROUP: u32 = 64u;

struct Patch {
    /// Local centroid in `.xyz`, area in `.w`.
    position_area: vec4<f32>,
    /// Local outward normal in `.xyz`.
    normal: vec4<f32>,
}

struct Body {
    rotation: vec4<f32>,
    /// World position of the body's origin in `.xyz`, voxel size in `.w`.
    position: vec4<f32>,
    /// World centre of mass in `.xyz`, mass in kilograms in `.w`.
    center_of_mass: vec4<f32>,
    /// Linear velocity in `.xyz`, hull length for the Reynolds number in `.w`.
    linear_velocity: vec4<f32>,
    /// Angular velocity in `.xyz`, total hull surface area in `.w`.
    angular_velocity: vec4<f32>,
    /// First voxel, voxel count, first surface patch and patch count.
    ranges: vec4<u32>,
}

struct BodyForce {
    /// Force in newtons in `.xyz`, submerged volume in cubic meters in `.w`.
    force: vec4<f32>,
    /// Torque about the centre of mass in newton meters in `.xyz`, wetted area in `.w`.
    torque: vec4<f32>,
}

struct ForceOutput {
    /// Frame id and body count.
    header: vec4<u32>,
    bodies: array<BodyForce>,
}

struct ForceParams {
    gravity: f32,
    density: f32,
    surface_y: f32,
    dt: f32,
    kinematic_viscosity: f32,
    pressure_linear: f32,
    pressure_quadratic: f32,
    pressure_falloff: f32,
    suction_linear: f32,
    suction_quadratic: f32,
    suction_falloff: f32,
    slam_rate_max: f32,
    slam_power: f32,
    body_count: u32,
    frame: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: ForceParams;
/// Local voxel centre in `.xyz`, hull volume inside it in `.w`.
@group(0) @binding(1) var<storage, read> voxels: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> patches: array<Patch>;
@group(0) @binding(3) var<storage, read> bodies: array<Body>;
/// Per surface patch, the wetted area times normal speed of the previous frame, for slamming.
@group(0) @binding(4) var<storage, read_write> patch_state: array<f32>;
@group(0) @binding(5) var<storage, read_write> output: ForceOutput;
@group(0) @binding(6) var<uniform> lod: WaterLodSlices;
@group(0) @binding(7) var lod_displacement: texture_2d_array<f32>;
@group(0) @binding(8) var lod_foam: texture_2d_array<f32>;
@group(0) @binding(9) var lod_flow: texture_2d_array<f32>;

var<workgroup> shared_force: array<vec4<f32>, WORKGROUP>;
var<workgroup> shared_torque: array<vec4<f32>, WORKGROUP>;

fn rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

struct Water {
    height: f32,
    velocity: vec3<f32>,
}

/// The water surface above world `target_xz` and the water velocity there.
fn water_at(target_xz: vec2<f32>) -> Water {
    var xz = target_xz;
    for (var step = 0u; step < INVERT_STEPS; step++) {
        let dx = sample_lod_finest(lod_displacement, lod, xz).a;
        let dz = sample_lod_finest(lod_foam, lod, xz).b;
        xz = target_xz - vec2<f32>(dx, dz);
    }
    let height = params.surface_y + sample_lod_finest(lod_displacement, lod, xz).b;
    // Flow lanes are velocity along X, along Z, then vertical.
    let flow = sample_lod_finest(lod_flow, lod, xz);
    return Water(height, vec3<f32>(flow.x, flow.z, flow.y));
}

/// ITTC 1957 friction coefficient. Reynolds numbers below 10⁵ use 10⁵, where the line is still
/// meaningful.
fn ittc_friction(reynolds: f32) -> f32 {
    let l = log2(max(reynolds, 1.0e5)) / log2(10.0) - 2.0;
    return 0.075 / (l * l);
}

@compute @workgroup_size(64, 1, 1)
fn body_forces(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
) {
    let b = group.x;
    if b == 0u && lane == 0u {
        output.header = vec4<u32>(params.frame, params.body_count, 0u, 0u);
    }
    if b >= params.body_count {
        return;
    }
    let body = bodies[b];
    let s = body.position.w;
    let com = body.center_of_mass.xyz;
    let mass = body.center_of_mass.w;
    let total_area = max(body.angular_velocity.w, 1.0e-6);
    let rho = params.density;

    var force = vec3<f32>(0.0);
    var torque = vec3<f32>(0.0);
    var submerged = 0.0;
    var wetted = 0.0;
    // Buoyancy from each voxel's submerged part, at that part's centroid.
    for (var i = lane; i < body.ranges.y; i += WORKGROUP) {
        let voxel = voxels[body.ranges.x + i];
        let p = body.position.xyz + rotate(body.rotation, voxel.xyz);
        let water = water_at(p.xz);
        // The voxel's share below the surface, treating it as an upright cube.
        let bottom = p.y - 0.5 * s;
        let fraction = clamp((water.height - bottom) / s, 0.0, 1.0);
        if fraction <= 0.0 {
            continue;
        }
        let volume = voxel.w * fraction;
        let lift = vec3<f32>(0.0, rho * params.gravity * volume, 0.0);
        let at = vec3<f32>(p.x, bottom + 0.5 * fraction * s, p.z);
        force += lift;
        torque += cross(at - com, lift);
        submerged += volume;
    }

    // Friction, pressure drag and slamming on the wetted surface.
    for (var i = lane; i < body.ranges.w; i += WORKGROUP) {
        let index = body.ranges.z + i;
        let piece = patches[index];
        let p = body.position.xyz + rotate(body.rotation, piece.position_area.xyz);
        let water = water_at(p.xz);
        // The patch is wet in proportion to how much of a voxel-high column around it is.
        let fraction = clamp((water.height - (p.y - 0.5 * s)) / s, 0.0, 1.0);
        let area = piece.position_area.w * fraction;
        if area <= 0.0 {
            patch_state[index] = 0.0;
            continue;
        }
        wetted += area;
        let n = rotate(body.rotation, piece.normal.xyz);
        let point_velocity = body.linear_velocity.xyz + cross(body.angular_velocity.xyz, p - com);
        let relative = point_velocity - water.velocity;
        let speed = length(relative);
        let normal_speed = dot(relative, n);
        var f = vec3<f32>(0.0);

        // Friction along the hull.
        let tangential = relative - normal_speed * n;
        let slip = length(tangential);
        let friction = ittc_friction(slip * body.linear_velocity.w / params.kinematic_viscosity);
        f -= 0.5 * rho * friction * area * slip * tangential;

        // Pressure drag on surface moving into the water, suction on surface moving away.
        if speed > 1.0e-4 {
            let cos_theta = normal_speed / speed;
            if cos_theta > 0.0 {
                let magnitude = params.pressure_linear * speed + params.pressure_quadratic * speed * speed;
                f -= magnitude * area * pow(cos_theta, params.pressure_falloff) * n;
            } else {
                let magnitude = params.suction_linear * speed + params.suction_quadratic * speed * speed;
                f += magnitude * area * pow(-cos_theta, params.suction_falloff) * n;
            }
        }

        // Slamming: the growth rate of wetted area times normal speed, against the body's share
        // of momentum that this surface would have to stop within a frame.
        let flux = area * max(normal_speed, 0.0);
        let rate = max(flux - patch_state[index], 0.0) / (total_area * max(params.dt, 1.0e-4));
        patch_state[index] = flux;
        if normal_speed > 0.0 && params.slam_rate_max > 0.0 {
            let strength = pow(clamp(rate / params.slam_rate_max, 0.0, 1.0), params.slam_power);
            let stop = mass * point_velocity * (2.0 * area / total_area) / max(params.dt, 1.0e-4);
            f -= strength * (normal_speed / max(speed, 1.0e-4)) * stop;
        }

        force += f;
        torque += cross(p - com, f);
    }

    shared_force[lane] = vec4<f32>(force, submerged);
    shared_torque[lane] = vec4<f32>(torque, wetted);
    workgroupBarrier();
    for (var stride = WORKGROUP / 2u; stride > 0u; stride /= 2u) {
        if lane < stride {
            shared_force[lane] += shared_force[lane + stride];
            shared_torque[lane] += shared_torque[lane + stride];
        }
        workgroupBarrier();
    }
    if lane == 0u {
        output.bodies[b] = BodyForce(shared_force[0], shared_torque[0]);
    }
}
