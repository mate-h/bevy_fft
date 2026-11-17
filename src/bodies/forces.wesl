// Water forces on voxelized floating bodies. One workgroup per body loops over its voxels, samples
// the water LOD data at each one, and sums buoyancy, and separately friction, pressure drag and
// slamming, into forces and torques about the body's centre of mass. They stay apart so the CPU
// can correct each for the frames the readback takes in its own way.
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
    /// 1 in `.x` when the body presses the water of shallow water grids down.
    flags: vec4<u32>,
}

struct BodyForce {
    /// Buoyancy in newtons in `.xyz`, submerged volume in cubic meters in `.w`.
    buoyancy: vec4<f32>,
    /// Torque of buoyancy about the centre of mass in `.xyz`, waterplane area in square meters in
    /// `.w`: how fast the submerged volume grows per meter of sinking.
    buoyancy_torque: vec4<f32>,
    /// Friction, pressure drag and slamming in newtons in `.xyz`, wetted area in `.w`.
    drag: vec4<f32>,
    /// Torque of those about the centre of mass in `.xyz`.
    drag_torque: vec4<f32>,
    /// Water velocity averaged over the wetted surface by area, in `.xyz`.
    water_velocity: vec4<f32>,
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
    /// Splash particles per m² of entering waterline surface per m/s above `splash_min_speed`
    /// per second. Zero is off.
    splash_rate: f32,
    splash_min_speed: f32,
    splash_size: f32,
    splash_lifetime: f32,
    /// Random share of the entry speed added in any direction.
    splash_spread: f32,
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

struct Particle {
    position: vec3<f32>,
    size: f32,
    velocity: vec3<f32>,
    life: f32,
}

/// `WaterParticleQueue`: splashes go to the water particles for their next frame.
struct ParticleQueue {
    count: atomic<u32>,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    items: array<Particle>,
}

const QUEUE_CAPACITY: u32 = 4096u;

@group(0) @binding(10) var<storage, read_write> particle_queue: ParticleQueue;

fn hash(v: u32) -> u32 {
    var x = v;
    x ^= x >> 16u;
    x *= 0x7feb352du;
    x ^= x >> 15u;
    x *= 0x846ca68bu;
    return x ^ (x >> 16u);
}

fn random(state: ptr<function, u32>) -> f32 {
    *state = hash(*state + 0x9e3779b9u);
    return f32(*state >> 8u) / 16777216.0;
}

/// Throws water off a surface patch at the waterline that moves into the water at
/// `normal_speed` along its outward normal `n`: up and out along the hull, with the water's
/// velocity, in proportion to the patch's wetted area and how much faster than the threshold it
/// enters.
fn splash(index: u32, at: vec3<f32>, n: vec3<f32>, area: f32, normal_speed: f32, water: vec3<f32>) {
    let excess = normal_speed - params.splash_min_speed;
    if params.splash_rate <= 0.0 || excess <= 0.0 {
        return;
    }
    let mean = params.splash_rate * area * excess * params.dt;
    var rng = hash(index ^ hash(params.frame));
    var count = u32(floor(mean));
    if random(&rng) < fract(mean) {
        count += 1u;
    }
    if count == 0u {
        return;
    }
    let first = atomicAdd(&particle_queue.count, count);
    let out = normalize(vec3<f32>(n.x, 0.0, n.z) + vec3<f32>(1.0e-4, 0.0, 0.0));
    for (var k = 0u; k < count; k++) {
        let slot = first + k;
        if slot >= QUEUE_CAPACITY {
            return;
        }
        var cone = vec3<f32>(random(&rng), random(&rng), random(&rng)) * 2.0 - 1.0;
        cone /= max(length(cone), 1.0);
        let launch = normal_speed * (0.5 + random(&rng));
        var p: Particle;
        p.position = at;
        p.velocity = water + out * 0.6 * launch + vec3<f32>(0.0, launch, 0.0)
            + cone * params.splash_spread * normal_speed;
        p.size = params.splash_size * (0.5 + random(&rng));
        p.life = params.splash_lifetime * (0.5 + random(&rng));
        particle_queue.items[slot] = p;
    }
}

var<workgroup> shared_buoyancy: array<vec4<f32>, WORKGROUP>;
var<workgroup> shared_buoyancy_torque: array<vec4<f32>, WORKGROUP>;
var<workgroup> shared_drag: array<vec4<f32>, WORKGROUP>;
var<workgroup> shared_drag_torque: array<vec4<f32>, WORKGROUP>;
var<workgroup> shared_water: array<vec4<f32>, WORKGROUP>;

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
    // Flow lanes are velocity along X, along Z, vertical, then the pressure head that bodies
    // press the surface down by. Adding it back gives the surface the water would have without
    // the bodies, so a body doesn't sink into its own dent.
    let flow = sample_lod_finest(lod_flow, lod, xz);
    let height = params.surface_y + sample_lod_finest(lod_displacement, lod, xz).b + flow.w;
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

    var buoyancy = vec3<f32>(0.0);
    var buoyancy_torque = vec3<f32>(0.0);
    var drag = vec3<f32>(0.0);
    var drag_torque = vec3<f32>(0.0);
    var water_flux = vec3<f32>(0.0);
    var submerged = 0.0;
    var wetted = 0.0;
    var waterplane = 0.0;
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
        if fraction < 1.0 {
            waterplane += voxel.w / s;
        }
        let lift = vec3<f32>(0.0, rho * params.gravity * volume, 0.0);
        let at = vec3<f32>(p.x, bottom + 0.5 * fraction * s, p.z);
        buoyancy += lift;
        buoyancy_torque += cross(at - com, lift);
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
        water_flux += water.velocity * area;
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
        // Splashes from surface at the waterline that moves into the water.
        if fraction < 1.0 {
            splash(index, vec3<f32>(p.x, water.height, p.z), n, area, normal_speed, water.velocity);
        }

        let flux = area * max(normal_speed, 0.0);
        let rate = max(flux - patch_state[index], 0.0) / (total_area * max(params.dt, 1.0e-4));
        patch_state[index] = flux;
        if normal_speed > 0.0 && params.slam_rate_max > 0.0 {
            let strength = pow(clamp(rate / params.slam_rate_max, 0.0, 1.0), params.slam_power);
            let stop = mass * point_velocity * (2.0 * area / total_area) / max(params.dt, 1.0e-4);
            f -= strength * (normal_speed / max(speed, 1.0e-4)) * stop;
        }

        drag += f;
        drag_torque += cross(p - com, f);
    }

    shared_buoyancy[lane] = vec4<f32>(buoyancy, submerged);
    shared_buoyancy_torque[lane] = vec4<f32>(buoyancy_torque, waterplane);
    shared_drag[lane] = vec4<f32>(drag, wetted);
    shared_drag_torque[lane] = vec4<f32>(drag_torque, 0.0);
    shared_water[lane] = vec4<f32>(water_flux, 0.0);
    workgroupBarrier();
    for (var stride = WORKGROUP / 2u; stride > 0u; stride /= 2u) {
        if lane < stride {
            shared_buoyancy[lane] += shared_buoyancy[lane + stride];
            shared_buoyancy_torque[lane] += shared_buoyancy_torque[lane + stride];
            shared_drag[lane] += shared_drag[lane + stride];
            shared_drag_torque[lane] += shared_drag_torque[lane + stride];
            shared_water[lane] += shared_water[lane + stride];
        }
        workgroupBarrier();
    }
    if lane == 0u {
        let wetted_total = shared_drag[0].w;
        let water = select(vec3<f32>(0.0), shared_water[0].xyz / wetted_total, wetted_total > 0.0);
        output.bodies[b] = BodyForce(
            shared_buoyancy[0],
            shared_buoyancy_torque[0],
            shared_drag[0],
            shared_drag_torque[0],
            vec4<f32>(water, 0.0),
        );
    }
}

// Footprints of the bodies on one shallow water grid, as a pressure head (Kellomäki 2014): the
// volume each body has below the surface plus the head, spread over the cells under it, divided
// by the cell area. At rest the water under a body meets its hull and the head is its draft. `body_footprints` adds each voxel's submerged volume into its cell with integer atomics,
// and `resolve_pressure` turns the sums into the head.

struct Grid {
    /// World X and Z of the grid centre in meters.
    center: vec2<f32>,
    cells: vec2<u32>,
    cell_size: f32,
    /// Meters per unit of the grid's heights.
    height_scale: f32,
    /// Share of the way the head moves toward its new value each frame.
    relaxation: f32,
    /// 1 when `obstruction_out` is the grid's own map, 0 for the placeholder.
    has_obstruction: u32,
}

/// Fixed-point unit of the waterline area sums, in square meters.
const AREA_UNIT: f32 = 1.0e-4;
/// Fixed-point unit of the volume sums, in cubic meters.
const VOLUME_UNIT: f32 = 1.0e-6;

@group(1) @binding(0) var<uniform> grid: Grid;
@group(1) @binding(1) var<storage, read_write> displaced: array<atomic<u32>>;
@group(1) @binding(2) var pressure_out: texture_storage_2d<r32float, write>;
/// The head of the previous frame per cell, in the grid's height units.
@group(1) @binding(3) var<storage, read_write> previous_head: array<f32>;
/// Obstruction per cell (Gilligan §3.3): 1 in open water, 0 where a hull fills the cell at the
/// waterline. Waves multiplied by it reflect off hulls.
@group(1) @binding(4) var obstruction_out: texture_storage_2d<r32float, write>;

@compute @workgroup_size(64, 1, 1)
fn body_footprints(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
) {
    let b = group.x;
    if b >= params.body_count {
        return;
    }
    let body = bodies[b];
    if body.flags.x == 0u {
        return;
    }
    let s = body.position.w;
    let n = vec2<f32>(grid.cells);
    for (var i = lane; i < body.ranges.y; i += WORKGROUP) {
        let voxel = voxels[body.ranges.x + i];
        let p = body.position.xyz + rotate(body.rotation, voxel.xyz);
        let g = (p.xz - grid.center) / grid.cell_size + 0.5 * n;
        if any(g < vec2(0.0)) || any(g >= n) {
            continue;
        }
        // A grid with an obstruction map holds the surface plus the head at its open level in
        // every cell a hull fills, so the head there is the hull's draft below that level. Read it
        // at the cell's centre: at a voxel's own position the bilinear surface mixes in the waves
        // in the open cells next to the hull, the head follows them, and the hull radiates.
        var at = p.xz;
        if grid.has_obstruction != 0u {
            at = grid.center + (floor(g) + 0.5 - 0.5 * n) * grid.cell_size;
        }
        let water = water_at(at);
        let fraction = clamp((water.height - (p.y - 0.5 * s)) / s, 0.0, 1.0);
        let volume = voxel.w * fraction;
        if volume <= 0.0 {
            continue;
        }
        let cell = vec2<u32>(g);
        let index = cell.y * grid.cells.x + cell.x;
        atomicAdd(&displaced[index], u32(round(volume / VOLUME_UNIT)));
        // A voxel the surface cuts through, or touches at its top, puts its horizontal area on the
        // waterline. The second half of the buffer sums that area per cell.
        if (water.height - (p.y - 0.5 * s)) / s <= 1.0 {
            let n_cells = grid.cells.x * grid.cells.y;
            atomicAdd(&displaced[n_cells + index], u32(round(s * s / AREA_UNIT)));
        }
    }
}

@compute @workgroup_size(8, 8, 1)
fn resolve_pressure(@builtin(global_invocation_id) gid: vec3<u32>) {
    if any(gid.xy >= grid.cells) {
        return;
    }
    let index = gid.y * grid.cells.x + gid.x;
    let volume = f32(atomicLoad(&displaced[index])) * VOLUME_UNIT;
    let target_head = volume / (grid.cell_size * grid.cell_size) / grid.height_scale;
    // The volume is measured against the surface plus last frame's head, which the head itself
    // just moved. Taken whole, the head swings between none and the full draft from frame to
    // frame, so it only moves part of the way.
    // The memory is kept per world cell, wrapped onto the grid, so a grid that moves by whole
    // cells, such as a dispersive patch that follows a boat, finds each cell's last head again.
    let origin = vec2<i32>(round(grid.center / grid.cell_size));
    let cells = vec2<i32>(grid.cells);
    let world = ((vec2<i32>(gid.xy) + origin) % cells + cells) % cells;
    let memory = u32(world.y) * grid.cells.x + u32(world.x);
    let head = mix(previous_head[memory], target_head, grid.relaxation);
    previous_head[memory] = head;
    textureStore(pressure_out, gid.xy, vec4<f32>(head, 0.0, 0.0, 0.0));
    if grid.has_obstruction != 0u {
        let n_cells = grid.cells.x * grid.cells.y;
        let area = f32(atomicLoad(&displaced[n_cells + index])) * AREA_UNIT;
        let open = 1.0 - clamp(area / (grid.cell_size * grid.cell_size), 0.0, 1.0);
        textureStore(obstruction_out, gid.xy, vec4<f32>(open, 0.0, 0.0, 0.0));
    }
}
