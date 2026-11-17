// Water particles: a fixed ring pool in world space. `spawnParticles` writes this frame's spawns
// at an atomic cursor, overwriting the oldest slots, and `simulateParticles` moves every live
// particle under gravity and quadratic air drag. A particle that falls through the water surface
// in the LOD data dies there and leaves active foam in the foam sim's deposit buffer, and one past
// the LOD data's reach dies too. Mist, marked by a negative size, drifts with its own drag and
// a share of gravity and never lands. The state counters satisfy spawned - died = alive after each
// frame.

import bevy_fft::water::lod::{
    WaterLodSlices, lod_coverage, lod_texel_world, lod_uv, sample_lod_finest,
};

struct Particle {
    position: vec3<f32>,
    /// Quad edge in meters. Negative marks mist, `-size` across.
    size: f32,
    velocity: vec3<f32>,
    /// Seconds left to live. Zero or less is a free slot.
    life: f32,
}

struct ParticleState {
    /// Next slot to write, modulo the capacity.
    cursor: atomic<u32>,
    spawned: atomic<u32>,
    died: atomic<u32>,
    /// Live particles after this frame's simulation.
    alive: atomic<u32>,
    /// Spray emitted from breaking crests this frame, for the per-frame cap.
    frame_spray: atomic<u32>,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

struct ParticleParams {
    gravity: vec3<f32>,
    dt: f32,
    /// Air drag acceleration per (m/s)², `a = -drag |v - wind| (v - wind)`.
    wind: vec3<f32>,
    drag: f32,
    capacity: u32,
    spawn_count: u32,
    /// Nonzero tests particles against the water surface.
    collide: u32,
    /// Height of the water surface mesh plane in meters.
    surface_y: f32,
    /// Active foam a landing particle leaves over its own footprint, capped at a whole texel.
    foam_per_drop: f32,
    /// Spray from breaking crests in particles per m² per second at full breaking. Zero is off.
    spray_rate: f32,
    /// Slices the crest emitter scans, finest first.
    spray_slices: u32,
    /// Most spray particles one frame emits.
    spray_cap: u32,
    /// Upward launch speed at full breaking, and the random speed added in any direction.
    spray_speed: f32,
    spray_spread: f32,
    spray_size: f32,
    spray_lifetime: f32,
    /// Changes every frame, for the emitter's random numbers.
    seed: u32,
    /// Distance from the viewer in meters at which crest spray thins to half. Zero keeps it
    /// everywhere.
    spray_half_distance: f32,
    /// Nonzero replaces the spray with one still marker per emitting texel, sized by its share.
    spray_debug: u32,
    /// Drag and share of gravity for mist, which never lands.
    mist_drag: f32,
    mist_gravity: f32,
    _pad: vec3<u32>,
}

@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<storage, read_write> state: ParticleState;
@group(0) @binding(2) var<storage, read> spawns: array<Particle>;
@group(0) @binding(3) var<uniform> params: ParticleParams;
@group(0) @binding(4) var<uniform> lod: WaterLodSlices;
/// Slopes in `.rg`, height in `.b` and horizontal displacement X in `.a`.
@group(0) @binding(5) var lod_displacement: texture_2d_array<f32>;
/// Breaking intensity this frame in `.r` and horizontal displacement Z in `.b`.
@group(0) @binding(6) var lod_foam: texture_2d_array<f32>;
/// The foam sim's deposit, `WaterFoamDeposit`.
@group(0) @binding(7) var<storage, read_write> deposit: array<atomic<u32>>;
/// Water velocity: horizontal in `.xy` and the rate the surface rises in `.z`.
@group(0) @binding(8) var lod_flow: texture_2d_array<f32>;
/// Particles other GPU passes queued since the last frame, such as floating bodies' splashes.
@group(0) @binding(9) var<storage, read_write> queue: ParticleQueue;
/// The slots the draw reads, after this frame's simulation.
@group(0) @binding(10) var<storage, read_write> draw: DrawList;

/// A non-indexed indirect draw of six vertices per live particle, then the live slots.
struct DrawList {
    vertex_count: atomic<u32>,
    instance_count: u32,
    first_vertex: u32,
    first_instance: u32,
    slots: array<u32>,
}

struct ParticleQueue {
    /// Particles appended. Past `QUEUE_CAPACITY` they were dropped.
    count: atomic<u32>,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    items: array<Particle>,
}

const QUEUE_CAPACITY: u32 = 4096u;

const FOAM_DEPOSIT_UNIT: f32 = 65536.0;

/// Height of the water surface above world `xz`, relative to the mesh plane. Two fixed-point
/// steps undo the horizontal displacement, which is enough for a collision test.
fn surface_height(xz: vec2<f32>, coverage: f32) -> f32 {
    var at = xz;
    for (var step = 0u; step < 2u; step++) {
        let dx = sample_lod_finest(lod_displacement, lod, at).a;
        let dz = sample_lod_finest(lod_foam, lod, at).b;
        at = xz - vec2<f32>(dx, dz) * coverage;
    }
    return sample_lod_finest(lod_displacement, lod, at).b * coverage;
}

/// Leaves foam where particle `p` landed, in every slice that holds its point. Each slice gets
/// the share of a texel the particle covers there.
fn deposit_foam(p: Particle) {
    let res = lod.resolution;
    for (var i = 0u; i < lod.count; i++) {
        let uv = lod_uv(lod, p.position.xz, i);
        if any(uv < vec2(0.0)) || any(uv >= vec2(1.0)) {
            continue;
        }
        let texel = vec2<u32>(uv * f32(res));
        let width = lod.slices[i].z;
        let share = min(p.size * p.size / (width * width), 1.0) * params.foam_per_drop;
        let amount = u32(round(share * FOAM_DEPOSIT_UNIT));
        if amount > 0u {
            atomicAdd(&deposit[(i * res + texel.y) * res + texel.x], amount);
        }
    }
}

/// Clears the per-frame counts.
@compute @workgroup_size(1)
fn beginFrame() {
    atomicStore(&state.alive, 0u);
    atomicStore(&state.frame_spray, 0u);
    atomicStore(&draw.vertex_count, 0u);
    draw.instance_count = 1u;
    draw.first_vertex = 0u;
    draw.first_instance = 0u;
}

/// Writes `p` at the next slot, counting the particle it overwrites as dead.
fn emit(p: Particle) {
    let slot = atomicAdd(&state.cursor, 1u) % params.capacity;
    if particles[slot].life > 0.0 {
        atomicAdd(&state.died, 1u);
    }
    particles[slot] = p;
    atomicAdd(&state.spawned, 1u);
}

fn hash(v: u32) -> u32 {
    var x = v;
    x ^= x >> 16u;
    x *= 0x7feb352du;
    x ^= x >> 15u;
    x *= 0x846ca68bu;
    return x ^ (x >> 16u);
}

/// A uniform number in [0, 1) from `state`, which it advances.
fn random(state: ptr<function, u32>) -> f32 {
    *state = hash(*state + 0x9e3779b9u);
    return f32(*state >> 8u) / 16777216.0;
}

/// Spawns the particles other passes queued.
@compute @workgroup_size(64)
fn spawnQueued(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= min(atomicLoad(&queue.count), QUEUE_CAPACITY) {
        return;
    }
    emit(queue.items[id.x]);
}

/// Empties the queue once it has been spawned.
@compute @workgroup_size(1)
fn clearQueue() {
    atomicStore(&queue.count, 0u);
}

/// Spray from breaking crests. One thread per texel of the finest `spray_slices` slices; a
/// texel a finer slice also covers is left to that slice, so nested slices emit each patch once.
/// A texel emits `spray_rate · area · dt` particles at full breaking, more where the surface rises
/// fast, rounded at random so the mean is exact. Each starts on the displaced surface with the
/// water's velocity plus an upward launch.
@compute @workgroup_size(8, 8, 1)
fn emitCrestSpray(@builtin(global_invocation_id) gid: vec3<u32>) {
    let res = lod.resolution;
    let slice = gid.z;
    if gid.x >= res || gid.y >= res || slice >= min(lod.count, params.spray_slices) {
        return;
    }
    let xz = lod_texel_world(lod, gid.xy, slice);
    if slice > 0u {
        let finer = lod_uv(lod, xz, slice - 1u);
        if all(finer >= vec2(0.0)) && all(finer < vec2(1.0)) {
            return;
        }
    }
    let breaking = saturate(textureLoad(lod_foam, gid.xy, slice, 0).r);
    if breaking <= 0.0 {
        return;
    }
    let flow = textureLoad(lod_flow, gid.xy, slice, 0);
    let width = lod.slices[slice].z;
    let rise = saturate(0.5 + 0.5 * flow.z / max(params.spray_speed, 1e-3));
    // Fewer, larger drops away from the viewer, at the centre of the finest slice: their count
    // falls with the falloff and their area grows by its inverse, so the spray covers the same
    // share of the surface.
    var falloff = 1.0;
    if params.spray_half_distance > 0.0 {
        let d = length(xz - lod.slices[0].xy) / params.spray_half_distance;
        falloff = 1.0 / (1.0 + d * d);
    }
    let potential = breaking * rise * falloff;
    let displacement = textureLoad(lod_displacement, gid.xy, slice, 0);
    let dz = textureLoad(lod_foam, gid.xy, slice, 0).b;
    if params.spray_debug != 0u {
        var marker: Particle;
        marker.position = vec3<f32>(
            xz.x + displacement.a,
            params.surface_y + displacement.b,
            xz.y + dz,
        );
        marker.size = width * sqrt(potential);
        marker.life = 1.5 * params.dt;
        emit(marker);
        return;
    }
    let mean = params.spray_rate * width * width * params.dt * potential;
    let grow = inverseSqrt(falloff);
    var rng = hash(((slice * res + gid.y) * res + gid.x) ^ params.seed);
    var count = u32(floor(mean));
    if random(&rng) < fract(mean) {
        count += 1u;
    }
    if count == 0u {
        return;
    }
    let first = atomicAdd(&state.frame_spray, count);
    if first >= params.spray_cap {
        return;
    }
    count = min(count, params.spray_cap - first);
    for (var k = 0u; k < count; k++) {
        var p: Particle;
        let jitter = vec2<f32>(random(&rng), random(&rng)) - 0.5;
        p.position = vec3<f32>(
            xz.x + displacement.a + jitter.x * width,
            params.surface_y + displacement.b,
            xz.y + dz + jitter.y * width,
        );
        var cone = vec3<f32>(random(&rng), random(&rng), random(&rng)) * 2.0 - 1.0;
        cone /= max(length(cone), 1.0);
        p.velocity = vec3<f32>(flow.x, max(flow.z, 0.0), flow.y)
            + vec3<f32>(0.0, params.spray_speed * (0.5 + 0.5 * breaking), 0.0)
            + cone * params.spray_spread;
        p.size = params.spray_size * grow * (0.5 + random(&rng));
        p.life = params.spray_lifetime * (0.5 + random(&rng));
        emit(p);
    }
}

@compute @workgroup_size(64)
fn spawnParticles(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= params.spawn_count {
        return;
    }
    emit(spawns[id.x]);
}

@compute @workgroup_size(64)
fn simulateParticles(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= params.capacity {
        return;
    }
    var p = particles[id.x];
    if p.life <= 0.0 {
        return;
    }
    let dt = params.dt;
    let mist = p.size < 0.0;
    let relative = p.velocity - params.wind;
    var a = params.gravity - params.drag * length(relative) * relative;
    if mist {
        a = params.gravity * params.mist_gravity - params.mist_drag * length(relative) * relative;
    }
    // Exact for a constant acceleration, so a drag-free arc matches the ballistic one.
    p.position += p.velocity * dt + 0.5 * a * dt * dt;
    p.velocity += a * dt;
    p.life -= dt;
    if params.collide != 0u && p.life > 0.0 {
        let coverage = lod_coverage(lod, p.position.xz);
        if coverage <= 0.0 {
            p.life = 0.0;
        } else if !mist && p.velocity.y < 0.0
            && p.position.y <= params.surface_y + surface_height(p.position.xz, coverage) {
            deposit_foam(p);
            p.life = 0.0;
        }
    }
    if p.life <= 0.0 {
        p.life = 0.0;
        atomicAdd(&state.died, 1u);
    } else {
        atomicAdd(&state.alive, 1u);
        draw.slots[atomicAdd(&draw.vertex_count, 6u) / 6u] = id.x;
    }
    particles[id.x] = p;
}
