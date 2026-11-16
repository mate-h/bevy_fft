// Water particles: a fixed ring pool in world space. `spawnParticles` writes this frame's spawns
// at an atomic cursor, overwriting the oldest slots, and `simulateParticles` moves every live
// particle under gravity and quadratic air drag. A particle that falls through the water surface
// in the LOD data dies there and leaves active foam in the foam sim's deposit buffer, and one past
// the LOD data's reach dies too. The state counters satisfy spawned - died = alive after each
// frame.

import bevy_fft::water::lod::{WaterLodSlices, lod_coverage, lod_uv, sample_lod_finest};

struct Particle {
    position: vec3<f32>,
    /// Quad edge in meters.
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
    _pad: vec3<u32>,
}

@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<storage, read_write> state: ParticleState;
@group(0) @binding(2) var<storage, read> spawns: array<Particle>;
@group(0) @binding(3) var<uniform> params: ParticleParams;
@group(0) @binding(4) var<uniform> lod: WaterLodSlices;
/// Slopes in `.rg`, height in `.b` and horizontal displacement X in `.a`.
@group(0) @binding(5) var lod_displacement: texture_2d_array<f32>;
/// Horizontal displacement Z in `.b`.
@group(0) @binding(6) var lod_foam: texture_2d_array<f32>;
/// The foam sim's deposit, `WaterFoamDeposit`.
@group(0) @binding(7) var<storage, read_write> deposit: array<atomic<u32>>;

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

/// Clears the live count before the frame's simulation counts it again.
@compute @workgroup_size(1)
fn beginFrame() {
    atomicStore(&state.alive, 0u);
}

@compute @workgroup_size(64)
fn spawnParticles(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= params.spawn_count {
        return;
    }
    let slot = atomicAdd(&state.cursor, 1u) % params.capacity;
    if particles[slot].life > 0.0 {
        atomicAdd(&state.died, 1u);
    }
    particles[slot] = spawns[id.x];
    atomicAdd(&state.spawned, 1u);
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
    let relative = p.velocity - params.wind;
    let a = params.gravity - params.drag * length(relative) * relative;
    // Exact for a constant acceleration, so a drag-free arc matches the ballistic one.
    p.position += p.velocity * dt + 0.5 * a * dt * dt;
    p.velocity += a * dt;
    p.life -= dt;
    if params.collide != 0u && p.life > 0.0 {
        let coverage = lod_coverage(lod, p.position.xz);
        if coverage <= 0.0 {
            p.life = 0.0;
        } else if p.velocity.y < 0.0
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
    }
    particles[id.x] = p;
}
