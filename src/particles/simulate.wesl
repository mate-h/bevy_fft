// Water particles: a fixed ring pool in world space. `spawnParticles` writes this frame's spawns
// at an atomic cursor, overwriting the oldest slots, and `simulateParticles` moves every live
// particle under gravity and quadratic air drag. The state counters satisfy
// spawned - died = alive after each frame.

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
    _pad: vec2<u32>,
}

@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<storage, read_write> state: ParticleState;
@group(0) @binding(2) var<storage, read> spawns: array<Particle>;
@group(0) @binding(3) var<uniform> params: ParticleParams;

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
    if p.life <= 0.0 {
        p.life = 0.0;
        atomicAdd(&state.died, 1u);
    } else {
        atomicAdd(&state.alive, 1u);
    }
    particles[id.x] = p;
}
