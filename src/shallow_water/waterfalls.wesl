// Spray at waterfall lips. One thread per cell checks the faces on its left and lower sides with
// the CMF10 waterfall test (Chentanez and Müller-Fischer 2010): water flows across the face from
// the higher cell, the bed falls by more than `slope_cap` cell widths across it, and the lower
// cell is not filled up to the higher cell's bed. Such a face throws `rate` particles per cubic
// meter that crosses it into the water particle queue, rounded at random so the mean is exact.
// The particles are spray for the eye: the grid still passes the water down the drop itself.

struct Params {
    size: vec2<u32>,
    /// World XZ of cell (0, 0)'s corner, and the cell width in meters.
    origin: vec2<f32>,
    cell_size: f32,
    /// World height of the solver's zero.
    datum: f32,
    dt: f32,
    slope_cap: f32,
    rate: f32,
    spread: f32,
    size_m: f32,
    lifetime: f32,
    seed: u32,
    _pad: vec3<u32>,
}

struct Particle {
    position: vec3<f32>,
    size: f32,
    velocity: vec3<f32>,
    life: f32,
}

struct ParticleQueue {
    count: atomic<u32>,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    items: array<Particle>,
}

const QUEUE_CAPACITY: u32 = 4096u;

@group(0) @binding(0) var<uniform> params: Params;
/// Bed in `.x` and depth in `.y` per cell.
@group(0) @binding(1) var bed_water: texture_2d<f32>;
/// Velocity of the faces across x, `size.x + 1` wide.
@group(0) @binding(2) var face_u: texture_2d<f32>;
/// Velocity of the faces across z, `size.y + 1` tall.
@group(0) @binding(3) var face_w: texture_2d<f32>;
@group(0) @binding(4) var<storage, read_write> queue: ParticleQueue;

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

/// Emits over the face between cells `a` and `b`, which lie along `axis` with `b = a + axis`,
/// carrying velocity `v` from `a` to `b`. `face` is the face's centre in cells.
fn lip(a: vec2<i32>, b: vec2<i32>, axis: vec2<f32>, v: f32, face: vec2<f32>, salt: u32) {
    var up = a;
    var down = b;
    var dir = axis;
    if v < 0.0 {
        up = b;
        down = a;
        dir = -axis;
    }
    let speed = abs(v);
    let high = textureLoad(bed_water, up, 0);
    let low = textureLoad(bed_water, down, 0);
    if high.y <= 1e-3 || speed <= 0.0 {
        return;
    }
    let drop = high.x - low.x;
    if drop <= params.slope_cap * params.cell_size || low.x + low.y >= high.x {
        return;
    }
    // Water over the lip this step: flux per width times the face width.
    let volume = speed * high.y * params.cell_size * params.dt;
    let mean = params.rate * volume;
    var rng = hash(salt ^ hash(params.seed));
    var count = u32(floor(mean));
    if random(&rng) < fract(mean) {
        count += 1u;
    }
    if count == 0u {
        return;
    }
    let first = atomicAdd(&queue.count, count);
    let across = vec2<f32>(-dir.y, dir.x);
    for (var k = 0u; k < count; k++) {
        let slot = first + k;
        if slot >= QUEUE_CAPACITY {
            return;
        }
        let along = (random(&rng) - 0.5) * params.cell_size;
        let xz = params.origin + face * params.cell_size + across * along;
        var jitter = vec3<f32>(random(&rng), random(&rng), random(&rng)) * 2.0 - 1.0;
        jitter /= max(length(jitter), 1.0);
        var p: Particle;
        p.position = vec3<f32>(xz.x, params.datum + high.x + high.y * random(&rng), xz.y);
        p.velocity = vec3<f32>(dir.x, 0.0, dir.y) * speed * (0.8 + 0.4 * random(&rng))
            + jitter * params.spread * speed;
        p.size = params.size_m * (0.5 + random(&rng));
        p.life = params.lifetime * (0.5 + random(&rng));
        queue.items[slot] = p;
    }
}

@compute @workgroup_size(16, 16, 1)
fn emitWaterfalls(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= params.size.x || id.y >= params.size.y || params.rate <= 0.0 {
        return;
    }
    let c = vec2<i32>(id.xy);
    let salt = (id.y * params.size.x + id.x) * 2u;
    if id.x >= 1u {
        let u = textureLoad(face_u, c, 0).r;
        lip(c - vec2(1, 0), c, vec2(1.0, 0.0), u, vec2<f32>(f32(id.x), f32(id.y) + 0.5), salt);
    }
    if id.y >= 1u {
        let w = textureLoad(face_w, c, 0).r;
        lip(c - vec2(0, 1), c, vec2(0.0, 1.0), w, vec2<f32>(f32(id.x) + 0.5, f32(id.y)), salt + 1u);
    }
}
