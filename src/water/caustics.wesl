// Caustic map on a few level receivers under the water, one array layer each.
//
// `caustics_splat` sends a grid of sun rays through one LOD slice for each receiver. Each ray
// refracts at the displaced surface by Snell's law, runs down to the receiver, and adds its share
// of the light to the four receiver texels around its hit, weighted bilinearly. `caustics_resolve` turns the sums
// into intensity relative to a level surface, so the mean stays one and focused light shows as the
// area ratio between the cells the rays leave and the cells they land in.

import bevy_fft::water::lod::WaterLodSlices;

struct CausticsUniform {
    /// Unit vector toward the sun.
    direction_to_light: vec3<f32>,
    receiver_count: u32,
    /// Depth of each receiver below the mesh plane, in meters.
    receiver_depths: vec4<f32>,
    /// LOD slice the map covers.
    slice: u32,
    /// Rays along each edge of a texel.
    rays: u32,
    ior: f32,
    _pad: f32,
}

@group(0) @binding(0) var<uniform> caustics: CausticsUniform;
@group(0) @binding(1) var<uniform> lod: WaterLodSlices;
@group(0) @binding(2) var lod_displacement: texture_2d_array<f32>;
@group(0) @binding(3) var lod_foam: texture_2d_array<f32>;
/// Linear, clamped to edge.
@group(0) @binding(4) var lod_sampler: sampler;
/// Light gathered per receiver texel, in units of `ENERGY_SCALE` per texel's worth.
@group(0) @binding(5) var<storage, read_write> energy: array<atomic<u32>>;
@group(0) @binding(6) var map: texture_storage_2d_array<r32float, write>;

/// Fixed-point scale for one texel's worth of light.
const ENERGY_SCALE: f32 = 65536.0;

fn deposit(texel: vec2<i32>, receiver: u32, amount: f32) {
    let n = i32(lod.resolution);
    if any(texel < vec2<i32>(0)) || any(texel >= vec2<i32>(n)) || amount <= 0.0 {
        return;
    }
    let layer = receiver * lod.resolution * lod.resolution;
    atomicAdd(&energy[layer + u32(texel.y * n + texel.x)], u32(amount * ENERGY_SCALE + 0.5));
}

@compute @workgroup_size(8, 8, 1)
fn caustics_splat(@builtin(global_invocation_id) gid: vec3<u32>) {
    let rays_per_edge = lod.resolution * caustics.rays;
    let receiver = gid.z;
    if any(gid.xy >= vec2<u32>(rays_per_edge)) || receiver >= caustics.receiver_count {
        return;
    }
    let s = lod.slices[caustics.slice];
    // Ray start in texel units across the slice.
    let start = (vec2<f32>(gid.xy) + 0.5) / f32(caustics.rays);
    let uv = start / f32(lod.resolution);
    let surface = textureSampleLevel(lod_displacement, lod_sampler, uv, caustics.slice, 0.0);
    let dz = textureSampleLevel(lod_foam, lod_sampler, uv, caustics.slice, 0.0).b;
    let n = normalize(vec3<f32>(-surface.r, 1.0, -surface.g));
    let refracted = refract(-caustics.direction_to_light, n, 1.0 / caustics.ior);
    if refracted.y >= 0.0 {
        return;
    }
    // Horizontal travel from the displaced surface point down to the receiver, in texels.
    let drop = surface.b + caustics.receiver_depths[receiver];
    let travel = (vec2<f32>(surface.a, dz) + refracted.xz * drop / -refracted.y) / s.z;
    let hit = start + travel - 0.5;
    let base = vec2<i32>(floor(hit));
    let f = hit - floor(hit);
    let share = 1.0 / f32(caustics.rays * caustics.rays);
    deposit(base, receiver, share * (1.0 - f.x) * (1.0 - f.y));
    deposit(base + vec2<i32>(1, 0), receiver, share * f.x * (1.0 - f.y));
    deposit(base + vec2<i32>(0, 1), receiver, share * (1.0 - f.x) * f.y);
    deposit(base + vec2<i32>(1, 1), receiver, share * f.x * f.y);
}

@compute @workgroup_size(8, 8, 1)
fn caustics_resolve(@builtin(global_invocation_id) gid: vec3<u32>) {
    if any(gid.xy >= vec2<u32>(lod.resolution)) || gid.z >= caustics.receiver_count {
        return;
    }
    let i = (gid.z * lod.resolution + gid.y) * lod.resolution + gid.x;
    let intensity = f32(atomicLoad(&energy[i])) / ENERGY_SCALE;
    textureStore(map, gid.xy, gid.z, vec4<f32>(intensity, 0.0, 0.0, 0.0));
}
