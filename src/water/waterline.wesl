// Waterline mask for the water viewer.
//
// Each pixel stores how far its point on the camera's near plane lies above the displaced water
// surface, in meters. Negative values are under water, so the zero crossing is where the near
// plane cuts the surface. One extra invocation measures the same height at the eye and writes it
// to a small buffer the main world reads back.

import bevy_fft::water::lod::{WaterLodSlices, displaced_height};

struct WaterlineUniform {
    /// Maps clip space of the viewer to world space.
    world_from_clip: mat4x4<f32>,
    /// Eye position in `xyz`. `w` is the height of the water mesh plane.
    eye: vec4<f32>,
    /// Mask size in pixels.
    size: vec2<u32>,
    _pad: vec2<u32>,
}

@group(0) @binding(0) var<uniform> waterline: WaterlineUniform;
@group(0) @binding(1) var<uniform> lod: WaterLodSlices;
@group(0) @binding(2) var lod_displacement: texture_2d_array<f32>;
@group(0) @binding(3) var lod_foam: texture_2d_array<f32>;
/// Linear, clamped to edge.
@group(0) @binding(4) var lod_sampler: sampler;
@group(0) @binding(5) var mask: texture_storage_2d<r32float, write>;
/// Eye height above the surface in `x`, the surface height under the eye in `y`.
@group(0) @binding(6) var<storage, read_write> eye_out: vec4<f32>;

/// Height of the displaced water surface above world `xz`, matching what pixels of the water
/// surface material draw.
fn surface_height(xz: vec2<f32>) -> f32 {
    return waterline.eye.w + displaced_height(lod_displacement, lod_foam, lod_sampler, lod, xz);
}

@compute @workgroup_size(8, 8, 1)
fn waterline_mask(@builtin(global_invocation_id) gid: vec3<u32>) {
    if all(gid.xy == vec2<u32>(0u)) {
        let h = surface_height(waterline.eye.xz);
        eye_out = vec4<f32>(waterline.eye.y - h, h, 0.0, 0.0);
    }
    if any(gid.xy >= waterline.size) {
        return;
    }
    // Bevy uses reversed depth, so the near plane sits at clip depth one.
    let uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(waterline.size);
    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    let p = waterline.world_from_clip * vec4<f32>(ndc, 1.0, 1.0);
    let world = p.xyz / p.w;
    textureStore(mask, gid.xy, vec4<f32>(world.y - surface_height(world.xz), 0.0, 0.0, 0.0));
}
