// Writes the summed ocean FFT cascades into every water LOD slice.
//
// Each texel takes the cascades whose waves its texel width can resolve. A cascade fades out of a
// slice once its texels are too coarse for the shorter end of the cascade's band, so coarse slices
// hold only long waves and never alias.
//
// Flow is the change in displacement since the previous frame. Linear waves move each surface
// particle with the time derivative of its displacement, so this is the orbital velocity. It uses
// the physical chop, without the artistic choppiness factor.

import bevy_fft::ocean::sampling::bilinear_wrap;
import bevy_fft::water::lod::{WaterLodSlices, lod_chop_dir, lod_texel_world};

struct OceanLodWriteParams {
    /// World tile edge in meters for cascades 0 to 3.
    cascade_tile_sizes: vec4<f32>,
    cascade_count: u32,
    amplitude: f32,
    choppiness: f32,
    /// Water depth in meters for the foam depth lane.
    depth: f32,
    /// One over the simulated time since the previous frame, or zero without a previous frame.
    inv_dt: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

@group(0) @binding(0) var<uniform> params: OceanLodWriteParams;
@group(0) @binding(1) var<uniform> lod: WaterLodSlices;
@group(0) @binding(2) var displacement_out: texture_storage_2d_array<rgba32float, write>;
@group(0) @binding(3) var foam_out: texture_storage_2d_array<rgba32float, write>;
@group(0) @binding(4) var flow_out: texture_storage_2d_array<rgba32float, write>;
@group(0) @binding(5) var displacement_0: texture_2d<f32>;
@group(0) @binding(6) var displacement_1: texture_2d<f32>;
@group(0) @binding(7) var displacement_2: texture_2d<f32>;
@group(0) @binding(8) var displacement_3: texture_2d<f32>;
@group(0) @binding(9) var foam_0: texture_2d<f32>;
@group(0) @binding(10) var foam_1: texture_2d<f32>;
@group(0) @binding(11) var foam_2: texture_2d<f32>;
@group(0) @binding(12) var foam_3: texture_2d<f32>;
@group(0) @binding(13) var previous_0: texture_2d<f32>;
@group(0) @binding(14) var previous_1: texture_2d<f32>;
@group(0) @binding(15) var previous_2: texture_2d<f32>;
@group(0) @binding(16) var previous_3: texture_2d<f32>;

fn displacement_cascade(i: u32, uv: vec2<f32>) -> vec4<f32> {
    if i == 0u {
        return bilinear_wrap(displacement_0, uv);
    } else if i == 1u {
        return bilinear_wrap(displacement_1, uv);
    } else if i == 2u {
        return bilinear_wrap(displacement_2, uv);
    }
    return bilinear_wrap(displacement_3, uv);
}

fn previous_cascade(i: u32, uv: vec2<f32>) -> vec4<f32> {
    if i == 0u {
        return bilinear_wrap(previous_0, uv);
    } else if i == 1u {
        return bilinear_wrap(previous_1, uv);
    } else if i == 2u {
        return bilinear_wrap(previous_2, uv);
    }
    return bilinear_wrap(previous_3, uv);
}

fn foam_cascade(i: u32, uv: vec2<f32>) -> f32 {
    if i == 0u {
        return bilinear_wrap(foam_0, uv).r;
    } else if i == 1u {
        return bilinear_wrap(foam_1, uv).r;
    } else if i == 2u {
        return bilinear_wrap(foam_2, uv).r;
    }
    return bilinear_wrap(foam_3, uv).r;
}

/// Share of cascade `i` kept in a slice with texel width `texel`. A cascade's band runs down to
/// roughly a sixth of its tile, and a wave needs about four texels, so the cascade is kept fully
/// while `texel <= tile / 24` and gone by `tile / 8`.
fn cascade_weight(i: u32, texel: f32) -> f32 {
    let tile = params.cascade_tile_sizes[i];
    return 1.0 - smoothstep(tile / 24.0, tile / 8.0, texel);
}

@compute @workgroup_size(8, 8, 1)
fn ocean_lod_write(@builtin(global_invocation_id) gid: vec3<u32>) {
    let res = lod.resolution;
    let slice = gid.z;
    if gid.x >= res || gid.y >= res || slice >= lod.count {
        return;
    }
    let xz = lod_texel_world(lod, gid.xy, slice);
    let texel = lod.slices[slice].z;

    var disp = vec4<f32>(0.0);
    var change = vec4<f32>(0.0);
    var foam = 0.0;
    let count = min(params.cascade_count, 4u);
    for (var i = 0u; i < count; i++) {
        let uv = xz / params.cascade_tile_sizes[i];
        let w = cascade_weight(i, texel);
        if w > 0.0 {
            let now = displacement_cascade(i, uv);
            disp += now * w;
            change += (now - previous_cascade(i, uv)) * w;
        }
        foam = max(foam, foam_cascade(i, uv));
    }
    let amp = params.amplitude;
    let scaled = vec4<f32>(disp.rgb * amp, disp.a * amp * params.choppiness);
    textureStore(displacement_out, gid.xy, slice, scaled);
    textureStore(foam_out, gid.xy, slice, vec4<f32>(foam, params.depth, 0.0, 1.0));
    let velocity = change * (amp * params.inv_dt);
    let horizontal = velocity.a * lod_chop_dir(lod);
    textureStore(flow_out, gid.xy, slice, vec4<f32>(horizontal, velocity.b, 0.0));
}
