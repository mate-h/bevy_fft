// Writes the summed ocean FFT cascades into every water LOD slice.
//
// Each texel takes the cascades whose waves its texel width can resolve. A cascade fades out of a
// slice once its texels are too coarse for the shorter end of the cascade's band, so coarse slices
// hold only long waves and never alias.
//
// Horizontal displacement X comes from each cascade's FFT output and Z from its foam texture,
// which also keeps last frame's Z. Flow is the change in displacement since the previous frame.
// Linear waves move each surface particle with the time derivative of its displacement, so this is
// the orbital velocity. It uses the physical displacement, without the artistic choppiness factor.
// Drift is the cascades' summed Stokes drift, which the CPU integrates from the spectrum.
//
// Breaking follows Gilligan (Tessendorf 2017, §5.3.3, with the eigenvalues of course notes §4.6).
// A texel breaks where the smaller eigenvalue of the horizontal displacement Jacobian drops below
// `foam_cutoff`. The Jacobian sums the cascades marked in `cascade_breaking`, taken from the
// derivatives of each cascade's bilinear fetch, as Dupuy and Bruneton (2012, Eq. 4) sum their wave
// patterns. Short waves riding on long ones then break together. Short waves alone also fold, but
// their Jacobian barely depends on the wind, so leaving them out is what lets coverage follow the
// wind as Monahan and Woolf (1989) measured. Breaking cascades count in every slice, also those too
// coarse to show their waves, because foam coverage is a texel average and point samples of the
// Jacobian keep its statistics.

import bevy_fft::ocean::sampling::{
    BilinearSample, WrapTaps, bilinear_taps, bilinear_taps_with_gradient, wrap_taps,
};
import bevy_fft::water::lod::{WaterLodSlices, lod_texel_world};

struct OceanLodWriteParams {
    /// World tile edge in meters for cascades 0 to 3.
    cascade_tile_sizes: vec4<f32>,
    /// One for cascades whose waves count toward breaking, zero for the rest.
    cascade_breaking: vec4<f32>,
    cascade_count: u32,
    amplitude: f32,
    choppiness: f32,
    /// Water depth in meters for the foam depth lane.
    depth: f32,
    /// One over the simulated time since the previous frame, or zero without a previous frame.
    inv_dt: f32,
    _pad0: f32,
    /// Surface Stokes drift of all cascades in meters per second.
    stokes_drift: vec2<f32>,
    /// Smaller Jacobian eigenvalue below which waves break.
    foam_cutoff: f32,
    /// Width of the breaking transition in eigenvalue units.
    foam_falloff: f32,
    /// Threshold on the Jacobian determinant for far whitecaps.
    far_foam_cutoff: f32,
    /// Texel width where the far range takes over, fully at twice this width.
    far_foam_texel: f32,
}

@group(0) @binding(0) var<uniform> params: OceanLodWriteParams;
@group(0) @binding(1) var<uniform> lod: WaterLodSlices;
@group(0) @binding(2) var displacement_out: texture_storage_2d_array<rgba16float, write>;
@group(0) @binding(3) var foam_out: texture_storage_2d_array<rgba16float, write>;
@group(0) @binding(4) var flow_out: texture_storage_2d_array<rgba32float, write>;
@group(0) @binding(5) var drift_out: texture_storage_2d_array<rg32float, write>;
@group(0) @binding(6) var displacement_0: texture_2d<f32>;
@group(0) @binding(7) var displacement_1: texture_2d<f32>;
@group(0) @binding(8) var displacement_2: texture_2d<f32>;
@group(0) @binding(9) var displacement_3: texture_2d<f32>;
@group(0) @binding(10) var foam_0: texture_2d<f32>;
@group(0) @binding(11) var foam_1: texture_2d<f32>;
@group(0) @binding(12) var foam_2: texture_2d<f32>;
@group(0) @binding(13) var foam_3: texture_2d<f32>;
@group(0) @binding(14) var previous_0: texture_2d<f32>;
@group(0) @binding(15) var previous_1: texture_2d<f32>;
@group(0) @binding(16) var previous_2: texture_2d<f32>;
@group(0) @binding(17) var previous_3: texture_2d<f32>;
/// Each cascade's term of the Jacobian determinant in `.r` and its square in `.g`, with mips.
@group(0) @binding(18) var jacobian_0: texture_2d<f32>;
@group(0) @binding(19) var jacobian_1: texture_2d<f32>;
@group(0) @binding(20) var jacobian_2: texture_2d<f32>;
@group(0) @binding(21) var jacobian_3: texture_2d<f32>;
/// Trilinear and repeating.
@group(0) @binding(22) var jacobian_sampler: sampler;

/// Mean of cascade `i`'s determinant term and of its square over a footprint `level` mips wide.
fn jacobian_cascade(i: u32, uv: vec2<f32>, level: f32) -> vec2<f32> {
    if i == 0u {
        return textureSampleLevel(jacobian_0, jacobian_sampler, uv, level).rg;
    } else if i == 1u {
        return textureSampleLevel(jacobian_1, jacobian_sampler, uv, level).rg;
    } else if i == 2u {
        return textureSampleLevel(jacobian_2, jacobian_sampler, uv, level).rg;
    }
    return textureSampleLevel(jacobian_3, jacobian_sampler, uv, level).rg;
}

/// Error function, after Abramowitz and Stegun 7.1.26, with absolute error below 1.5e-7.
fn erf(x: f32) -> f32 {
    let t = 1.0 / (1.0 + 0.3275911 * abs(x));
    let poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741
        + t * (-1.453152027 + t * 1.061405429))));
    return sign(x) * (1.0 - poly * exp(-x * x));
}

fn displacement_cascade(i: u32, t: WrapTaps) -> BilinearSample {
    if i == 0u {
        return bilinear_taps_with_gradient(displacement_0, t);
    } else if i == 1u {
        return bilinear_taps_with_gradient(displacement_1, t);
    } else if i == 2u {
        return bilinear_taps_with_gradient(displacement_2, t);
    }
    return bilinear_taps_with_gradient(displacement_3, t);
}

fn previous_cascade(i: u32, t: WrapTaps) -> vec4<f32> {
    if i == 0u {
        return bilinear_taps(previous_0, t);
    } else if i == 1u {
        return bilinear_taps(previous_1, t);
    } else if i == 2u {
        return bilinear_taps(previous_2, t);
    }
    return bilinear_taps(previous_3, t);
}

/// Horizontal displacement Z in `.g` and last frame's displacement Z in `.b`.
fn foam_cascade(i: u32, t: WrapTaps) -> BilinearSample {
    if i == 0u {
        return bilinear_taps_with_gradient(foam_0, t);
    } else if i == 1u {
        return bilinear_taps_with_gradient(foam_1, t);
    } else if i == 2u {
        return bilinear_taps_with_gradient(foam_2, t);
    }
    return bilinear_taps_with_gradient(foam_3, t);
}

/// Size of cascade `i`. The displacement, previous and foam textures of a cascade share it.
fn cascade_dims(i: u32) -> vec2<i32> {
    if i == 0u {
        return vec2<i32>(textureDimensions(displacement_0, 0));
    } else if i == 1u {
        return vec2<i32>(textureDimensions(displacement_1, 0));
    } else if i == 2u {
        return vec2<i32>(textureDimensions(displacement_2, 0));
    }
    return vec2<i32>(textureDimensions(displacement_3, 0));
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
    var dz = 0.0;
    var dz_change = 0.0;
    // Unscaled derivatives dDx/dx, dDx/dz, dDz/dx and dDz/dz in meters per meter.
    var jacobian = vec4<f32>(0.0);
    // Mean and variance of the Jacobian determinant over the texel, after Dupuy and Bruneton
    // (2012, Eqs. 7 and 8). The cascades are independent, so the variance is the sum of each
    // cascade's own variance.
    var det_mean = 1.0;
    var det_variance = 0.0;
    let count = min(params.cascade_count, 4u);
    for (var i = 0u; i < count; i++) {
        let dims = cascade_dims(i);
        let taps = wrap_taps(dims, xz / params.cascade_tile_sizes[i]);
        let w = cascade_weight(i, texel);
        let d = displacement_cascade(i, taps);
        let f = foam_cascade(i, taps);
        let per_meter = vec2<f32>(dims) / params.cascade_tile_sizes[i];
        let breaking = params.cascade_breaking[i];
        jacobian += vec4<f32>(d.d_dx.a, d.d_dy.a, f.d_dx.g, f.d_dy.g) * per_meter.xyxy * breaking;
        let level = log2(max(texel * per_meter.x, 1.0));
        let stats = jacobian_cascade(i, xz / params.cascade_tile_sizes[i], level) * breaking;
        det_mean += stats.x;
        det_variance += max(stats.y - stats.x * stats.x, 0.0);
        if w > 0.0 {
            disp += d.value * w;
            change += (d.value - previous_cascade(i, taps)) * w;
            dz += f.value.g * w;
            dz_change += (f.value.g - f.value.b) * w;
        }
    }
    let amp = params.amplitude;
    let chop = amp * params.choppiness;
    let jxx = 1.0 + chop * jacobian.x;
    let jzz = 1.0 + chop * jacobian.w;
    let half_trace = 0.5 * (jxx + jzz);
    let det = jxx * jzz - chop * chop * jacobian.y * jacobian.z;
    let lambda_min = half_trace - sqrt(max(half_trace * half_trace - det, 0.0));
    let near = saturate((params.foam_cutoff - lambda_min) / max(params.foam_falloff, 1e-5));
    // Dupuy and Bruneton (2012, Eq. 6): the share of the texel whose determinant falls below the
    // threshold, for a normally distributed determinant.
    let sigma = sqrt(max(det_variance, 1e-10));
    let far = 0.5 + 0.5 * erf((params.far_foam_cutoff - det_mean) / (sqrt(2.0) * sigma));
    let far_weight = smoothstep(params.far_foam_texel, 2.0 * params.far_foam_texel, texel);
    let foam = mix(near, far, far_weight);
    let scaled = vec4<f32>(disp.rgb * amp, disp.a * chop);
    textureStore(displacement_out, gid.xy, slice, scaled);
    textureStore(foam_out, gid.xy, slice, vec4<f32>(foam, params.depth, dz * chop, 1.0));
    let rate = amp * params.inv_dt;
    let velocity = vec3<f32>(change.a, dz_change, change.b) * rate;
    textureStore(flow_out, gid.xy, slice, vec4<f32>(velocity, 0.0));
    textureStore(drift_out, gid.xy, slice, vec4<f32>(params.stokes_drift, 0.0, 0.0));
}
