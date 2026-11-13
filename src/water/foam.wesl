// Persistent foam on the water LOD slices, carried by the drift lane and faded by half-lives.
//
// Foam is two quantities per texel: active foam over breaking crests in `.r` and the residual film
// it leaves behind in `.g`. Each frame `foam_forward` moves last frame's foam along the drift with
// a semi-Lagrangian lookup, and `foam_correct` applies the MacCormack correction, limits it to the
// values the lookup could see, and scales by the local area change exp(-dt div u), so foam piles up
// where the drift converges. It then adds foam and decays both quantities with the exact solution
// of da/dt = b / t_f - la a, dr/dt = k la a - lr r over the step, which makes the result
// independent of the frame rate. The breaking intensity `b` is what the writers stored in the LOD
// foam lane this frame, from 0 to 1.
//
// Last frame's foam sits on last frame's slice placement. A lookup that leaves its own slice there
// falls back to the next coarser slice, as Crest's foam sim does, and returns no foam past the
// coarsest one.
//
// `foam_flow` carries the advected texture coordinates of `foam_flow.wesl` the same way, with a
// plain semi-Lagrangian step, and restarts each phase when its latency runs out.

import bevy_fft::water::lod::{WaterLodSlices, bilinear_clamp_layer, lod_texel_world};
import bevy_fft::water::foam_flow::{
    FOAM_FLOW_LATENCY,
    FOAM_FLOW_LAYERS_PER_SLICE,
    foam_flow_age,
    foam_flow_seed,
};

struct WaterFoamParams {
    /// Slice placement the previous foam was written on.
    previous: WaterLodSlices,
    /// Simulated seconds since the previous frame.
    dt: f32,
    /// Share of active foam left after `dt`, 0.5^(dt / active half-life).
    active_decay: f32,
    /// Share of residual foam left after `dt`.
    residual_decay: f32,
    /// Residual foam gained per unit of active foam at the start of the step.
    residual_gain: f32,
    /// Active foam gained per unit of breaking intensity held over the step.
    source_active: f32,
    /// Residual foam gained per unit of breaking intensity held over the step.
    source_residual: f32,
    /// Wrapped seconds since startup at the end of the step, the surface material's time.
    time: f32,
    /// One on the first step, which gives every phase a fresh offset.
    restart: f32,
}

@group(0) @binding(0) var<uniform> params: WaterFoamParams;
@group(0) @binding(1) var<uniform> lod: WaterLodSlices;
/// Foam at the end of the previous frame, on `params.previous`.
@group(0) @binding(2) var previous_foam: texture_2d_array<f32>;
/// Result of `foam_forward`, on `lod`. Read by `foam_correct` only.
@group(0) @binding(3) var forward_foam: texture_2d_array<f32>;
/// LOD foam lane, whose `.r` holds the breaking intensity the writers produced this frame.
@group(0) @binding(4) var lod_foam: texture_2d_array<f32>;
/// LOD drift lane in meters per second along X and Z.
@group(0) @binding(5) var lod_drift: texture_2d_array<f32>;
@group(0) @binding(6) var foam_out: texture_storage_2d_array<rg32float, write>;
/// Advected texture coordinates and deformation at the end of the previous frame, on
/// `params.previous`, with `FOAM_FLOW_LAYERS_PER_SLICE` layers per slice.
@group(0) @binding(7) var previous_flow: texture_2d_array<f32>;
@group(0) @binding(8) var flow_out: texture_storage_2d_array<rgba32float, write>;

fn slice_uv(slices: WaterLodSlices, xz: vec2<f32>, i: u32) -> vec2<f32> {
    let s = slices.slices[i];
    return (xz - s.xy) / s.w + 0.5;
}

/// Finest slice of `slices`, from `start` outward, whose interior holds a bilinear footprint at
/// `xz`, or -1 when none does.
fn slice_holding(slices: WaterLodSlices, xz: vec2<f32>, start: u32) -> i32 {
    let margin = 1.0 / f32(max(slices.resolution, 2u));
    for (var i = start; i < slices.count; i++) {
        let uv = slice_uv(slices, xz, i);
        if all(uv > vec2(margin)) && all(uv < vec2(1.0 - margin)) {
            return i32(i);
        }
    }
    return -1;
}

/// Previous foam bilinearly blended at a lookup, with the smallest and largest of the four texels
/// it blends.
struct PreviousLookup {
    foam: vec2<f32>,
    lo: vec2<f32>,
    hi: vec2<f32>,
}

/// Previous foam at world `xz`, looked up from slice `start` outward. Full-precision foam has no
/// hardware filtering on every backend, so this blends four loads.
fn previous_at(xz: vec2<f32>, start: u32) -> PreviousLookup {
    let i = slice_holding(params.previous, xz, start);
    if i < 0 {
        return PreviousLookup(vec2<f32>(0.0), vec2<f32>(0.0), vec2<f32>(0.0));
    }
    let x = slice_uv(params.previous, xz, u32(i)) * f32(params.previous.resolution) - 0.5;
    let i0 = vec2<i32>(floor(x));
    let f = x - floor(x);
    let a = textureLoad(previous_foam, i0, i, 0).rg;
    let b = textureLoad(previous_foam, i0 + vec2(1, 0), i, 0).rg;
    let c = textureLoad(previous_foam, i0 + vec2(0, 1), i, 0).rg;
    let d = textureLoad(previous_foam, i0 + vec2(1, 1), i, 0).rg;
    let foam = mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
    return PreviousLookup(foam, min(min(a, b), min(c, d)), max(max(a, b), max(c, d)));
}

fn drift_at(texel: vec2<i32>, slice: u32) -> vec2<f32> {
    let hi = vec2<i32>(i32(lod.resolution) - 1);
    return textureLoad(lod_drift, clamp(texel, vec2<i32>(0), hi), slice, 0).xy;
}

@compute @workgroup_size(8, 8, 1)
fn foam_forward(@builtin(global_invocation_id) gid: vec3<u32>) {
    let slice = gid.z;
    if gid.x >= lod.resolution || gid.y >= lod.resolution || slice >= lod.count {
        return;
    }
    let xz = lod_texel_world(lod, gid.xy, slice);
    let u = drift_at(vec2<i32>(gid.xy), slice);
    let foam = previous_at(xz - u * params.dt, slice).foam;
    textureStore(foam_out, gid.xy, slice, vec4<f32>(foam, 0.0, 0.0));
}

@compute @workgroup_size(8, 8, 1)
fn foam_correct(@builtin(global_invocation_id) gid: vec3<u32>) {
    let slice = gid.z;
    if gid.x >= lod.resolution || gid.y >= lod.resolution || slice >= lod.count {
        return;
    }
    let texel = vec2<i32>(gid.xy);
    let xz = lod_texel_world(lod, gid.xy, slice);
    let u = drift_at(texel, slice);
    let step = u * params.dt;

    // MacCormack: carry the forward result back to the start of the step and correct by half the
    // round-trip error. A back lookup that leaves this slice keeps the forward result.
    let forward = textureLoad(forward_foam, texel, slice, 0).rg;
    var foam = forward;
    let back_uv = slice_uv(lod, xz + step, slice);
    let margin = 1.0 / f32(max(lod.resolution, 2u));
    if all(back_uv > vec2(margin)) && all(back_uv < vec2(1.0 - margin)) {
        let back = bilinear_clamp_layer(forward_foam, back_uv, slice).rg;
        let range = previous_at(xz - step, slice);
        foam = clamp(forward + 0.5 * (previous_at(xz, slice).foam - back), range.lo, range.hi);
    }

    // Area change of a drifting patch, so pure transport keeps the total amount of foam.
    let texel_width = lod.slices[slice].z;
    let du = drift_at(texel + vec2(1, 0), slice) - drift_at(texel - vec2(1, 0), slice);
    let dv = drift_at(texel + vec2(0, 1), slice) - drift_at(texel - vec2(0, 1), slice);
    let divergence = (du.x + dv.y) / (2.0 * texel_width);
    foam *= exp(-params.dt * divergence);

    let breaking = saturate(textureLoad(lod_foam, texel, slice, 0).r);
    let fresh = foam.x * params.active_decay + breaking * params.source_active;
    let residual = foam.y * params.residual_decay + foam.x * params.residual_gain
        + breaking * params.source_residual;
    let result = saturate(vec2<f32>(fresh, residual));
    textureStore(foam_out, gid.xy, slice, vec4<f32>(result, 0.0, 0.0));
}

/// The three flow layers of one slice at a lookup, bilinearly blended, or nothing found.
struct FlowLookup {
    found: bool,
    short: vec4<f32>,
    long: vec4<f32>,
    deformation: vec4<f32>,
}

/// Previous flow state at world `xz`, looked up from slice `start` outward.
fn previous_flow_at(xz: vec2<f32>, start: u32) -> FlowLookup {
    var out: FlowLookup;
    let i = slice_holding(params.previous, xz, start);
    out.found = i >= 0;
    if !out.found {
        return out;
    }
    let uv = slice_uv(params.previous, xz, u32(i));
    let base = u32(i) * FOAM_FLOW_LAYERS_PER_SLICE;
    out.short = bilinear_clamp_layer(previous_flow, uv, base);
    out.long = bilinear_clamp_layer(previous_flow, uv, base + 1u);
    out.deformation = bilinear_clamp_layer(previous_flow, uv, base + 2u);
    return out;
}

/// Carries each phase's texture offset and accumulated deformation along the drift, Neyret (2003)
/// §2.1, and starts a phase over with a fresh offset and no deformation when its latency runs out.
/// A pattern read at `xz + offset` then moves with the drift. The deformation grows by the norm of
/// the strain rate, the symmetric part of the drift gradient, times the step.
@compute @workgroup_size(8, 8, 1)
fn foam_flow(@builtin(global_invocation_id) gid: vec3<u32>) {
    let slice = gid.z;
    if gid.x >= lod.resolution || gid.y >= lod.resolution || slice >= lod.count {
        return;
    }
    let texel = vec2<i32>(gid.xy);
    let xz = lod_texel_world(lod, gid.xy, slice);
    let u = drift_at(texel, slice);
    let step = u * params.dt;

    let texel_width = lod.slices[slice].z;
    let du = (drift_at(texel + vec2(1, 0), slice) - drift_at(texel - vec2(1, 0), slice))
        / (2.0 * texel_width);
    let dv = (drift_at(texel + vec2(0, 1), slice) - drift_at(texel - vec2(0, 1), slice))
        / (2.0 * texel_width);
    let shear = 0.5 * (du.y + dv.x);
    let strain = sqrt(du.x * du.x + dv.y * dv.y + 2.0 * shear * shear);

    let previous = previous_flow_at(xz - step, slice);
    var offsets = array<vec4<f32>, 2>(previous.short - step.xyxy, previous.long - step.xyxy);
    var deformation = previous.deformation + strain * params.dt;
    let before = params.time - params.dt;
    for (var layer = 0u; layer < 2u; layer++) {
        for (var phase = 0u; phase < 2u; phase++) {
            let restart = !previous.found || params.restart > 0.5
                || params.dt >= FOAM_FLOW_LATENCY[layer]
                || foam_flow_age(params.time, layer, phase) < foam_flow_age(before, layer, phase);
            if restart {
                let seed = foam_flow_seed(params.time, layer, phase);
                if phase == 0u {
                    offsets[layer] = vec4<f32>(seed, offsets[layer].zw);
                } else {
                    offsets[layer] = vec4<f32>(offsets[layer].xy, seed);
                }
                deformation[2u * layer + phase] = 0.0;
            }
        }
    }
    let base = slice * FOAM_FLOW_LAYERS_PER_SLICE;
    textureStore(flow_out, gid.xy, base, offsets[0]);
    textureStore(flow_out, gid.xy, base + 1u, offsets[1]);
    textureStore(flow_out, gid.xy, base + 2u, deformation);
}
