// Advected foam texture coordinates after Neyret (2003), shared by the foam sim and the surface.
//
// The foam pattern is read at `xz + offset`, where each offset is carried along the drift on the
// LOD slices, so the pattern moves with the foam. An offset stretches the pattern as the drift
// shears it, so each one starts over at a fresh random value once per latency, while its weight is
// zero. Neyret §1.3 blends phase-shifted parameterizations so the weights always sum to one. Here
// each layer has two phases half a latency apart.
//
// One latency cannot suit calm and strained flow at once (Neyret §1.3). So there are two layers
// with different latencies, and each texel accumulates the deformation of every phase since it
// started over (Neyret §2.1). Pixels blend the two layers by how far their deformation sits from a
// target, as Neyret §2 does between the two layers that bracket it.
//
// Each LOD slice `s` owns three layers of the flow array. Layer `3 s` holds the offsets of the
// short layer, phase A in `.xy` and phase B in `.zw`. Layer `3 s + 1` holds those of the long
// layer. Layer `3 s + 2` holds the accumulated deformation of short A, short B, long A and long B.

/// Flow array layers per LOD slice.
const FOAM_FLOW_LAYERS_PER_SLICE: u32 = 3u;

/// Seconds between restarts of each phase of the short and the long layer. Both divide the hour
/// after which the time wraps, so the wrap lines up with a restart. The long latency carries calm
/// drift, such as the Stokes drift of the open ocean, far enough to hide the restart. The short one
/// keeps sheared foam from stretching. Appearance choices, since Neyret leaves latencies to the user.
const FOAM_FLOW_LATENCY: vec2<f32> = vec2<f32>(1.0, 4.0);

/// Deformation at which the pattern switches from the long to the short layer, Neyret's d*. The
/// deformation is the time integral of the strain-rate norm, summed over both phases of a layer, so
/// a steady strain rate e gives a layer about e times its latency. An appearance choice.
const FOAM_FLOW_TARGET_DEFORMATION: f32 = 0.5;

/// Meters of water covered by one repeat of the foam texture. Fresh offsets spread over one repeat.
const FOAM_FLOW_TILE: f32 = 6.0;

/// Fraction of its latency that phase `phase` (0 or 1) of `layer` (0 short, 1 long) has run at
/// `time`. Phase 1 runs half a latency ahead of phase 0.
fn foam_flow_age(time: f32, layer: u32, phase: u32) -> f32 {
    let latency = FOAM_FLOW_LATENCY[layer];
    return fract(time / latency + 0.5 * f32(phase));
}

/// Weight of phase 0 of `layer` at `time`. It is zero when that phase starts over and one when
/// phase 1 does, and phase 1 takes the rest.
fn foam_flow_weight(time: f32, layer: u32) -> f32 {
    return 1.0 - abs(2.0 * foam_flow_age(time, layer, 0u) - 1.0);
}

/// Fresh offset for phase `phase` of `layer` when it starts over at `time`. Every texel draws the
/// same value, so the pattern starts over in one piece.
fn foam_flow_seed(time: f32, layer: u32, phase: u32) -> vec2<f32> {
    let latency = FOAM_FLOW_LATENCY[layer];
    let cycle = floor(time / latency + 0.5 * f32(phase));
    let n = 2.0 * cycle + f32(phase) + 977.0 * f32(layer);
    let seed = vec2<f32>(fract(sin(n * 12.9898) * 43758.547), fract(sin(n * 78.233) * 43758.547));
    return seed * FOAM_FLOW_TILE;
}

/// Weight of the long layer for accumulated deformations `d` (short A, short B, long A, long B).
/// It is one while the long layer stays under the target and falls to zero as the short layer
/// reaches it, Neyret's back-interpolation of the ideal layer number.
fn foam_flow_long_weight(d: vec4<f32>) -> f32 {
    let short = d.x + d.y;
    let long = d.z + d.w;
    if long <= FOAM_FLOW_TARGET_DEFORMATION {
        return 1.0;
    }
    return saturate((FOAM_FLOW_TARGET_DEFORMATION - short) / max(long - short, 1e-6));
}
