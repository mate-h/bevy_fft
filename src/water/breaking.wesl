// Breaking tests shared by the solver LOD writers. Each returns a breaking intensity from 0 to 1,
// which the writers store in the LOD foam lane for the foam sim.
//
// `breaking_front` follows the three conditions of Chentanez and Müller 2010 (CMF10 §2.4.2): the
// surface is steep, it rises fast, and it sits near a crest. CMF10 states them in grid units with
// α = 0.45, v = 4, and l = -4 at their time step and cell size. These are the same tests in meters
// and seconds, with soft edges so the foam does not flicker as a front crosses a texel.

/// Surface slope across which a front counts as steep. CMF10's α g Δt / Δx comes to about 0.07
/// at 60 steps per second and 1 m cells.
const STEEP_START: f32 = 0.12;
const STEEP_END: f32 = 0.3;
/// Rate the surface rises at, in meters per second, across which a front counts as rising fast.
/// A half-meter bore over a meter of water climbs at roughly 1 m/s across a one-meter face.
const RISE_START: f32 = 0.1;
const RISE_END: f32 = 0.5;
/// Surface curvature in 1/m below which a texel counts as near a crest.
const CREST_CURVATURE: f32 = -0.1;
/// Water depth in meters below which a rising surface counts as swash on the beach.
const SWASH_DEPTH_START: f32 = 0.05;
const SWASH_DEPTH_END: f32 = 0.3;
/// Strain rate of the current in 1/s across which shear behind rocks and hulls whitens the water.
const STRAIN_START: f32 = 1.0;
const STRAIN_END: f32 = 3.0;

/// Breaking at a front. `slope` is the steepest one-sided surface slope, `rise` the rate the
/// surface climbs in m/s, and `curvature` the Laplacian of the surface height in 1/m.
fn breaking_front(slope: f32, rise: f32, curvature: f32) -> f32 {
    let steep = smoothstep(STEEP_START, STEEP_END, slope);
    let rising = smoothstep(RISE_START, RISE_END, rise);
    let crest = 1.0 - smoothstep(CREST_CURVATURE, 0.0, curvature);
    return steep * rising * crest;
}

/// Foam where a bore runs into thin water on the beach. `depth` is the water column in meters.
fn breaking_swash(rise: f32, depth: f32) -> f32 {
    return smoothstep(RISE_START, RISE_END, rise) * (1.0 - smoothstep(SWASH_DEPTH_START, SWASH_DEPTH_END, depth));
}

/// Foam from shear in the current. `strain` is the strain-rate magnitude sqrt(2 S:S) in 1/s.
fn breaking_turbulence(strain: f32) -> f32 {
    return smoothstep(STRAIN_START, STRAIN_END, strain);
}

/// Larger of the two one-sided differences, CMF10 eq. 19, so a sharp step is not averaged away.
fn max_abs(a: f32, b: f32) -> f32 {
    return select(b, a, abs(a) > abs(b));
}
