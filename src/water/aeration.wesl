// Air in the water column of white water, for the solver LOD writers and the water surface.
//
// The writers estimate the air fraction C, the volume of air over the volume of air and water,
// that the flow would hold where it loses energy fast enough to entrain air: at hydraulic jumps
// and bores, in steep fast flow, and at waterfall lips and landings. They store it in the LOD
// aeration lane. The foam sim carries the air along the drift and relaxes it toward that value
// while bubbles leave the column. The surface turns the air fraction into a bright, scattering
// layer that hides more of the bed the more air it holds.

/// Gravity in m/s².
const AIR_GRAVITY: f32 = 9.81;
/// Air fraction over the roller of a hydraulic jump with inflow Froude number 2. Vallé and
/// Pasternack (2006, §3.2 and Table 1) measured a median of 33 percent over the roller of an
/// unsubmerged jump in a bedrock step-pool channel with F1 = 2.
const JUMP_ROLLER_AIR: f32 = 0.33;
/// Inflow Froude number below which a jump has no roller and entrains no air. Chanson (2009,
/// §2.3) puts the end of the undular surge, which has no roller, at a Froude number of 1.5 to 1.8.
const JUMP_ONSET_FROUDE: f32 = 1.5;
/// Most air the column holds. Vallé and Pasternack (2006, §3.3) found peaks of 40 to 50 percent
/// just past the toe of the unsubmerged jump.
const AIR_MAX: f32 = 0.5;
/// Velocity drop across a front, in m/s, over which it starts to entrain air. Below it the drop is
/// solver noise in thin water rather than a roller. An appearance choice.
const JUMP_DROP_START: f32 = 0.3;
const JUMP_DROP_END: f32 = 0.8;
/// Speed in m/s over which steep fast flow starts to entrain air at its surface. An appearance
/// choice, near the 1 m/s at which a plunging jet starts to entrain air.
const CHUTE_SPEED_START: f32 = 1.0;
const CHUTE_SPEED_END: f32 = 2.0;
/// Air fraction where a waterfall lands. Vallé and Pasternack (2006, §3.3) measured 15 to 25
/// percent at the toe of the jump below a ballistic jet that plunged into its pool.
const FALL_AIR: f32 = 0.2;
/// Impact speed in m/s, `sqrt(2 g drop)`, over which a fall starts to entrain air. Half a meter of
/// drop gives about 3 m/s. An appearance choice.
const FALL_SPEED_START: f32 = 1.0;
const FALL_SPEED_END: f32 = 3.0;

/// Air fraction of a front across which the flow slows by `drop` m/s, where the shallower side is
/// `inflow_depth` m deep. In the front's own frame a jump slows its inflow by
/// Δu / sqrt(g d1) = Fr1 (1 - d1 / d2), with d2 / d1 from Bélanger's equation (Chanson 2009,
/// eq. 1), and `1 + 0.9 Δu / sqrt(g d1)` inverts that to within 4 percent for Fr1 from 1.5 to 5.
/// The velocity drop is the same in every frame, so the test also holds for a bore that moves.
/// The air fraction grows linearly with Fr1 from the onset of the roller to `JUMP_ROLLER_AIR` at
/// Fr1 = 2, as Rajaratnam (1967) found peak air growing with Fr1, up to `AIR_MAX`.
fn aeration_jump(drop: f32, inflow_depth: f32) -> f32 {
    let froude = 1.0 + 0.9 * drop / sqrt(AIR_GRAVITY * max(inflow_depth, 1e-3));
    let air = JUMP_ROLLER_AIR * (froude - JUMP_ONSET_FROUDE) / (2.0 - JUMP_ONSET_FROUDE);
    return clamp(air, 0.0, AIR_MAX) * smoothstep(JUMP_DROP_START, JUMP_DROP_END, drop);
}

/// Air fraction of fast flow down a steep surface of slope `slope` at Froude number `froude` and
/// speed `speed` m/s. Self-aerated flow down a chute holds C = 0.9 sin θ in uniform equilibrium,
/// Chanson's (1993) fit to Straub and Anderson (1958). It fades in from Froude number 1 to 2,
/// since subcritical flow down a slope is not fast enough to tear its surface.
fn aeration_chute(slope: f32, froude: f32, speed: f32) -> f32 {
    let sin_theta = slope / sqrt(1.0 + slope * slope);
    return min(0.9 * sin_theta, AIR_MAX) * smoothstep(1.0, 2.0, froude)
        * smoothstep(CHUTE_SPEED_START, CHUTE_SPEED_END, speed);
}

/// Air fraction at the lip and the foot of a waterfall whose bed drops `drop` m.
fn aeration_fall(drop: f32) -> f32 {
    let impact = sqrt(2.0 * AIR_GRAVITY * max(drop, 0.0));
    return FALL_AIR * smoothstep(FALL_SPEED_START, FALL_SPEED_END, impact);
}

/// Radius in meters that sets how much light the air of white water scatters per unit of air: the
/// Sauter mean radius, three times the air's volume over its surface. A jump roller holds bubbles
/// from a fraction of a millimetre to centimetres and larger air packets that it traps and sheds.
/// The many small bubbles dominate the count, but the few large ones hold most of the air, so the
/// Sauter radius sits well above the most common bubble. The same large packets set the
/// detrainment speed of about 1 m/s that `WaterAerationSettings` takes from Vallé and Pasternack
/// (2006), several times what a millimetre bubble rises at. An appearance choice in that range: at
/// 4 mm, 5 percent of air over 0.2 m of water hides 43 percent of the bed and 30 percent hides 97
/// percent. With 2 mm bubbles 5 percent already hid 68 percent, and every patch looked opaque.
const AIR_BUBBLE_RADIUS: f32 = 0.004;
/// Mean cosine of the light a bubble scatters. Geometric optics gives about 0.85 for air bubbles in
/// water much larger than the wavelength, so most light carries on nearly straight.
const AIR_BUBBLE_ASYMMETRY: f32 = 0.85;
/// Share of diffuse light the water surface sends back down: total internal reflection past
/// 48.6 degrees keeps all but 1 / n² of it, and Fresnel reflection takes 6.6 percent of the rest,
/// 1 - (1 - 0.066) / 1.333².
const AIR_SURFACE_INTERNAL_REFLECTANCE: f32 = 0.47;
/// Share of diffuse light the water surface reflects from above.
const AIR_SURFACE_EXTERNAL_REFLECTANCE: f32 = 0.066;
/// Absorption of pure water in 1/m at 650, 550 and 450 nm (Pope and Fry 1997), for surfaces drawn
/// without the water optics.
const AIR_PURE_WATER_ABSORPTION: vec3<f32> = vec3<f32>(0.34, 0.0565, 0.0092);

/// Look of an aerated water column, seen from above.
struct AeratedLayer {
    /// Diffuse reflectance of the bubbles, after the light crosses the surface both ways.
    reflectance: vec3<f32>,
    /// Share of light the bubbles stop on the way through the column, which hides the bed.
    opacity: f32,
}

/// Transport scattering coefficient in 1/m of water holding air fraction `air`. Bubbles of radius
/// r scatter with an efficiency of 2, so N = C / (4/3 π r³) of them per cubic meter scatter
/// 3 C / (2 r) per meter, and (1 - g) of that turns the light around.
fn aerated_scattering(air: f32) -> f32 {
    return 1.5 * max(air, 0.0) / AIR_BUBBLE_RADIUS * (1.0 - AIR_BUBBLE_ASYMMETRY);
}

/// Look of a column `depth` m deep holding air fraction `air` in water of absorption `absorption`
/// per meter, with the scattering of `aerated_scattering`. The reflectance of a deep layer is
/// Kubelka and Munk's, with K = 2 a and S the transport scattering, scaled by how much of the light
/// the layer stops, and the surface returns part of what leaves it back down into the bubbles.
fn aerated_layer(air: f32, depth: f32, absorption: vec3<f32>) -> AeratedLayer {
    var out: AeratedLayer;
    let scattering = aerated_scattering(air);
    out.opacity = 1.0 - exp(-scattering * max(depth, 0.0));
    if out.opacity <= 0.0 {
        return out;
    }
    let x = 2.0 * max(absorption, vec3<f32>(0.0)) / max(scattering, 1e-6);
    let deep = 1.0 + x - sqrt(x * x + 2.0 * x);
    let r = deep * out.opacity;
    let ri = AIR_SURFACE_INTERNAL_REFLECTANCE;
    out.reflectance = (1.0 - AIR_SURFACE_EXTERNAL_REFLECTANCE) * (1.0 - ri) * r / (1.0 - ri * r);
    return out;
}
