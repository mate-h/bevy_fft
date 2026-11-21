// Writes the dispersive patch into every water LOD slice that overlaps it.
//
// In add mode the patch's deviation from rest goes on top of the data below, so ocean waves run
// through it and a wake rides on them. In replace mode the patch takes over the height and the
// water depth lane. Flow follows the same mode, with velocity from the solver's flux and the
// vertical lane from mass conservation. Drift takes the solver's velocity in replace mode. In add
// mode the patch carries only waves on top of the water below, so drift stays as it was. The
// foam lane gets the breaking intensity of the patch's crests, by the CMF10 test with the slope
// band of a breaking deep-water wave, and of the water a moving hull pushes aside. With the bore
// band of the shallow solvers, the diverging waves of a boat's wake broke along their whole
// length and covered the Kelvin wedge in foam. Both modes fade out along the patch edge over `blend_cells`. The data below comes
// from a scratch copy made just before this pass.
//
// In add mode the patch also modulates the short waves of the data below, as a long wave strains
// the wind ripples riding on it: their spectral level grows over its crests and falls in its
// troughs, by `exp(m |k| η)` (see `short_wave_modulation`). The unresolved slope variance in the
// foam lane's alpha takes that factor, and the resolved slopes shorter than the patch's waves take
// its square root.

import bevy_fft::water::breaking::{breaking_crest, max_abs};
import bevy_fft::water::lod::{WaterLodSlices, lod_texel_world};

const BLEND_REPLACE: u32 = 1u;

struct DispersiveLodParams {
    /// World X and Z of the patch centre in meters.
    center: vec2<f32>,
    cell_size: f32,
    /// Height above the surface mesh plane of the solver's rest surface, in meters.
    datum: f32,
    cells: u32,
    blend_cells: f32,
    blend: u32,
    /// 1 when `pressure` holds the head bodies press the patch down by.
    has_pressure: u32,
    /// Hull position this frame and last frame, in cells.
    hull: vec2<f32>,
    hull_prev: vec2<f32>,
    /// Hull radius in cells.
    hull_radius: f32,
    /// Hull speed in meters per second, zero while no hull is in the water.
    hull_speed: f32,
    /// Hydrodynamic modulation transfer factor of the short waves on the patch, zero for none.
    short_wave_modulation: f32,
    _pad2: f32,
}

@group(0) @binding(0) var<uniform> params: DispersiveLodParams;
@group(0) @binding(1) var<uniform> lod: WaterLodSlices;
@group(0) @binding(2) var displacement_out: texture_storage_2d_array<rgba16float, write>;
@group(0) @binding(3) var foam_out: texture_storage_2d_array<rgba16float, write>;
@group(0) @binding(4) var flow_out: texture_storage_2d_array<rgba32float, write>;
@group(0) @binding(5) var base_displacement: texture_2d_array<f32>;
@group(0) @binding(6) var base_foam: texture_2d_array<f32>;
@group(0) @binding(7) var base_flow: texture_2d_array<f32>;
/// Water depth in `.r` in meters, and volume flux per unit width along X and Z in `.gb` in m²/s.
@group(0) @binding(8) var state: texture_2d<f32>;
/// Bed height in `.r`, in meters relative to the rest surface.
@group(0) @binding(9) var bed: texture_2d<f32>;
@group(0) @binding(10) var drift_out: texture_storage_2d_array<rg32float, write>;
@group(0) @binding(11) var base_drift: texture_2d_array<f32>;
/// Head in meters that floating bodies press the surface down by, when `has_pressure` is 1.
@group(0) @binding(12) var pressure: texture_2d<f32>;

/// Bilinear pressure head at grid coordinate `g`.
fn pressure_at(g: vec2<f32>) -> f32 {
    if params.has_pressure == 0u {
        return 0.0;
    }
    let hi = vec2<i32>(i32(params.cells) - 1);
    let x = g - 0.5;
    let i0 = vec2<i32>(floor(x));
    let f = x - floor(x);
    let a = textureLoad(pressure, clamp(i0, vec2(0), hi), 0).r;
    let b = textureLoad(pressure, clamp(i0 + vec2(1, 0), vec2(0), hi), 0).r;
    let c = textureLoad(pressure, clamp(i0 + vec2(0, 1), vec2(0), hi), 0).r;
    let d = textureLoad(pressure, clamp(i0 + vec2(1, 1), vec2(0), hi), 0).r;
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

/// Depth below which velocity is taken from a floor depth, so flux over nearly dry cells does not
/// turn into runaway speeds.
const MIN_FLOW_DEPTH: f32 = 0.05;
/// Hull speed in m/s across which the water it pushes aside turns white.
const HULL_FOAM_SPEED_START: f32 = 0.5;
const HULL_FOAM_SPEED_END: f32 = 3.0;

/// Breaking intensity in the ring of water around the hull's outline, swept from last frame's
/// position to this one's.
fn hull_foam(g: vec2<f32>) -> f32 {
    if params.hull_speed <= 0.0 {
        return 0.0;
    }
    let seg = params.hull - params.hull_prev;
    let t = saturate(dot(g - params.hull_prev, seg) / max(dot(seg, seg), 1e-6));
    let d = length(g - (params.hull_prev + seg * t));
    let r = max(params.hull_radius, 1.0);
    let ring = smoothstep(0.3 * r, 0.8 * r, d) * (1.0 - smoothstep(0.8 * r, 1.4 * r, d));
    return ring * smoothstep(HULL_FOAM_SPEED_START, HULL_FOAM_SPEED_END, params.hull_speed);
}

/// Bilinear bed, depth and flux at grid coordinate `g`, sharing one footprint: `.x` bed, `.y`
/// depth, `.zw` flux along X and Z.
fn cell_at(g: vec2<f32>) -> vec4<f32> {
    let hi = vec2<i32>(i32(params.cells) - 1);
    let x = g - 0.5;
    let i0 = vec2<i32>(floor(x));
    let f = x - floor(x);
    let c00 = clamp(i0, vec2(0), hi);
    let c10 = clamp(i0 + vec2(1, 0), vec2(0), hi);
    let c01 = clamp(i0 + vec2(0, 1), vec2(0), hi);
    let c11 = clamp(i0 + vec2(1, 1), vec2(0), hi);
    let a = vec4<f32>(textureLoad(bed, c00, 0).r, textureLoad(state, c00, 0).rgb);
    let b = vec4<f32>(textureLoad(bed, c10, 0).r, textureLoad(state, c10, 0).rgb);
    let c = vec4<f32>(textureLoad(bed, c01, 0).r, textureLoad(state, c01, 0).rgb);
    let d = vec4<f32>(textureLoad(bed, c11, 0).r, textureLoad(state, c11, 0).rgb);
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

/// Bilinear free-surface elevation relative to rest at grid coordinate `g`.
fn eta_at(g: vec2<f32>) -> f32 {
    let c = cell_at(g);
    return c.x + c.y;
}

/// Exponent the short-wave modulation stays within, so steep patch features such as the hull's
/// own trough neither blank the ripples out nor blow them up: a factor between 0.22 and 4.5.
const MODULATION_LIMIT: f32 = 1.5;

/// `|k| η` at grid coordinate `g`: the patch's elevation weighted by its local wavenumber, which
/// for one long wave of amplitude a is `k a cos θ`, its slope amplitude times the phase, largest at
/// crests and most negative in troughs. For deep-water waves it is also the time integral of the
/// convergence of the surface velocity, `-∇·u = |k| ∂η/∂t`, the strain that compresses the short
/// waves. The local wavenumber comes from the elevation and its first three derivatives: for a
/// plane wave `(∇²η)² + |∇η||∇∇²η| = a²k⁴` and `|∇η|² + |η||∇²η| = a²k²` at every phase, so
/// `|k| η = -∇²η / k` needs no wavelength given. `c`, `e`, `w`, `n` and `s` are the elevation at
/// the centre and one step `h` meters east, west, north and south, `slope` its gradient. Returns
/// `|k| η` in `.x` and the local wavenumber in rad/m in `.y`, zero where the patch is flat.
fn wavenumber_elevation(
    g: vec2<f32>,
    span: f32,
    h: f32,
    c: f32,
    e: f32,
    w: f32,
    n: f32,
    s: f32,
    slope: vec2<f32>,
) -> vec2<f32> {
    let dx = vec2<f32>(span, 0.0);
    let dz = vec2<f32>(0.0, span);
    let ne = eta_at(g + dx + dz);
    let nw = eta_at(g - dx + dz);
    let se = eta_at(g + dx - dz);
    let sw = eta_at(g - dx - dz);
    let ih2 = 1.0 / (h * h);
    let lap = (e + w + n + s - 4.0 * c) * ih2;
    let lap_e = (eta_at(g + 2.0 * dx) + c + ne + se - 4.0 * e) * ih2;
    let lap_w = (eta_at(g - 2.0 * dx) + c + nw + sw - 4.0 * w) * ih2;
    let lap_n = (eta_at(g + 2.0 * dz) + c + ne + nw - 4.0 * n) * ih2;
    let lap_s = (eta_at(g - 2.0 * dz) + c + se + sw - 4.0 * s) * ih2;
    let grad_lap = vec2<f32>(lap_e - lap_w, lap_n - lap_s) / (2.0 * h);
    let k4 = lap * lap + length(slope) * length(grad_lap);
    let k2 = dot(slope, slope) + abs(c * lap);
    if k4 <= 1e-20 || k2 <= 1e-20 {
        return vec2<f32>(0.0);
    }
    // No shorter than four steps of the stencil, which can't resolve shorter waves, and no longer
    // than the patch.
    let k_max = 0.5 * 3.14159265 / h;
    let k_min = 6.2831853 / (f32(params.cells) * params.cell_size);
    let k = clamp(sqrt(sqrt(k4 / k2)), k_min, k_max);
    return vec2<f32>(-lap / k, k);
}

/// Slopes of `base` in `.rg` at a texel, with the part shorter than about `radius` texels scaled
/// by `gain`. The 4-tap mean around the texel keeps what is longer, so swell and the patch's
/// own scale pass unchanged and the ripples on them take the gain.
fn modulate_short_slopes(texel: vec2<u32>, slice: u32, radius: f32, gain: f32, base: vec4<f32>) -> vec4<f32> {
    if abs(gain - 1.0) < 1e-4 {
        return base;
    }
    let r = i32(clamp(round(radius), 1.0, 64.0));
    let hi = vec2<i32>(i32(lod.resolution) - 1);
    let t = vec2<i32>(texel);
    let a = textureLoad(base_displacement, clamp(t + vec2(r, 0), vec2(0), hi), slice, 0).rg;
    let b = textureLoad(base_displacement, clamp(t - vec2(r, 0), vec2(0), hi), slice, 0).rg;
    let c = textureLoad(base_displacement, clamp(t + vec2(0, r), vec2(0), hi), slice, 0).rg;
    let d = textureLoad(base_displacement, clamp(t - vec2(0, r), vec2(0), hi), slice, 0).rg;
    let long = 0.25 * (a + b + c + d);
    return vec4<f32>(long + (base.rg - long) * gain, base.ba);
}

@compute @workgroup_size(8, 8, 1)
fn dispersive_lod_write(@builtin(global_invocation_id) gid: vec3<u32>) {
    let res = lod.resolution;
    let slice = gid.z;
    if gid.x >= res || gid.y >= res || slice >= lod.count {
        return;
    }
    let xz = lod_texel_world(lod, gid.xy, slice);
    let n = f32(params.cells);
    let g = (xz - params.center) / params.cell_size + 0.5 * n;
    if any(g < vec2(0.0)) || any(g > vec2(n)) {
        return;
    }

    let edge = min(min(g.x, n - g.x), min(g.y, n - g.y));
    let w = smoothstep(0.0, max(params.blend_cells, 1e-3), edge);

    // Difference over at least one texel so coarse slices see the slope they can resolve.
    let span = max(lod.slices[slice].z / params.cell_size, 1.0);
    let dx = vec2<f32>(span, 0.0);
    let dz = vec2<f32>(0.0, span);
    let inv = 1.0 / (2.0 * span * params.cell_size);
    // The five points of the stencil, each read once for its bed, depth and flux.
    let c = cell_at(g);
    let east = cell_at(g + dx);
    let west = cell_at(g - dx);
    let north = cell_at(g + dz);
    let south = cell_at(g - dz);
    let ex = vec2<f32>(east.x + east.y, west.x + west.y);
    let ez = vec2<f32>(north.x + north.y, south.x + south.y);
    let slope = vec2<f32>(ex.x - ex.y, ez.x - ez.y) * inv;
    let bd = c.xy;
    let eta = bd.x + bd.y;
    let st = c.yzw;
    let rise = -((east.z - west.z) + (north.w - south.w)) * inv;
    // The flow lane `.w` carries the pressure head, so body forces read the surface the water
    // would have without the bodies. Under a hull it carries the whole deviation instead, and the
    // patch adds no flow there, so the body reads the water below the patch and not the patch's
    // response to its own pressure. A hull at rest holds the patch at its open level anyway, by
    // the obstruction, but a moving hull lets the water under it go free so that its pressure
    // makes its wake. The voxel forces, which arrive frames late and have no planing lift, can't
    // ride that wave field: the boat example sat in its own trough and porpoised. Its drag is
    // calibrated without it.
    let head = pressure_at(g);
    var flow = vec4<f32>(st.yz / max(st.x, MIN_FLOW_DEPTH), rise, head);
    if head > 0.0 {
        flow = vec4<f32>(0.0, 0.0, 0.0, -eta);
    }
    let base_v = textureLoad(base_flow, gid.xy, slice, 0);

    let base = textureLoad(base_displacement, gid.xy, slice, 0);
    let base_f = textureLoad(base_foam, gid.xy, slice, 0);
    let h = span * params.cell_size;
    let steepest = length(vec2<f32>(max_abs(ex.x - eta, eta - ex.y), max_abs(ez.x - eta, eta - ez.y))) / h;
    let curvature = (ex.x + ex.y + ez.x + ez.y - 4.0 * eta) / (h * h);
    let breaking = max(breaking_crest(steepest, rise, curvature), hull_foam(g)) * w;
    if params.blend == BLEND_REPLACE {
        let surface = vec4<f32>(slope, params.datum + eta, 0.0);
        textureStore(displacement_out, gid.xy, slice, mix(base, surface, w));
        let foam = vec4<f32>(
            base_f.r * (1.0 - w) + breaking,
            mix(base_f.g, bd.y, w),
            base_f.b * (1.0 - w),
            base_f.a,
        );
        textureStore(foam_out, gid.xy, slice, foam);
        textureStore(flow_out, gid.xy, slice, mix(base_v, flow, w));
        let base_d = textureLoad(base_drift, gid.xy, slice, 0).xy;
        textureStore(drift_out, gid.xy, slice, vec4<f32>(mix(base_d, flow.xy, w), 0.0, 0.0));
    } else {
        // Short-wave modulation by the patch's waves. Under a hull the patch is the hull's own
        // trough, covered in foam, and drives none.
        var exponent = 0.0;
        var radius = 1.0;
        if params.short_wave_modulation != 0.0 && head <= 0.0 {
            let ke = wavenumber_elevation(g, span, h, eta, ex.x, ex.y, ez.x, ez.y, slope);
            exponent = clamp(params.short_wave_modulation * ke.x, -MODULATION_LIMIT, MODULATION_LIMIT) * w;
            // The resolved slopes shorter than a quarter of the patch's local wavelength count as
            // the short waves, in this slice's texels.
            radius = 0.25 * 6.2831853 / max(ke.y, 1e-3) / lod.slices[slice].z;
        }
        let level = exp(exponent);
        let modulated = modulate_short_slopes(gid.xy, slice, radius, sqrt(level), base);
        let deviation = vec4<f32>(slope, params.datum + eta, 0.0) * w;
        textureStore(displacement_out, gid.xy, slice, modulated + deviation);
        let variance = base_f.a * level;
        textureStore(foam_out, gid.xy, slice, vec4<f32>(max(base_f.r, breaking), base_f.gb, variance));
        textureStore(flow_out, gid.xy, slice, base_v + flow * w);
    }
}
