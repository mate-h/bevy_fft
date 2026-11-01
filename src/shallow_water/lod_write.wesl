// Blends the shallow water surface into every water LOD slice that overlaps the grid.
//
// Inside the grid the solver replaces the long waves. Along the grid edge that replacement fades
// into the data below it over `blend_cells`, so the ocean shows through without a seam. Cascades
// the cells cannot resolve are added back from the FFT, faded out in the last half meter of
// depth, and stay at full strength through the edge blend. Dry cells contribute the still-water
// level so the mesh does not follow the bed up onto land. Flow takes the solver's velocity the
// same way, with the vertical lane from mass conservation. The foam lane gets the breaking
// intensity of CMF10 fronts between wet columns, of bores running up the beach, and of shear in the
// current. The base data comes from a scratch copy made just before this pass.

import bevy_fft::water::breaking::{breaking_front, breaking_swash, breaking_turbulence, max_abs};
import bevy_fft::water::lod::{WaterLodSlices, lod_texel_world};

struct ShallowWaterLodParams {
    /// World X and Z of the grid centre in meters.
    center: vec2<f32>,
    cell_size: f32,
    /// Height above the surface mesh plane of the solver's zero level, in meters.
    datum: f32,
    cells: vec2<u32>,
    blend_cells: f32,
    /// Still water level in solver height units. Dry cells render at `datum + fill_level`.
    fill_level: f32,
    /// World tile edge in meters for cascades 0 to 3.
    cascade_tile_sizes: vec4<f32>,
    amplitude: f32,
    choppiness: f32,
    cascade_count: u32,
    _pad: u32,
}

/// Depth at or below this, in meters, is treated as dry for the mesh height.
const DRY_DEPTH: f32 = 1e-4;
/// Unresolved FFT detail fades in across this depth range, in meters.
const DETAIL_FADE_START: f32 = 0.05;
const DETAIL_FADE_END: f32 = 0.5;
/// Water this deep on both sides of the front counts as breaking. Shallower columns are the
/// wet and dry edge, where dry cells render at the still-water level and the slope is not a wave.
const BREAK_MIN_DEPTH: f32 = 0.03;
/// The grid resolves depth-limited breaking only. Deep-water whitecaps come from the ocean, so
/// fronts fade out of foam across this depth range, in meters.
const BREAK_FADE_START: f32 = 1.5;
const BREAK_FADE_END: f32 = 3.0;

@group(0) @binding(0) var<uniform> params: ShallowWaterLodParams;
@group(0) @binding(1) var<uniform> lod: WaterLodSlices;
@group(0) @binding(2) var displacement_out: texture_storage_2d_array<rgba16float, write>;
@group(0) @binding(3) var foam_out: texture_storage_2d_array<rgba16float, write>;
@group(0) @binding(4) var flow_out: texture_storage_2d_array<rgba32float, write>;
@group(0) @binding(5) var base_displacement: texture_2d_array<f32>;
@group(0) @binding(6) var base_foam: texture_2d_array<f32>;
@group(0) @binding(7) var base_flow: texture_2d_array<f32>;
/// Bed height in `.x` and water depth in `.y`, both in meters.
@group(0) @binding(8) var bed_water: texture_2d<f32>;
/// Depth-averaged velocity at cell centres in `.xy`, in meters per second.
@group(0) @binding(9) var velocity: texture_2d<f32>;
@group(0) @binding(10) var displacement_0: texture_2d<f32>;
@group(0) @binding(11) var displacement_1: texture_2d<f32>;
@group(0) @binding(12) var displacement_2: texture_2d<f32>;
@group(0) @binding(13) var displacement_3: texture_2d<f32>;
/// Ocean foam textures, whose `.g` holds horizontal displacement Z.
@group(0) @binding(14) var foam_0: texture_2d<f32>;
@group(0) @binding(15) var foam_1: texture_2d<f32>;
@group(0) @binding(16) var foam_2: texture_2d<f32>;
@group(0) @binding(17) var foam_3: texture_2d<f32>;
@group(0) @binding(18) var drift_out: texture_storage_2d_array<rg32float, write>;
@group(0) @binding(19) var base_drift: texture_2d_array<f32>;

fn wrap_texel(p: vec2<i32>, dims: vec2<i32>) -> vec2<i32> {
    return ((p % dims) + dims) % dims;
}

/// Bilinear fetch with periodic wrap. `uv` is in tiles. Matches `bilinear_wrap` in the ocean sampler.
fn bilinear_wrap(tex: texture_2d<f32>, uv: vec2<f32>) -> vec4<f32> {
    let dims = vec2<i32>(textureDimensions(tex, 0));
    let x = uv * vec2<f32>(dims) - 0.5;
    let i0 = vec2<i32>(floor(x));
    let f = x - floor(x);
    let a = textureLoad(tex, wrap_texel(i0, dims), 0);
    let b = textureLoad(tex, wrap_texel(i0 + vec2(1, 0), dims), 0);
    let c = textureLoad(tex, wrap_texel(i0 + vec2(0, 1), dims), 0);
    let d = textureLoad(tex, wrap_texel(i0 + vec2(1, 1), dims), 0);
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

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

fn displacement_z_cascade(i: u32, uv: vec2<f32>) -> f32 {
    if i == 0u {
        return bilinear_wrap(foam_0, uv).g;
    } else if i == 1u {
        return bilinear_wrap(foam_1, uv).g;
    } else if i == 2u {
        return bilinear_wrap(foam_2, uv).g;
    }
    return bilinear_wrap(foam_3, uv).g;
}

/// Share of cascade `i` a sample of width `texel` can resolve. Matches the ocean LOD writer.
fn cascade_weight(i: u32, texel: f32) -> f32 {
    let tile = params.cascade_tile_sizes[i];
    return 1.0 - smoothstep(tile / 24.0, tile / 8.0, texel);
}

/// FFT displacement the grid cells do not own, scaled the same way as the ocean LOD writer.
struct Unresolved {
    /// Slopes in `.rg`, height in `.b`, and horizontal displacement X in `.a`.
    displacement: vec4<f32>,
    /// Horizontal displacement Z, which the LOD data keeps in the foam texture.
    displacement_z: f32,
}

fn unresolved_displacement(xz: vec2<f32>, texel: f32) -> Unresolved {
    var disp = vec4<f32>(0.0);
    var dz = 0.0;
    let count = min(params.cascade_count, 4u);
    for (var i = 0u; i < count; i++) {
        let w = cascade_weight(i, texel) * (1.0 - cascade_weight(i, params.cell_size));
        if w > 0.0 {
            let uv = xz / params.cascade_tile_sizes[i];
            disp += displacement_cascade(i, uv) * w;
            dz += displacement_z_cascade(i, uv) * w;
        }
    }
    let amp = params.amplitude;
    let chop = amp * params.choppiness;
    return Unresolved(vec4<f32>(disp.rgb * amp, disp.a * chop), dz * chop);
}

/// Bilinear bed and depth at grid coordinate `g`, where cell `i` spans `i..i + 1`.
fn bed_water_at(g: vec2<f32>) -> vec2<f32> {
    let hi = vec2<i32>(params.cells) - 1;
    let x = g - 0.5;
    let i0 = vec2<i32>(floor(x));
    let f = x - floor(x);
    let a = textureLoad(bed_water, clamp(i0, vec2(0), hi), 0).xy;
    let b = textureLoad(bed_water, clamp(i0 + vec2(1, 0), vec2(0), hi), 0).xy;
    let c = textureLoad(bed_water, clamp(i0 + vec2(0, 1), vec2(0), hi), 0).xy;
    let d = textureLoad(bed_water, clamp(i0 + vec2(1, 1), vec2(0), hi), 0).xy;
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

/// Bilinear velocity at grid coordinate `g`.
fn velocity_at(g: vec2<f32>) -> vec2<f32> {
    let hi = vec2<i32>(params.cells) - 1;
    let x = g - 0.5;
    let i0 = vec2<i32>(floor(x));
    let f = x - floor(x);
    let a = textureLoad(velocity, clamp(i0, vec2(0), hi), 0).xy;
    let b = textureLoad(velocity, clamp(i0 + vec2(1, 0), vec2(0), hi), 0).xy;
    let c = textureLoad(velocity, clamp(i0 + vec2(0, 1), vec2(0), hi), 0).xy;
    let d = textureLoad(velocity, clamp(i0 + vec2(1, 1), vec2(0), hi), 0).xy;
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

/// Volume flux per unit width, depth times velocity, at grid coordinate `g`.
fn flux_at(g: vec2<f32>) -> vec2<f32> {
    return bed_water_at(g).y * velocity_at(g);
}

/// Solver free-surface height used for the mesh. Dry cells sit at the still-water level so a
/// triangle onto land does not climb the bed.
fn render_eta(bw: vec2<f32>) -> f32 {
    if bw.y <= DRY_DEPTH {
        return params.fill_level;
    }
    return bw.x + bw.y;
}

/// Free-surface height at grid coordinate `g`. Each cell is resolved before the bilinear blend.
/// Blending bed and depth first, then adding them, lifts a sample that straddles the shore onto
/// the averaged bed.
fn eta_at(g: vec2<f32>) -> f32 {
    let hi = vec2<i32>(params.cells) - 1;
    let x = g - 0.5;
    let i0 = vec2<i32>(floor(x));
    let f = x - floor(x);
    let a = render_eta(textureLoad(bed_water, clamp(i0, vec2(0), hi), 0).xy);
    let b = render_eta(textureLoad(bed_water, clamp(i0 + vec2(1, 0), vec2(0), hi), 0).xy);
    let c = render_eta(textureLoad(bed_water, clamp(i0 + vec2(0, 1), vec2(0), hi), 0).xy);
    let d = render_eta(textureLoad(bed_water, clamp(i0 + vec2(1, 1), vec2(0), hi), 0).xy);
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

@compute @workgroup_size(8, 8, 1)
fn shallow_water_lod_write(@builtin(global_invocation_id) gid: vec3<u32>) {
    let res = lod.resolution;
    let slice = gid.z;
    if gid.x >= res || gid.y >= res || slice >= lod.count {
        return;
    }
    let xz = lod_texel_world(lod, gid.xy, slice);
    let n = vec2<f32>(params.cells);
    let g = (xz - params.center) / params.cell_size + 0.5 * n;
    // This pass is write-only. A texel left untouched would keep the previous frame's value,
    // which was written for the previous slice centres, so the outer rings would trail the camera.
    if any(g < vec2(0.0)) || any(g > n) {
        textureStore(displacement_out, gid.xy, slice, textureLoad(base_displacement, gid.xy, slice, 0));
        textureStore(foam_out, gid.xy, slice, textureLoad(base_foam, gid.xy, slice, 0));
        textureStore(flow_out, gid.xy, slice, textureLoad(base_flow, gid.xy, slice, 0));
        textureStore(drift_out, gid.xy, slice, textureLoad(base_drift, gid.xy, slice, 0));
        return;
    }

    let edge = min(min(g.x, n.x - g.x), min(g.y, n.y - g.y));
    let w = smoothstep(0.0, max(params.blend_cells, 1e-3), edge);

    // Difference over at least one texel so coarse slices see the slope they can resolve.
    let span = max(lod.slices[slice].z / params.cell_size, 1.0);
    let dx = vec2<f32>(span, 0.0);
    let dz = vec2<f32>(0.0, span);
    let inv = 1.0 / (2.0 * span * params.cell_size);
    let slope = vec2<f32>(eta_at(g + dx) - eta_at(g - dx), eta_at(g + dz) - eta_at(g - dz)) * inv;
    // Mass conservation: the surface rises where flux converges.
    let rise = -((flux_at(g + dx).x - flux_at(g - dx).x) + (flux_at(g + dz).y - flux_at(g - dz).y)) * inv;
    let bw = bed_water_at(g);
    let surface = vec4<f32>(slope, params.datum + eta_at(g), 0.0);
    let fade = smoothstep(DETAIL_FADE_START, DETAIL_FADE_END, bw.y);
    let detail = unresolved_displacement(xz, lod.slices[slice].z);
    let swe = surface + detail.displacement * fade;

    let base = textureLoad(base_displacement, gid.xy, slice, 0);
    let base_f = textureLoad(base_foam, gid.xy, slice, 0);
    textureStore(displacement_out, gid.xy, slice, mix(base, swe, w));
    let banks = min(
        min(bed_water_at(g + dx).y, bed_water_at(g - dx).y),
        min(bed_water_at(g + dz).y, bed_water_at(g - dz).y),
    );
    let wet = smoothstep(BREAK_MIN_DEPTH, 2.0 * BREAK_MIN_DEPTH, banks);
    let h = span * params.cell_size;
    let e = eta_at(g);
    let ex = vec2<f32>(eta_at(g + dx), eta_at(g - dx));
    let ez = vec2<f32>(eta_at(g + dz), eta_at(g - dz));
    let steepest = length(vec2<f32>(max_abs(ex.x - e, e - ex.y), max_abs(ez.x - e, e - ez.y))) / h;
    let curvature = (ex.x + ex.y + ez.x + ez.y - 4.0 * e) / (h * h);
    let front = breaking_front(steepest, rise, curvature) * wet
        * (1.0 - smoothstep(BREAK_FADE_START, BREAK_FADE_END, bw.y));
    let swash = breaking_swash(rise, bw.y) * smoothstep(DRY_DEPTH, BREAK_MIN_DEPTH, bw.y);
    let ux = velocity_at(g + dx) - velocity_at(g - dx);
    let uz = velocity_at(g + dz) - velocity_at(g - dz);
    let strain = sqrt(2.0 * (ux.x * ux.x + uz.y * uz.y) + (ux.y + uz.x) * (ux.y + uz.x)) * inv;
    let turbulence = breaking_turbulence(strain) * wet;
    let breaking = max(max(front, swash), turbulence);
    let foam = vec4<f32>(
        mix(base_f.r, breaking, w),
        mix(base_f.g, bw.y, w),
        mix(base_f.b, detail.displacement_z * fade, w),
        base_f.a,
    );
    textureStore(foam_out, gid.xy, slice, foam);
    let base_v = textureLoad(base_flow, gid.xy, slice, 0);
    let current = velocity_at(g);
    let flow = vec4<f32>(current, rise, 0.0);
    textureStore(flow_out, gid.xy, slice, mix(base_v, flow, w));
    let base_d = textureLoad(base_drift, gid.xy, slice, 0).xy;
    textureStore(drift_out, gid.xy, slice, vec4<f32>(mix(base_d, current, w), 0.0, 0.0));
}
