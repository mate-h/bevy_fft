import bevy_pbr::render::mesh_functions::{
    get_world_from_local,
    get_visibility_range_dither_level,
    mesh_position_local_to_world,
};
import bevy_pbr::render::view_transformations::position_world_to_clip;
import bevy_fft::water::lod::{
    LodPick,
    WaterLodSlices,
    lod_coverage,
    lod_pick,
    sample_lod_filtered,
    sample_pick,
    sample_pick_filtered,
};
import bevy_fft::water::foam_flow::{
    FOAM_FLOW_LAYERS_PER_SLICE,
    FOAM_FLOW_TARGET_DEFORMATION,
    foam_flow_long_weight,
    foam_flow_weight,
};

@if(PREPASS_PIPELINE)
import bevy_pbr::prepass::io::{
    Vertex,
    VertexOutput,
};
@else
import bevy_pbr::render::forward_io::{
    Vertex,
    VertexOutput,
};

@if(!PREPASS_PIPELINE)
import bevy_pbr::render::pbr_fragment::pbr_input_from_standard_material;
@if(!PREPASS_PIPELINE)
import bevy_pbr::render::pbr_functions::alpha_discard;
@if(!PREPASS_PIPELINE)
import bevy_pbr::render::pbr_functions::prepare_world_normal;
@if(!PREPASS_PIPELINE)
import bevy_pbr::render::pbr_types;
@if(!PREPASS_PIPELINE)
import bevy_pbr::render::pbr_types::STANDARD_MATERIAL_FLAGS_DOUBLE_SIDED_BIT;
@if(!PREPASS_PIPELINE)
import bevy_pbr::render::pbr_types::STANDARD_MATERIAL_FLAGS_UNLIT_BIT;
@if(!PREPASS_PIPELINE)
import bevy_pbr::render::forward_io::{
    VertexOutput as FragVertexOutput,
    FragmentOutput,
};
@if(!PREPASS_PIPELINE)
import bevy_pbr::render::pbr_functions::{
    apply_pbr_lighting,
    main_pass_post_lighting_processing,
};

@if(PREPASS_PIPELINE && PREPASS_FRAGMENT)
import bevy_pbr::prepass::io::{
    VertexOutput as FragVertexOutput,
    FragmentOutput,
};
@if(PREPASS_PIPELINE && PREPASS_FRAGMENT && NORMAL_PREPASS_OR_DEFERRED_PREPASS)
import bevy_pbr::render::pbr_fragment::pbr_input_from_standard_material;
@if(PREPASS_PIPELINE && PREPASS_FRAGMENT && NORMAL_PREPASS_OR_DEFERRED_PREPASS)
import bevy_pbr::render::pbr_functions::alpha_discard;
@if(PREPASS_PIPELINE && PREPASS_FRAGMENT && NORMAL_PREPASS_OR_DEFERRED_PREPASS)
import bevy_pbr::render::pbr_functions::prepare_world_normal;
@if(PREPASS_PIPELINE && PREPASS_FRAGMENT && NORMAL_PREPASS_OR_DEFERRED_PREPASS)
import bevy_pbr::deferred::functions::deferred_output;
@if(PREPASS_PIPELINE && PREPASS_FRAGMENT && NORMAL_PREPASS_OR_DEFERRED_PREPASS)
import bevy_pbr::render::pbr_types::STANDARD_MATERIAL_FLAGS_DOUBLE_SIDED_BIT;

@if(!PREPASS_PIPELINE && VIEW_TRANSMISSION_TEXTURE)
import bevy_fft::water::optics::{optics, shade_water, water_point};
import bevy_fft::water::aeration::{AIR_PURE_WATER_ABSORPTION, aerated_layer};
@if(!PREPASS_PIPELINE)
import bevy_fft::water::underside::shade_underside;
@if(!PREPASS_PIPELINE)
import bevy_pbr::render::mesh_view_bindings::view;

struct WaterSurfaceUniform {
    foam_intensity: f32,
    vertex_lod_bias: f32,
    dry_depth: f32,
    time: f32,
    /// Height of the level plane the mesh displaces from.
    plane_height: f32,
    /// Root mean square slope of the flow detail at full strength.
    flow_detail: f32,
}

@group(constants::MATERIAL_BIND_GROUP) @binding(100) var<uniform> surface: WaterSurfaceUniform;
@group(constants::MATERIAL_BIND_GROUP) @binding(101) var<uniform> water_lod: WaterLodSlices;
@group(constants::MATERIAL_BIND_GROUP) @binding(102) var lod_displacement: texture_2d_array<f32>;
@group(constants::MATERIAL_BIND_GROUP) @binding(103) var lod_foam: texture_2d_array<f32>;
/// Linear, clamped to edge. Shared by the filterable LOD arrays.
@group(constants::MATERIAL_BIND_GROUP) @binding(105) var lod_sampler: sampler;
/// Active foam in `.r`, residual foam in `.g` and the air fraction of the water column in `.b`,
/// kept by the foam sim.
@group(constants::MATERIAL_BIND_GROUP) @binding(108) var lod_foam_state: texture_2d_array<f32>;

/// Tiling foam pattern whose values are evenly spread over [0, 1].
@group(constants::MATERIAL_BIND_GROUP) @binding(109) var foam_texture: texture_2d<f32>;
/// Linear and repeating.
@group(constants::MATERIAL_BIND_GROUP) @binding(110) var foam_sampler: sampler;
/// Advected texture coordinates of the foam pattern, laid out as `foam_flow.wesl` describes.
@group(constants::MATERIAL_BIND_GROUP) @binding(111) var lod_foam_flow: texture_2d_array<f32>;
/// Drift of floating matter in meters per second along X and Z.
@group(constants::MATERIAL_BIND_GROUP) @binding(115) var lod_drift: texture_2d_array<f32>;

/// Reflectance of dense foam. Koepke (1984, §III) takes about 55 percent from Whitlock et al.'s
/// laboratory foam of more than 25 bubble layers. Matches `FOAM_ALBEDO` in `optics.wesl` and
/// `underside.wesl`.
const FOAM_ALBEDO: f32 = 0.55;
/// Mean opacity of residual foam. Koepke (1984, §IV) measured foam streaks near 10 percent
/// reflectance, a single bubble layer, against 55 percent for dense foam.
const RESIDUAL_FOAM_OPACITY: f32 = 0.1 / 0.55;
/// Meters of water covered by one repeat of the foam texture. Matches `FOAM_FLOW_TILE` in
/// `foam_flow.wesl`.
const FOAM_TILE: f32 = 6.0;
/// Scale of the second, finer foam texture sample relative to the first.
const FOAM_DETAIL_SCALE: f32 = 0.37;
/// Texels along each edge of the foam texture. Matches `WATER_FOAM_TEXTURE_SIZE`.
const FOAM_TEXTURE_SIZE: f32 = 256.0;
/// Height in meters given to opaque foam when it bends the normal. An appearance choice.
const FOAM_RELIEF: f32 = 0.01;

struct WaterQuadtreeTile {
    /// Corner X and Z in meters, edge length in meters, level.
    rect: vec4<f32>,
    /// Distances where morphing to the next coarser grid starts and ends.
    morph: vec4<f32>,
}

struct WaterQuadtreeTiles {
    /// Viewer position the tiles were selected for.
    eye: vec4<f32>,
    count: u32,
    /// Grid cells along each tile edge.
    cells: u32,
    /// Zero when the mesh is not a quadtree tile pool.
    enabled: u32,
    _pad: u32,
    tiles: array<WaterQuadtreeTile>,
}

@group(constants::MATERIAL_BIND_GROUP) @binding(104) var<storage, read> quadtree: WaterQuadtreeTiles;

/// World XZ of a tile pool vertex. Local X and Z hold the grid coordinate and Y the pool slot.
/// Vertices near the far edge of the tile's range slide onto the next coarser grid, so the tile
/// meets coarser neighbours without cracks. The third lane is zero for unused slots.
fn water_quadtree_xz(position_local: vec3<f32>) -> vec3<f32> {
    let slot = u32(position_local.y + 0.5);
    if slot >= quadtree.count {
        return vec3<f32>(0.0, 0.0, 0.0);
    }
    let tile = quadtree.tiles[slot];
    let spacing = tile.rect.z / f32(quadtree.cells);
    let g = position_local.xz;
    let flat = tile.rect.xy + g * spacing;
    let d = distance(quadtree.eye.xyz, vec3<f32>(flat.x, 0.0, flat.y));
    let k = saturate((d - tile.morph.x) / max(tile.morph.y - tile.morph.x, 1e-4));
    let odd = fract(g * 0.5) * 2.0;
    let xz = tile.rect.xy + (g - odd * k) * spacing;
    return vec3<f32>(xz, 1.0);
}

/// Water LOD data at world `xz`: R/G = height slopes along X and Z, B = elevation in meters,
/// A = horizontal displacement X in meters. `bias` moves to coarser slices.
/// Past the coarsest slice the water is flat.
fn water_sample(xz: vec2<f32>, bias: f32) -> vec4<f32> {
    return sample_lod_filtered(lod_displacement, lod_sampler, water_lod, xz, bias) * lod_coverage(water_lod, xz);
}

/// Horizontal displacement Z in meters at world `xz`, kept in the foam texture's blue lane.
fn water_displacement_z(xz: vec2<f32>, bias: f32) -> f32 {
    return sample_lod_filtered(lod_foam, lod_sampler, water_lod, xz, bias).b * lod_coverage(water_lod, xz);
}

/// Undisplaced position whose horizontal displacement lands on `xz`, found with one fixed-point step.
fn water_undisplace(xz: vec2<f32>) -> vec2<f32> {
    return xz - vec2<f32>(water_sample(xz, 0.0).a, water_displacement_z(xz, 0.0));
}

fn water_normal_from_slopes(s: vec4<f32>) -> vec3<f32> {
    return normalize(vec3<f32>(-s.r, 1.0, -s.g));
}

struct DisplacedSurface {
    clip_position: vec4<f32>,
    world_position: vec4<f32>,
    world_normal: vec3<f32>,
    /// World XZ before horizontal displacement moves the vertex. The LOD data is indexed by this position.
    flat_xz: vec2<f32>,
}

/// Displaces the mesh in world space so any mesh layout, such as a plane or a clipmap, samples the same water.
/// Vertices read slightly coarser data than pixels so the mesh never undersamples it.
/// `spacing` is the vertex's world grid size in meters, or zero when the mesh has no such grid.
fn water_displace_surface(instance_index: u32, position_local: vec3<f32>, spacing: f32) -> DisplacedSurface {
    var out: DisplacedSurface;
    var world: vec4<f32>;
    if quadtree.enabled != 0u {
        let q = water_quadtree_xz(position_local);
        if q.z == 0.0 {
            // Unused pool slot. Every vertex lands on one point outside the clip volume.
            out.clip_position = vec4<f32>(2.0, 2.0, 2.0, 1.0);
            out.world_position = vec4<f32>(0.0, 0.0, 0.0, 1.0);
            out.world_normal = vec3<f32>(0.0, 1.0, 0.0);
            out.flat_xz = vec2<f32>(0.0);
            return out;
        }
        world = vec4<f32>(q.x, 0.0, q.y, 1.0);
    } else {
        let world_from_local = get_world_from_local(instance_index);
        world = mesh_position_local_to_world(world_from_local, vec4<f32>(position_local, 1.0));
        // Shift the whole ring onto its own lattice. The mesh origin steps more finely than the
        // outer rings. Every vertex of a ring carries the same spacing and moves together.
        // The skirt between rings uses both spacings, so it stretches across the gutter.
        if spacing > 0.0 {
            let origin = mesh_position_local_to_world(world_from_local, vec4<f32>(0.0, 0.0, 0.0, 1.0)).xz;
            let parked = round(origin / spacing) * spacing;
            let delta = parked - origin;
            world.x = world.x + delta.x;
            world.z = world.z + delta.y;
        }
    }
    out.flat_xz = world.xz;
    let s = water_sample(world.xz, surface.vertex_lod_bias);
    let dz = water_displacement_z(world.xz, surface.vertex_lod_bias);
    world = vec4<f32>(world.x + s.a, world.y + s.b, world.z + dz, 1.0);

    out.clip_position = position_world_to_clip(world.xyz);
    out.world_position = world;
    out.world_normal = water_normal_from_slopes(s);
    return out;
}

/// Per-pixel normal from the LOD data at the pixel's pick.
fn water_pixel_normal(pick: LodPick) -> vec3<f32> {
    return water_normal_from_slopes(sample_pick_filtered(lod_displacement, lod_sampler, pick) * pick.coverage);
}

/// Depth reported past the coarsest slice, deep enough that no bed shows through.
const OPEN_WATER_DEPTH: f32 = 1000.0;

/// What the LOD data says about the water at a pick, besides its shape.
struct WaterLodSurface {
    /// Active foam in `.x` and residual foam in `.y`.
    foam: vec2<f32>,
    /// Water depth in meters.
    depth: f32,
    /// Mean square slope of the waves too short for the slice.
    variance: f32,
    /// Air fraction of the water column.
    air: f32,
}

/// The LOD data at undisplaced position `xz`, read at its pick. Foam and air fade out past the
/// coarsest slice.
fn water_lod_surface(pick: LodPick) -> WaterLodSurface {
    let f = sample_pick_filtered(lod_foam, lod_sampler, pick);
    let state = sample_pick(lod_foam_state, pick, 1u, 0u);
    let c = pick.coverage;
    return WaterLodSurface(
        state.rg * c,
        mix(OPEN_WATER_DEPTH, f.g, c),
        max(f.a, 0.0) * c,
        max(state.b, 0.0) * c,
    );
}

/// `base_color` under a column `depth` meters deep holding air fraction `air`, for surfaces drawn
/// without the water optics: the bubbles' own color over the share of the water they hide.
fn water_mix_air_into_material(air: f32, depth: f32, base_color: vec4<f32>) -> vec4<f32> {
    let layer = aerated_layer(air, depth, AIR_PURE_WATER_ABSORPTION);
    return vec4<f32>(base_color.rgb * (1.0 - layer.opacity) + layer.reflectance, base_color.a);
}

/// Perceptual roughness of a surface of `perceptual_roughness` whose facets also carry waves of
/// mean square slope `variance` too short to resolve. As Bruneton et al. (2010) widen the sea's
/// reflection lobe by the variance of its unresolved slopes, the variances add: a GGX lobe's
/// width α is about its root mean square slope, so α² grows by the variance.
fn water_unresolved_roughness(perceptual_roughness: f32, variance: f32) -> f32 {
    let alpha = perceptual_roughness * perceptual_roughness;
    return sqrt(sqrt(alpha * alpha + variance));
}

/// Foam pattern at world `xz`, evenly spread over [0, 1]. Two samples at unrelated scales and
/// angles hide the repeat. The sum of two even values has a triangular histogram, and its
/// distribution function flattens it again. Where the finer sample is too small to resolve,
/// `detail_fade` hands over to the coarse sample alone, which is even by itself.
fn water_foam_pattern(xz: vec2<f32>, detail_fade: f32) -> f32 {
    let a = textureSampleLevel(foam_texture, foam_sampler, xz / FOAM_TILE, 0.0).r;
    let turn = mat2x2<f32>(0.8, 0.6, -0.6, 0.8);
    let detail_uv = turn * xz / (FOAM_TILE * FOAM_DETAIL_SCALE) + vec2<f32>(0.31, 0.17);
    let b = textureSampleLevel(foam_texture, foam_sampler, detail_uv, 0.0).r;
    let s = a + b;
    let both = select(1.0 - 0.5 * (2.0 - s) * (2.0 - s), 0.5 * s * s, s < 1.0);
    return mix(both, a, detail_fade);
}

/// One where pattern value `t` is above `1 - coverage`, with an edge up to `width` wide in pattern
/// units that antialiases the boundary without moving it. The edge narrows near zero and full
/// coverage so it stays inside the pattern's range, and no coverage gives no foam.
fn water_foam_mask(t: f32, coverage: f32, width: f32) -> f32 {
    let edge = max(min(width, 2.0 * min(coverage, 1.0 - coverage)), 1e-6);
    return select(saturate((t - (1.0 - coverage)) / edge + 0.5), 0.0, coverage <= 0.0);
}

/// Foam opacity for pattern value `t`. Active foam takes
/// the highest pattern values and is opaque. Residual foam takes the band below it, so it turns
/// to lace around the active cores as it thins. Its opacity grows with depth into the band, as a
/// thicker film holds more bubble layers.
fn water_foam_layers(t: f32, foam: vec2<f32>, width: f32) -> f32 {
    let total = min(foam.x + foam.y, 1.0);
    let dense = water_foam_mask(t, foam.x, width);
    let residual = max(water_foam_mask(t, total, width) - dense, 0.0);
    let depth = saturate((t - (1.0 - total)) / max(foam.y, 1e-4));
    return dense + residual * RESIDUAL_FOAM_OPACITY * (0.5 + depth);
}

struct WaterFoamSurface {
    /// Opacity of the foam layer over the water.
    opacity: f32,
    /// Slope of the foam relief along X and Z.
    slope: vec2<f32>,
}

/// Even [0, 1] value from the weighted sum `w a + (1 - w) b` of two independent even values. The
/// sum has a trapezoidal histogram, and this is its distribution function.
fn water_flatten_blend(a: f32, b: f32, w: f32) -> f32 {
    let p = max(w, 1.0 - w);
    let q = max(min(w, 1.0 - w), 1e-4);
    let x = w * a + (1.0 - w) * b;
    if x < q {
        return x * x / (2.0 * p * q);
    }
    if x > p {
        return 1.0 - (1.0 - x) * (1.0 - x) / (2.0 * p * q);
    }
    return (x - 0.5 * q) / p;
}

/// Advected texture offsets of the short and long layer at a pixel, and the long layer's weight.
struct WaterFoamFlow {
    short: vec4<f32>,
    long: vec4<f32>,
    long_weight: f32,
}

/// One flow array layer of the slice pair `sample_lod` would pick at `xz`.
fn water_foam_flow_layer(pick: LodPick, layer: u32) -> vec4<f32> {
    return sample_pick(lod_foam_flow, pick, FOAM_FLOW_LAYERS_PER_SLICE, layer);
}

fn water_foam_flow(pick: LodPick) -> WaterFoamFlow {
    var out: WaterFoamFlow;
    out.short = water_foam_flow_layer(pick, 0u);
    out.long = water_foam_flow_layer(pick, 1u);
    out.long_weight = foam_flow_long_weight(water_foam_flow_layer(pick, 2u));
    return out;
}

/// Pattern of one flow layer, whose two phases are read at `offsets.xy` and `offsets.zw` and blended
/// by the phase weights, so each phase starts over unseen.
fn water_foam_flow_layer_pattern(
    xz: vec2<f32>,
    offsets: vec4<f32>,
    layer: u32,
    detail_fade: f32,
) -> f32 {
    let a = water_foam_pattern(xz + offsets.xy, detail_fade);
    let b = water_foam_pattern(xz + offsets.zw, detail_fade);
    return water_flatten_blend(a, b, foam_flow_weight(surface.time, layer));
}

/// Pattern value at `xz` read through the advected texture coordinates of Neyret (2003). The foam
/// sim carries them with the drift, so the pattern follows the foam the sim advects instead of
/// standing still under it. Calm drift reads the long layer and strained drift the short one, and
/// every blend keeps the histogram flat, so contrast and coverage do not pulse.
fn water_foam_flow_pattern(xz: vec2<f32>, flow: WaterFoamFlow, detail_fade: f32) -> f32 {
    if flow.long_weight >= 1.0 {
        return water_foam_flow_layer_pattern(xz, flow.long, 1u, detail_fade);
    }
    let short = water_foam_flow_layer_pattern(xz, flow.short, 0u, detail_fade);
    if flow.long_weight <= 0.0 {
        return short;
    }
    let long = water_foam_flow_layer_pattern(xz, flow.long, 1u, detail_fade);
    return water_flatten_blend(long, short, flow.long_weight);
}

/// Active plus residual foam at or below this counts as none, so most of the water skips the
/// foam pattern. Coverage this low lights about one pixel in ten thousand.
const FOAM_SKIP_TOTAL: f32 = 1e-4;

/// True when `v` is nonzero at any pixel of this pixel's 2x2 quad. Fine derivatives give each
/// pixel the other three values, and differences of zeros are exactly zero, so all four pixels
/// agree. A branch on the result keeps the derivatives taken inside it defined.
fn water_quad_any(v: f32) -> bool {
    let dx = dpdxFine(v);
    let dy = dpdyFine(v);
    return abs(v) + abs(dx) + abs(dy) + abs(dpdyFine(dx)) > 0.0;
}

/// Foam at undisplaced position `xz` from the sim's active and residual coverage. Foam edges are
/// one pixel wide. Where a pixel spans several texels of a pattern sample, that sample would
/// alias, so the finer sample drops out first and the foam then fades to its mean opacity.
fn water_foam_surface(xz: vec2<f32>, pick: LodPick, foam: vec2<f32>) -> WaterFoamSurface {
    var out: WaterFoamSurface;
    let total = foam.x + foam.y;
    if !water_quad_any(select(total, 0.0, total <= FOAM_SKIP_TOTAL)) {
        return out;
    }
    let texel = FOAM_TILE / FOAM_TEXTURE_SIZE;
    let detail_texel = texel * FOAM_DETAIL_SCALE;
    // Geometric mean of the pixel's extent along both screen axes, so grazing views keep the lace
    // across the view direction.
    let footprint = sqrt(max(length(dpdx(xz)) * length(dpdy(xz)), 1e-12));
    let detail_fade = smoothstep(1.0, 4.0, footprint / detail_texel);
    let fade = smoothstep(1.0, 4.0, footprint / texel);
    let step = mix(detail_texel, texel, detail_fade);
    let flow = water_foam_flow(pick);
    let t = water_foam_flow_pattern(xz, flow, detail_fade);
    let width = max(fwidth(t), 1e-4);
    let here = water_foam_layers(t, foam, width);
    let x = water_foam_flow_pattern(xz + vec2<f32>(step, 0.0), flow, detail_fade);
    let z = water_foam_flow_pattern(xz + vec2<f32>(0.0, step), flow, detail_fade);
    let along = vec2<f32>(water_foam_layers(x, foam, width), water_foam_layers(z, foam, width));
    let mean = foam.x + foam.y * RESIDUAL_FOAM_OPACITY;
    out.slope = FOAM_RELIEF * (along - here) / step * (1.0 - fade);
    out.opacity = saturate(mix(here, mean, fade) * surface.foam_intensity);
    return out;
}

/// Slopes the foam texture's green and blue channels span either side of zero, in root mean
/// squares. Matches `RIPPLE_SLOPE_RANGE` in `foam_texture.rs`.
const FLOW_DETAIL_SLOPE_RANGE: f32 = 4.0;
/// Drift speeds in meters per second over which the flow detail grows from none to full. The
/// ocean's Stokes drift stays below the first, so the detail shows on currents.
const FLOW_DETAIL_SPEED: vec2<f32> = vec2<f32>(0.15, 1.0);

/// Ripple slopes at `xz`, with a root mean square of one.
fn water_ripple_slopes(xz: vec2<f32>) -> vec2<f32> {
    let gb = textureSampleLevel(foam_texture, foam_sampler, xz / FOAM_TILE, 0.0).gb;
    return (gb - 0.5) * (2.0 * FLOW_DETAIL_SLOPE_RANGE);
}

/// Weighted sum of two independent fields with unit variance, scaled back to unit variance, so the
/// ripples keep their contrast through a blend.
fn water_blend_slopes(a: vec2<f32>, b: vec2<f32>, w: f32) -> vec2<f32> {
    return (w * a + (1.0 - w) * b) * inverseSqrt(max(w * w + (1.0 - w) * (1.0 - w), 1e-4));
}

/// Ripple slopes of one flow layer, read through the advected offsets of its two phases.
fn water_flow_detail_layer(xz: vec2<f32>, offsets: vec4<f32>, layer: u32) -> vec2<f32> {
    let a = water_ripple_slopes(xz + offsets.xy);
    let b = water_ripple_slopes(xz + offsets.zw);
    return water_blend_slopes(a, b, foam_flow_weight(surface.time, layer));
}

/// Slopes of the flow detail at undisplaced `xz`: short ripples that move with the water. They
/// are read through the same advected texture coordinates as the foam (Neyret 2003), and grow
/// with the drift speed and, as Chentanez and Müller-Fischer (2010) weight their subgrid waves,
/// with the strain the flow puts on them, from half to twice their strength. `footprint` is the
/// pixel's width in meters; ripples too small for it fade out.
fn water_flow_detail(xz: vec2<f32>, pick: LodPick, footprint: f32) -> vec2<f32> {
    if surface.flow_detail <= 0.0 {
        return vec2<f32>(0.0);
    }
    let drift = sample_pick(lod_drift, pick, 1u, 0u).xy * pick.coverage;
    let speed = smoothstep(FLOW_DETAIL_SPEED.x, FLOW_DETAIL_SPEED.y, length(drift));
    let texel = FOAM_TILE / FOAM_TEXTURE_SIZE;
    let resolved = 1.0 - smoothstep(2.0, 8.0, footprint / texel);
    let strength = surface.flow_detail * speed * resolved;
    if strength <= 0.0 {
        return vec2<f32>(0.0);
    }
    let flow = water_foam_flow(pick);
    let d = water_foam_flow_layer(pick, 2u);
    let strain = clamp(0.5 + (d.x + d.y) / FOAM_FLOW_TARGET_DEFORMATION, 0.5, 2.0);
    var slopes = water_flow_detail_layer(xz, flow.long, 1u);
    if flow.long_weight < 1.0 {
        let short = water_flow_detail_layer(xz, flow.short, 0u);
        slopes = water_blend_slopes(slopes, short, flow.long_weight);
    }
    return strength * strain * slopes;
}

/// `wave_normal` with the flow detail's `slopes` added. Surface gradients add: the ripples ride on
/// the waves, and the foam relief later rides on both.
fn water_detail_normal(wave_normal: vec3<f32>, slopes: vec2<f32>) -> vec3<f32> {
    return normalize(wave_normal / max(wave_normal.y, 1e-3) - vec3<f32>(slopes.x, 0.0, slopes.y));
}

/// Meters of water covered by one repeat of the foam texture where it textures the air of white
/// water. At twice the foam's tile its larger cells are a meter across, near the size of the boils
/// and air packets a roller sheds. An appearance choice.
const AIR_TILE: f32 = 12.0;
/// Seconds of drift over which the air pattern is smeared along the flow. Each air packet leaves a
/// trail of bubbles as the current carries it, so the pattern draws streaks that grow with the
/// speed. An appearance choice.
const AIR_STREAK_TIME: f32 = 0.6;
/// Longest streak in meters.
const AIR_STREAK_MAX: f32 = 2.5;
/// Pattern samples along each streak.
const AIR_STREAK_TAPS: u32 = 4u;
/// Size of the coarse air pattern relative to the streaked one.
const AIR_COARSE_SCALE: f32 = 2.7;
/// Standard deviation of the air pattern before its contrast is set, which brings it to unit
/// spread. One sample of the foam texture is even over [0, 1], a deviation of 0.29. Averaging
/// along a streak lowers that toward 0.14 as the taps decorrelate, and with the coarse sample the
/// pattern lands between 0.23 and 0.29.
const AIR_PATTERN_SPREAD: f32 = 0.25;
/// How far the air fraction swings either side of its mean across the pattern, in units of the
/// mean. Air in a roller comes in packets and clusters, which a probe in the flow sees as an
/// intermittent signal rather than a steady fraction. An appearance choice.
const AIR_TEXTURE_CONTRAST: f32 = 0.85;
/// Air fractions at or below this skip the pattern.
const AIR_SKIP: f32 = 1e-3;

/// Air pattern of one phase, centred on zero: the foam texture averaged over a streak along
/// `along`, read at the advected texture coordinate `xz + offset`, plus one sample of it at
/// `AIR_COARSE_SCALE` times the size, so the air also gathers into patches several meters across
/// that still show from a distance. The sum is scaled back to the spread of one sample.
fn water_air_streak(xz: vec2<f32>, offset: vec2<f32>, along: vec2<f32>) -> f32 {
    let p = xz + offset;
    var sum = 0.0;
    for (var k = 0u; k < AIR_STREAK_TAPS; k++) {
        let s = (f32(k) + 0.5) / f32(AIR_STREAK_TAPS) - 0.5;
        sum += textureSampleLevel(foam_texture, foam_sampler, (p + along * s) / AIR_TILE, 0.0).r;
    }
    let turn = mat2x2<f32>(0.6, 0.8, -0.8, 0.6);
    let coarse_uv = turn * p / (AIR_TILE * AIR_COARSE_SCALE) + vec2<f32>(0.53, 0.29);
    let coarse = textureSampleLevel(foam_texture, foam_sampler, coarse_uv, 0.0).r;
    return (sum / f32(AIR_STREAK_TAPS) - 0.5 + coarse - 0.5) * inverseSqrt(2.0);
}

/// Air pattern of one flow layer, its two phases blended so the spread does not pulse as they
/// trade places.
fn water_air_layer(xz: vec2<f32>, offsets: vec4<f32>, layer: u32, along: vec2<f32>) -> f32 {
    let a = water_air_streak(xz, offsets.xy, along);
    let b = water_air_streak(xz, offsets.zw, along);
    let w = foam_flow_weight(surface.time, layer);
    return (w * a + (1.0 - w) * b) * inverseSqrt(max(w * w + (1.0 - w) * (1.0 - w), 1e-4));
}

/// Air fraction at undisplaced `xz`, where the column holds `air` on average. The foam sim carries
/// only the mean, so the air is spread into packets and streaks by the foam texture's cells, read
/// through the advected texture coordinates of Neyret (2003) that the foam uses, so the pattern
/// moves with the water. Along the flow each sample averages a streak of the pattern in texture
/// space, which moves with the coordinates, so the streaks ride the current instead of standing in
/// it. The mean air stays the same. `footprint` is the pixel's width in meters; where it covers the
/// pattern's cells the air falls back to its mean.
fn water_air_texture(xz: vec2<f32>, pick: LodPick, air: f32, footprint: f32) -> f32 {
    if !water_quad_any(select(air, 0.0, air <= AIR_SKIP)) {
        return air;
    }
    let texel = AIR_TILE / FOAM_TEXTURE_SIZE;
    let resolved = 1.0 - smoothstep(3.0, 12.0, footprint / texel);
    if resolved <= 0.0 {
        return air;
    }
    var along = sample_pick(lod_drift, pick, 1u, 0u).xy * AIR_STREAK_TIME;
    along *= min(1.0, AIR_STREAK_MAX / max(length(along), 1e-6));
    let flow = water_foam_flow(pick);
    var z = water_air_layer(xz, flow.long, 1u, along);
    if flow.long_weight < 1.0 {
        let short = water_air_layer(xz, flow.short, 0u, along);
        let w = flow.long_weight;
        z = (w * z + (1.0 - w) * short) * inverseSqrt(max(w * w + (1.0 - w) * (1.0 - w), 1e-4));
    }
    // A lognormal factor whose mean is one for a pattern of unit spread, so the mean air stays.
    let k = AIR_TEXTURE_CONTRAST * resolved;
    return air * exp(k * z / AIR_PATTERN_SPREAD - 0.5 * k * k);
}

/// Bends `n` by the foam relief.
fn water_foam_normal(n: vec3<f32>, foam: WaterFoamSurface) -> vec3<f32> {
    return normalize(n - vec3<f32>(foam.slope.x, 0.0, foam.slope.y));
}

/// Foam adds diffuse reflection over the water, as Koepke (1984, §III) treats whitecaps as
/// isotropic reflectors. The water between the bubbles keeps its specular reflection.
fn water_mix_foam_into_material(foam: f32, base_color: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(mix(base_color.rgb, vec3<f32>(FOAM_ALBEDO), foam), base_color.a);
}

/// True where `dry_depth` is set and the water is shallower, so the terrain below is drawn instead.
fn water_is_dry(depth: f32) -> bool {
    return surface.dry_depth > 0.0 && depth < surface.dry_depth;
}

// `PREPASS_PIPELINE` is only set for actual prepass draw pipelines. The main mesh pipeline can still
// define `DEFERRED_PREPASS` (for example when sampling prepass normals) without
// `NORMAL_PREPASS_OR_DEFERRED_PREPASS`, in which case `prepass_io::VertexOutput` has no `world_normal`.
// Match Bevy's `prepass.wgsl`: use `prepass_io` only when `PREPASS_PIPELINE` is set, otherwise `forward_io`.

@vertex
fn vertex(vertex: Vertex) -> VertexOutput {
    var spacing = 0.0;
    @if(VERTEX_UVS_B) {
        spacing = vertex.uv_b.x;
    }
    let surf = water_displace_surface(vertex.instance_index, vertex.position, spacing);

@if(PREPASS_PIPELINE && NORMAL_PREPASS_OR_DEFERRED_PREPASS) {
    var out: VertexOutput;
    out.position = surf.clip_position;
    out.world_position = surf.world_position;
@if(VERTEX_UVS_A)
    out.uv = surf.flat_xz;
@if(VERTEX_UVS_B)
    out.uv_b = vertex.uv_b;
    out.world_normal = surf.world_normal;
@if(VERTEX_OUTPUT_INSTANCE_INDEX)
    out.instance_index = vertex.instance_index;
@if(MOTION_VECTOR_PREPASS)
    out.previous_world_position = surf.world_position;
@if(VERTEX_COLORS)
    out.color = vertex.color;
@if(VISIBILITY_RANGE_DITHER)
    out.visibility_range_dither = get_visibility_range_dither_level(
        vertex.instance_index,
        get_world_from_local(vertex.instance_index)[3],
    );
    return out;
} @elif(PREPASS_PIPELINE) {
    var out: VertexOutput;
    out.position = surf.clip_position;
    out.world_position = surf.world_position;
@if(VERTEX_UVS_A)
    out.uv = surf.flat_xz;
@if(VERTEX_UVS_B)
    out.uv_b = vertex.uv_b;
@if(VERTEX_OUTPUT_INSTANCE_INDEX)
    out.instance_index = vertex.instance_index;
@if(MOTION_VECTOR_PREPASS)
    out.previous_world_position = surf.world_position;
@if(VERTEX_COLORS)
    out.color = vertex.color;
@if(UNCLIPPED_DEPTH_ORTHO_EMULATION) {
    out.unclipped_depth = surf.clip_position.z;
    out.position.z = min(out.position.z, 1.0);
}
@if(VISIBILITY_RANGE_DITHER)
    out.visibility_range_dither = get_visibility_range_dither_level(
        vertex.instance_index,
        get_world_from_local(vertex.instance_index)[3],
    );
    return out;
} @else {
    var out: VertexOutput;
    out.position = surf.clip_position;
    out.world_position = surf.world_position;
    out.world_normal = surf.world_normal;
@if(VERTEX_UVS_A)
    out.uv = surf.flat_xz;
@if(VERTEX_UVS_B)
    out.uv_b = vertex.uv_b;
@if(VERTEX_OUTPUT_INSTANCE_INDEX)
    out.instance_index = vertex.instance_index;
@if(VERTEX_COLORS)
    out.color = vertex.color;
@if(VISIBILITY_RANGE_DITHER)
    out.visibility_range_dither = get_visibility_range_dither_level(
        vertex.instance_index,
        get_world_from_local(vertex.instance_index)[3],
    );
    return out;
}
}

/// Undisplaced XZ under this pixel. The vertex stage passes it through `uv` when the mesh has
/// UVs. Otherwise one fixed-point step recovers it from the displaced position.
@if(!PREPASS_PIPELINE || PREPASS_FRAGMENT)
fn water_flat_xz(in: FragVertexOutput) -> vec2<f32> {
    @if(VERTEX_UVS_A) {
        return in.uv;
    }
    @if(!VERTEX_UVS_A) {
        return water_undisplace(in.world_position.xz);
    }
}

@if(!PREPASS_PIPELINE)
@fragment
fn fragment(in: FragVertexOutput, @builtin(front_facing) is_front: bool) -> FragmentOutput {
    let xz = water_flat_xz(in);
    let pick = lod_pick(water_lod, xz, 0.0);
    let lod_surface = water_lod_surface(pick);
    let foam = water_foam_surface(xz, pick, lod_surface.foam);
    let footprint = sqrt(max(length(dpdx(xz)) * length(dpdy(xz)), 1e-12));
    let air = water_air_texture(xz, pick, lod_surface.air, footprint);
    if water_is_dry(lod_surface.depth) {
        discard;
    }
    var pbr_input = pbr_input_from_standard_material(in, is_front);
    pbr_input.material.base_color = alpha_discard(
        pbr_input.material.flags,
        pbr_input.material.alpha_cutoff,
        pbr_input.material.base_color
    );
    let double_sided =
        (pbr_input.material.flags & STANDARD_MATERIAL_FLAGS_DOUBLE_SIDED_BIT) != 0u;
    let wave_normal = water_pixel_normal(pick);
    let detail = water_flow_detail(xz, pick, footprint);
    let n_world = water_foam_normal(water_detail_normal(wave_normal, detail), foam);
    pbr_input.world_normal = prepare_world_normal(n_world, double_sided, is_front);
    pbr_input.N = normalize(pbr_input.world_normal);
    pbr_input.material.base_color = water_mix_foam_into_material(
        foam.opacity,
        water_mix_air_into_material(air, lod_surface.depth, pbr_input.material.base_color),
    );
    pbr_input.material.perceptual_roughness =
        water_unresolved_roughness(pbr_input.material.perceptual_roughness, lod_surface.variance);
    var out: FragmentOutput;
    // Back faces are seen from under the water. They draw only when the material leaves them unculled.
    if !is_front {
        // The foam relief shades the foam seen from above. From below the foam is a diffuse
        // layer, so the surface keeps the wave normal.
        let radiance = shade_underside(
            in.world_position.xyz,
            view.world_position,
            wave_normal,
            surface.plane_height,
            lod_surface.depth,
            pbr_input.material.perceptual_roughness,
            foam.opacity,
            air,
            in.position.xy,
        );
        out.color = main_pass_post_lighting_processing(pbr_input, vec4<f32>(radiance, 1.0));
        return out;
    }
@if(VIEW_TRANSMISSION_TEXTURE) {
    if optics.enabled != 0u {
        out.color = shade_water(
            water_point(pbr_input),
            lod_surface.depth,
            foam.opacity,
            lod_surface.foam.x,
            air,
            surface.plane_height,
        );
        out.color = main_pass_post_lighting_processing(pbr_input, out.color);
        return out;
    }
}
    if (pbr_input.material.flags & STANDARD_MATERIAL_FLAGS_UNLIT_BIT) == 0u {
        out.color = apply_pbr_lighting(pbr_input);
    } else {
        out.color = pbr_input.material.base_color;
    }
    out.color = main_pass_post_lighting_processing(pbr_input, out.color);
    return out;
}

@if(PREPASS_PIPELINE && PREPASS_FRAGMENT)
@fragment
fn fragment(in: FragVertexOutput, @builtin(front_facing) is_front: bool) -> FragmentOutput {
    let xz = water_flat_xz(in);
    let pick = lod_pick(water_lod, xz, 0.0);
    let lod_surface = water_lod_surface(pick);
    if water_is_dry(lod_surface.depth) {
        discard;
    }
@if(!NORMAL_PREPASS_OR_DEFERRED_PREPASS) {
    var out: FragmentOutput;
@if(UNCLIPPED_DEPTH_ORTHO_EMULATION)
    out.frag_depth = in.position.z;
    return out;
} @else {
    var pbr_input = pbr_input_from_standard_material(in, is_front);
    pbr_input.material.base_color = alpha_discard(
        pbr_input.material.flags,
        pbr_input.material.alpha_cutoff,
        pbr_input.material.base_color
    );
    let double_sided =
        (pbr_input.material.flags & STANDARD_MATERIAL_FLAGS_DOUBLE_SIDED_BIT) != 0u;
    let foam = water_foam_surface(xz, pick, lod_surface.foam);
    let footprint = sqrt(max(length(dpdx(xz)) * length(dpdy(xz)), 1e-12));
    let air = water_air_texture(xz, pick, lod_surface.air, footprint);
    let detail = water_flow_detail(xz, pick, footprint);
    let n_world = water_foam_normal(water_detail_normal(water_pixel_normal(pick), detail), foam);
    pbr_input.world_normal = prepare_world_normal(n_world, double_sided, is_front);
    pbr_input.N = normalize(pbr_input.world_normal);
    pbr_input.material.base_color = water_mix_foam_into_material(
        foam.opacity,
        water_mix_air_into_material(air, lod_surface.depth, pbr_input.material.base_color),
    );
    // The G-buffer carries the roughness to the deferred lighting pass, so the unresolved slopes
    // widen the glitter there as they do in the forward pass.
    pbr_input.material.perceptual_roughness =
        water_unresolved_roughness(pbr_input.material.perceptual_roughness, lod_surface.variance);
    return deferred_output(in, pbr_input);
}
}
