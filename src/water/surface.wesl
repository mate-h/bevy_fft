import bevy_pbr::render::mesh_functions::{
    get_world_from_local,
    get_visibility_range_dither_level,
    mesh_position_local_to_world,
};
import bevy_pbr::render::view_transformations::position_world_to_clip;
import bevy_fft::water::lod::{WaterLodSlices, lod_coverage, sample_lod, sample_lod_filtered};

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
import bevy_fft::water::optics::{optics, shade_water};
@if(!PREPASS_PIPELINE)
import bevy_fft::water::underside::shade_underside;
@if(!PREPASS_PIPELINE)
import bevy_pbr::render::mesh_view_bindings::view;

struct WaterSurfaceUniform {
    foam_intensity: f32,
    vertex_lod_bias: f32,
    dry_depth: f32,
    time: f32,
}

@group(constants::MATERIAL_BIND_GROUP) @binding(100) var<uniform> surface: WaterSurfaceUniform;
@group(constants::MATERIAL_BIND_GROUP) @binding(101) var<uniform> water_lod: WaterLodSlices;
@group(constants::MATERIAL_BIND_GROUP) @binding(102) var lod_displacement: texture_2d_array<f32>;
@group(constants::MATERIAL_BIND_GROUP) @binding(103) var lod_foam: texture_2d_array<f32>;
/// Linear, clamped to edge. Shared by the filterable LOD arrays.
@group(constants::MATERIAL_BIND_GROUP) @binding(105) var lod_sampler: sampler;
/// Active foam in `.r` and residual foam in `.g`, kept by the foam sim.
@group(constants::MATERIAL_BIND_GROUP) @binding(108) var lod_foam_state: texture_2d_array<f32>;

/// Tiling foam pattern whose values are evenly spread over [0, 1].
@group(constants::MATERIAL_BIND_GROUP) @binding(109) var foam_texture: texture_2d<f32>;
/// Linear and repeating.
@group(constants::MATERIAL_BIND_GROUP) @binding(110) var foam_sampler: sampler;
/// Drift of floating matter in meters per second along X and Z.
@group(constants::MATERIAL_BIND_GROUP) @binding(111) var lod_drift: texture_2d_array<f32>;

/// Reflectance of dense foam. Koepke (1984, §III) takes about 55 percent from Whitlock et al.'s
/// laboratory foam of more than 25 bubble layers. Matches `FOAM_ALBEDO` in `optics.wesl`.
const FOAM_ALBEDO: f32 = 0.55;
/// Mean opacity of residual foam. Koepke (1984, §IV) measured foam streaks near 10 percent
/// reflectance, a single bubble layer, against 55 percent for dense foam.
const RESIDUAL_FOAM_OPACITY: f32 = 0.1 / 0.55;
/// Meters of water covered by one repeat of the foam texture.
const FOAM_TILE: f32 = 6.0;
/// Scale of the second, finer foam texture sample relative to the first.
const FOAM_DETAIL_SCALE: f32 = 0.37;
/// Texels along each edge of the foam texture. Matches `WATER_FOAM_TEXTURE_SIZE`.
const FOAM_TEXTURE_SIZE: f32 = 256.0;
/// Height in meters given to opaque foam when it bends the normal. An appearance choice.
const FOAM_RELIEF: f32 = 0.01;
/// Seconds each flow phase carries the foam texture before it starts over. Divides the hour
/// after which the time wraps.
const FOAM_FLOW_PERIOD: f32 = 4.0;

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

/// Per-pixel normal from the LOD data at undisplaced position `xz`.
fn water_pixel_normal(xz: vec2<f32>) -> vec3<f32> {
    return water_normal_from_slopes(water_sample(xz, 0.0));
}

/// Depth reported past the coarsest slice, deep enough that no bed shows through.
const OPEN_WATER_DEPTH: f32 = 1000.0;

/// Active foam in `.x`, residual foam in `.y` and water depth in meters in `.z` at undisplaced
/// position `xz`. Foam fades out past the coarsest slice.
fn water_foam_and_depth(xz: vec2<f32>) -> vec3<f32> {
    let f = sample_lod_filtered(lod_foam, lod_sampler, water_lod, xz, 0.0);
    let state = sample_lod(lod_foam_state, water_lod, xz, 0.0);
    let c = lod_coverage(water_lod, xz);
    return vec3<f32>(state.rg * c, mix(OPEN_WATER_DEPTH, f.g, c));
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

/// Where flow phase `phase` reads the foam pattern for undisplaced `xz`. The phase moves the
/// pattern with the drift and starts over at a fresh random offset each period, while its weight
/// is zero. The two phases draw their offsets from separate sequences, so they never coincide.
fn water_foam_flow_uv(xz: vec2<f32>, drift: vec2<f32>, phase_time: f32, phase: f32) -> vec2<f32> {
    let cycle = floor(phase_time / FOAM_FLOW_PERIOD);
    let age = phase_time - cycle * FOAM_FLOW_PERIOD;
    let n = 2.0 * cycle + phase;
    let seed = vec2<f32>(fract(sin(n * 12.9898) * 43758.547), fract(sin(n * 78.233) * 43758.547));
    return xz - drift * age + seed * FOAM_TILE;
}

/// Pattern value at `xz` from two flow phases half a period apart, after Neyret (2003). Each phase
/// carries the texture with the drift and restarts when its weight reaches zero, so the pattern
/// follows the foam the sim advects instead of standing still under it. The blend keeps the
/// histogram flat, so contrast and coverage do not pulse.
fn water_foam_flow_pattern(xz: vec2<f32>, drift: vec2<f32>, detail_fade: f32) -> f32 {
    let t0 = surface.time;
    let t1 = surface.time + 0.5 * FOAM_FLOW_PERIOD;
    let w = 1.0 - abs(2.0 * fract(t0 / FOAM_FLOW_PERIOD) - 1.0);
    let a = water_foam_pattern(water_foam_flow_uv(xz, drift, t0, 0.0), detail_fade);
    let b = water_foam_pattern(water_foam_flow_uv(xz, drift, t1, 1.0), detail_fade);
    return water_flatten_blend(a, b, w);
}

/// Foam at undisplaced position `xz` from the sim's active and residual coverage. Foam edges are
/// one pixel wide. Where a pixel spans several texels of a pattern sample, that sample would
/// alias, so the finer sample drops out first and the foam then fades to its mean opacity.
fn water_foam_surface(xz: vec2<f32>, foam: vec2<f32>) -> WaterFoamSurface {
    var out: WaterFoamSurface;
    let texel = FOAM_TILE / FOAM_TEXTURE_SIZE;
    let detail_texel = texel * FOAM_DETAIL_SCALE;
    // Geometric mean of the pixel's extent along both screen axes, so grazing views keep the lace
    // across the view direction.
    let footprint = sqrt(max(length(dpdx(xz)) * length(dpdy(xz)), 1e-12));
    let detail_fade = smoothstep(1.0, 4.0, footprint / detail_texel);
    let fade = smoothstep(1.0, 4.0, footprint / texel);
    let step = mix(detail_texel, texel, detail_fade);
    let drift = sample_lod(lod_drift, water_lod, xz, 0.0).xy * lod_coverage(water_lod, xz);
    let t = water_foam_flow_pattern(xz, drift, detail_fade);
    let width = max(fwidth(t), 1e-4);
    let here = water_foam_layers(t, foam, width);
    let x = water_foam_flow_pattern(xz + vec2<f32>(step, 0.0), drift, detail_fade);
    let z = water_foam_flow_pattern(xz + vec2<f32>(0.0, step), drift, detail_fade);
    let along = vec2<f32>(water_foam_layers(x, foam, width), water_foam_layers(z, foam, width));
    let mean = foam.x + foam.y * RESIDUAL_FOAM_OPACITY;
    out.slope = FOAM_RELIEF * (along - here) / step * (1.0 - fade);
    out.opacity = saturate(mix(here, mean, fade) * surface.foam_intensity);
    return out;
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
    let foam_depth = water_foam_and_depth(xz);
    let foam = water_foam_surface(xz, foam_depth.xy);
    if water_is_dry(foam_depth.z) {
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
    let n_world = water_foam_normal(water_pixel_normal(xz), foam);
    pbr_input.world_normal = prepare_world_normal(n_world, double_sided, is_front);
    pbr_input.N = normalize(pbr_input.world_normal);
    pbr_input.material.base_color = water_mix_foam_into_material(foam.opacity, pbr_input.material.base_color);
    var out: FragmentOutput;
    // Back faces are seen from under the water. They draw only when the material leaves them unculled.
    if !is_front {
        let radiance = shade_underside(in.world_position.xyz, view.world_position, n_world, foam_depth.z, in.position.xy);
        out.color = main_pass_post_lighting_processing(pbr_input, vec4<f32>(radiance, 1.0));
        return out;
    }
@if(VIEW_TRANSMISSION_TEXTURE) {
    if optics.enabled != 0u {
        out.color = shade_water(pbr_input, foam_depth.z, foam.opacity, foam_depth.x);
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
    let foam_depth = water_foam_and_depth(xz);
    if water_is_dry(foam_depth.z) {
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
    let foam = water_foam_surface(xz, foam_depth.xy);
    let n_world = water_foam_normal(water_pixel_normal(xz), foam);
    pbr_input.world_normal = prepare_world_normal(n_world, double_sided, is_front);
    pbr_input.N = normalize(pbr_input.world_normal);
    pbr_input.material.base_color = water_mix_foam_into_material(foam.opacity, pbr_input.material.base_color);
    return deferred_output(in, pbr_input);
}
}
