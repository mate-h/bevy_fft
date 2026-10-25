import bevy_pbr::render::mesh_functions::{
    get_world_from_local,
    get_visibility_range_dither_level,
    mesh_position_local_to_world,
};
import bevy_pbr::render::view_transformations::position_world_to_clip;
import bevy_fft::water::lod::{WaterLodSlices, lod_chop_dir, sample_lod};

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

struct WaterSurfaceUniform {
    foam_intensity: f32,
    vertex_lod_bias: f32,
    dry_depth: f32,
    _pad1: f32,
}

@group(constants::MATERIAL_BIND_GROUP) @binding(100) var<uniform> surface: WaterSurfaceUniform;
@group(constants::MATERIAL_BIND_GROUP) @binding(101) var<uniform> water_lod: WaterLodSlices;
@group(constants::MATERIAL_BIND_GROUP) @binding(102) var lod_displacement: texture_2d_array<f32>;
@group(constants::MATERIAL_BIND_GROUP) @binding(103) var lod_foam: texture_2d_array<f32>;

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
/// A = horizontal chop in meters along the chop direction. `bias` moves to coarser slices.
fn water_sample(xz: vec2<f32>, bias: f32) -> vec4<f32> {
    return sample_lod(lod_displacement, water_lod, xz, bias);
}

fn chop_dir() -> vec2<f32> {
    return lod_chop_dir(water_lod);
}

/// Undisplaced position whose chop lands on `xz`, found with one fixed-point step.
fn water_undisplace(xz: vec2<f32>) -> vec2<f32> {
    let chop = water_sample(xz, 0.0).a;
    return xz - chop * chop_dir();
}

fn water_normal_from_slopes(s: vec4<f32>) -> vec3<f32> {
    return normalize(vec3<f32>(-s.r, 1.0, -s.g));
}

struct DisplacedSurface {
    clip_position: vec4<f32>,
    world_position: vec4<f32>,
    world_normal: vec3<f32>,
}

/// Displaces the mesh in world space so any mesh layout, such as a plane or a clipmap, samples the same water.
/// Vertices read slightly coarser data than pixels so the mesh never undersamples it.
fn water_displace_surface(instance_index: u32, position_local: vec3<f32>) -> DisplacedSurface {
    var out: DisplacedSurface;
    var world: vec4<f32>;
    if quadtree.enabled != 0u {
        let q = water_quadtree_xz(position_local);
        if q.z == 0.0 {
            // Unused pool slot. Every vertex lands on one point outside the clip volume.
            out.clip_position = vec4<f32>(2.0, 2.0, 2.0, 1.0);
            out.world_position = vec4<f32>(0.0, 0.0, 0.0, 1.0);
            out.world_normal = vec3<f32>(0.0, 1.0, 0.0);
            return out;
        }
        world = vec4<f32>(q.x, 0.0, q.y, 1.0);
    } else {
        let world_from_local = get_world_from_local(instance_index);
        world = mesh_position_local_to_world(world_from_local, vec4<f32>(position_local, 1.0));
    }
    let s = water_sample(world.xz, surface.vertex_lod_bias);
    let w = chop_dir();
    world = vec4<f32>(world.x + s.a * w.x, world.y + s.b, world.z + s.a * w.y, 1.0);

    out.clip_position = position_world_to_clip(world.xyz);
    out.world_position = world;
    out.world_normal = water_normal_from_slopes(s);
    return out;
}

/// Per-pixel normal from the LOD data at the undisplaced position under this pixel.
fn water_pixel_normal(world_position: vec3<f32>) -> vec3<f32> {
    return water_normal_from_slopes(water_sample(water_undisplace(world_position.xz), 0.0));
}

fn water_foam_coverage(world_position: vec3<f32>) -> f32 {
    return sample_lod(lod_foam, water_lod, water_undisplace(world_position.xz), 0.0).r;
}

fn water_mix_foam_into_base_color(world_position: vec3<f32>, base_color: vec4<f32>) -> vec4<f32> {
    let foam_mix = saturate(water_foam_coverage(world_position) * surface.foam_intensity);
    return vec4<f32>(mix(base_color.rgb, vec3<f32>(1.0), foam_mix), base_color.a);
}

/// Water depth in meters from the depth lane.
fn water_depth(world_position: vec3<f32>) -> f32 {
    return sample_lod(lod_foam, water_lod, world_position.xz, 0.0).g;
}

/// True where `dry_depth` is set and the water is shallower, so the terrain below is drawn instead.
fn water_is_dry(world_position: vec3<f32>) -> bool {
    return surface.dry_depth > 0.0 && water_depth(world_position) < surface.dry_depth;
}

/// Share of the bed showing through at this pixel, from the water depth lane. One on dry ground,
/// fading out over the first few decimeters of water.
fn water_bed_visibility(world_position: vec3<f32>) -> f32 {
    return exp(-max(water_depth(world_position), 0.0) * 6.0);
}

/// Sand tint for shallow and dry areas with bed visibility `bed`. Regions without a shallow water
/// solver carry the open water depth, so `bed` is zero there and the color is unchanged.
fn water_mix_bed_into_base_color(bed: f32, base_color: vec4<f32>) -> vec4<f32> {
    let sand = vec3<f32>(0.52, 0.46, 0.32);
    return vec4<f32>(mix(base_color.rgb, sand, bed), base_color.a);
}

// `PREPASS_PIPELINE` is only set for actual prepass draw pipelines. The main mesh pipeline can still
// define `DEFERRED_PREPASS` (for example when sampling prepass normals) without
// `NORMAL_PREPASS_OR_DEFERRED_PREPASS`, in which case `prepass_io::VertexOutput` has no `world_normal`.
// Match Bevy's `prepass.wgsl`: use `prepass_io` only when `PREPASS_PIPELINE` is set, otherwise `forward_io`.

@vertex
fn vertex(vertex: Vertex) -> VertexOutput {
    let surf = water_displace_surface(vertex.instance_index, vertex.position);

@if(PREPASS_PIPELINE && NORMAL_PREPASS_OR_DEFERRED_PREPASS) {
    var out: VertexOutput;
    out.position = surf.clip_position;
    out.world_position = surf.world_position;
@if(VERTEX_UVS_A)
    out.uv = vertex.uv;
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
    out.uv = vertex.uv;
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
    out.uv = vertex.uv;
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

@if(!PREPASS_PIPELINE)
@fragment
fn fragment(in: FragVertexOutput, @builtin(front_facing) is_front: bool) -> FragmentOutput {
    if water_is_dry(in.world_position.xyz) {
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
    let n_world = water_pixel_normal(in.world_position.xyz);
    pbr_input.world_normal = prepare_world_normal(n_world, double_sided, is_front);
    pbr_input.N = normalize(pbr_input.world_normal);
    pbr_input.material.base_color = water_mix_foam_into_base_color(in.world_position.xyz, pbr_input.material.base_color);
    let bed = water_bed_visibility(in.world_position.xyz);
    pbr_input.material.base_color = water_mix_bed_into_base_color(bed, pbr_input.material.base_color);
    pbr_input.material.perceptual_roughness = mix(pbr_input.material.perceptual_roughness, 0.9, bed);
    var out: FragmentOutput;
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
    if water_is_dry(in.world_position.xyz) {
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
    let n_world = water_pixel_normal(in.world_position.xyz);
    pbr_input.world_normal = prepare_world_normal(n_world, double_sided, is_front);
    pbr_input.N = normalize(pbr_input.world_normal);
    pbr_input.material.base_color = water_mix_foam_into_base_color(in.world_position.xyz, pbr_input.material.base_color);
    let bed = water_bed_visibility(in.world_position.xyz);
    pbr_input.material.base_color = water_mix_bed_into_base_color(bed, pbr_input.material.base_color);
    pbr_input.material.perceptual_roughness = mix(pbr_input.material.perceptual_roughness, 0.9, bed);
    return deferred_output(in, pbr_input);
}
}
