import bevy_pbr::render::mesh_functions::{
    get_world_from_local,
    get_visibility_range_dither_level,
    mesh_position_local_to_world,
};
import bevy_pbr::render::view_transformations::position_world_to_clip;
import bevy_fft::ocean::sampling::bilinear_wrap;

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

struct OceanExtensionUniform {
    /// World-space tile edge in meters for cascades 0 to 3.
    cascade_tile_sizes: vec4<f32>,
    amplitude: f32,
    choppiness: f32,
    wind_direction: f32,
    cascade_count: u32,
    foam_intensity: f32,
    foam_cutoff: f32,
    foam_falloff: f32,
    foam_trail_decay: f32,
    lod_base_spacing: f32,
    lod_cells_per_level: f32,
    _lod_pad0: f32,
    _lod_pad1: f32,
}

@group(constants::MATERIAL_BIND_GROUP) @binding(100) var<uniform> ocean_ext: OceanExtensionUniform;
@group(constants::MATERIAL_BIND_GROUP) @binding(101) var displacement_0: texture_2d<f32>;
@group(constants::MATERIAL_BIND_GROUP) @binding(102) var displacement_1: texture_2d<f32>;
@group(constants::MATERIAL_BIND_GROUP) @binding(103) var displacement_2: texture_2d<f32>;
@group(constants::MATERIAL_BIND_GROUP) @binding(104) var displacement_3: texture_2d<f32>;
@group(constants::MATERIAL_BIND_GROUP) @binding(105) var foam_0: texture_2d<f32>;
@group(constants::MATERIAL_BIND_GROUP) @binding(106) var foam_1: texture_2d<f32>;
@group(constants::MATERIAL_BIND_GROUP) @binding(107) var foam_2: texture_2d<f32>;
@group(constants::MATERIAL_BIND_GROUP) @binding(108) var foam_3: texture_2d<f32>;

fn displacement_cascade(i: u32, xz: vec2<f32>) -> vec4<f32> {
    let uv = xz / ocean_ext.cascade_tile_sizes[i];
    if i == 0u {
        return bilinear_wrap(displacement_0, uv);
    } else if i == 1u {
        return bilinear_wrap(displacement_1, uv);
    } else if i == 2u {
        return bilinear_wrap(displacement_2, uv);
    }
    return bilinear_wrap(displacement_3, uv);
}

fn foam_cascade(i: u32, xz: vec2<f32>) -> f32 {
    let uv = xz / ocean_ext.cascade_tile_sizes[i];
    if i == 0u {
        return bilinear_wrap(foam_0, uv).r;
    } else if i == 1u {
        return bilinear_wrap(foam_1, uv).r;
    } else if i == 2u {
        return bilinear_wrap(foam_2, uv).r;
    }
    return bilinear_wrap(foam_3, uv).r;
}

/// Weight of cascade `i` for vertex displacement at a local vertex spacing of `spacing` meters.
/// Waves much shorter than the spacing cannot be resolved by the vertices, and sampling them
/// would make coarse rings shimmer as the mesh moves. Pixel normals still show them.
fn cascade_vertex_weight(i: u32, spacing: f32) -> f32 {
    let tile = ocean_ext.cascade_tile_sizes[i];
    return 1.0 - smoothstep(tile / 16.0, tile / 4.0, spacing);
}

/// Cascade sum for vertex displacement, fading cascades too fine for `spacing`.
fn ocean_sample_lod(xz: vec2<f32>, spacing: f32) -> vec4<f32> {
    var sum = vec4<f32>(0.0);
    let count = min(ocean_ext.cascade_count, 4u);
    for (var i = 0u; i < count; i++) {
        sum += displacement_cascade(i, xz) * cascade_vertex_weight(i, spacing);
    }
    let amp = ocean_ext.amplitude;
    return vec4<f32>(sum.rgb * amp, sum.a * amp * ocean_ext.choppiness);
}

/// Estimated clipmap vertex spacing at `world_xz`. Zero base spacing turns the fade off.
fn lod_spacing(world_xz: vec2<f32>, mesh_center_xz: vec2<f32>) -> f32 {
    if ocean_ext.lod_base_spacing <= 0.0 {
        return 0.0;
    }
    let d = max(abs(world_xz.x - mesh_center_xz.x), abs(world_xz.y - mesh_center_xz.y));
    return max(ocean_ext.lod_base_spacing, d / max(ocean_ext.lod_cells_per_level, 1.0));
}

/// Sum of all cascades at world `xz`, scaled by amplitude. R/G = height slopes along X and Z,
/// B = elevation, A = wind-aligned horizontal chop in meters (choppiness applied).
fn ocean_sample(xz: vec2<f32>) -> vec4<f32> {
    var sum = vec4<f32>(0.0);
    let count = min(ocean_ext.cascade_count, 4u);
    for (var i = 0u; i < count; i++) {
        sum += displacement_cascade(i, xz);
    }
    let amp = ocean_ext.amplitude;
    return vec4<f32>(sum.rgb * amp, sum.a * amp * ocean_ext.choppiness);
}

fn wind_dir() -> vec2<f32> {
    return vec2<f32>(cos(ocean_ext.wind_direction), sin(ocean_ext.wind_direction));
}

/// Undisplaced position whose chop lands on `xz`, found with one fixed-point step.
fn ocean_undisplace(xz: vec2<f32>) -> vec2<f32> {
    let chop = ocean_sample(xz).a;
    return xz - chop * wind_dir();
}

fn ocean_normal_from_slopes(s: vec4<f32>) -> vec3<f32> {
    return normalize(vec3<f32>(-s.r, 1.0, -s.g));
}

struct DisplacedSurface {
    clip_position: vec4<f32>,
    world_position: vec4<f32>,
    world_normal: vec3<f32>,
}

/// Displaces the mesh in world space so any mesh layout (plane, clipmap) samples the same ocean.
fn ocean_displace_surface(instance_index: u32, position_local: vec3<f32>) -> DisplacedSurface {
    let world_from_local = get_world_from_local(instance_index);
    var world = mesh_position_local_to_world(world_from_local, vec4<f32>(position_local, 1.0));
    let spacing = lod_spacing(world.xz, world_from_local[3].xz);
    let s = ocean_sample_lod(world.xz, spacing);
    let w = wind_dir();
    world = vec4<f32>(world.x + s.a * w.x, world.y + s.b, world.z + s.a * w.y, 1.0);

    var out: DisplacedSurface;
    out.clip_position = position_world_to_clip(world.xyz);
    out.world_position = world;
    out.world_normal = ocean_normal_from_slopes(s);
    return out;
}

/// Per-pixel normal from the summed cascade slopes at the undisplaced position under this pixel.
fn ocean_pixel_normal(world_position: vec3<f32>) -> vec3<f32> {
    return ocean_normal_from_slopes(ocean_sample(ocean_undisplace(world_position.xz)));
}

fn ocean_foam_coverage(world_position: vec3<f32>) -> f32 {
    let xz = ocean_undisplace(world_position.xz);
    var foam = 0.0;
    let count = min(ocean_ext.cascade_count, 4u);
    for (var i = 0u; i < count; i++) {
        foam = max(foam, foam_cascade(i, xz));
    }
    return foam;
}

fn ocean_mix_foam_into_base_color(world_position: vec3<f32>, base_color: vec4<f32>) -> vec4<f32> {
    let foam_mix = saturate(ocean_foam_coverage(world_position) * ocean_ext.foam_intensity);
    return vec4<f32>(mix(base_color.rgb, vec3<f32>(1.0), foam_mix), base_color.a);
}

// `PREPASS_PIPELINE` is only set for actual prepass draw pipelines. The main mesh pipeline can still
// define `DEFERRED_PREPASS` (for example when sampling prepass normals) without
// `NORMAL_PREPASS_OR_DEFERRED_PREPASS`, in which case `prepass_io::VertexOutput` has no `world_normal`.
// Match Bevy's `prepass.wgsl`: use `prepass_io` only when `PREPASS_PIPELINE` is set, otherwise `forward_io`.

@vertex
fn vertex(vertex: Vertex) -> VertexOutput {
    let surf = ocean_displace_surface(vertex.instance_index, vertex.position);

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
    var pbr_input = pbr_input_from_standard_material(in, is_front);
    pbr_input.material.base_color = alpha_discard(
        pbr_input.material.flags,
        pbr_input.material.alpha_cutoff,
        pbr_input.material.base_color
    );
    let double_sided =
        (pbr_input.material.flags & STANDARD_MATERIAL_FLAGS_DOUBLE_SIDED_BIT) != 0u;
    let n_world = ocean_pixel_normal(in.world_position.xyz);
    pbr_input.world_normal = prepare_world_normal(n_world, double_sided, is_front);
    pbr_input.N = normalize(pbr_input.world_normal);
    pbr_input.material.base_color = ocean_mix_foam_into_base_color(in.world_position.xyz, pbr_input.material.base_color);
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
    let n_world = ocean_pixel_normal(in.world_position.xyz);
    pbr_input.world_normal = prepare_world_normal(n_world, double_sided, is_front);
    pbr_input.N = normalize(pbr_input.world_normal);
    pbr_input.material.base_color = ocean_mix_foam_into_base_color(in.world_position.xyz, pbr_input.material.base_color);
    return deferred_output(in, pbr_input);
}
}
