import bevy_pbr::render::mesh_functions::{
    get_world_from_local,
    get_visibility_range_dither_level,
    mesh_normal_local_to_world,
    mesh_position_local_to_clip,
    mesh_position_local_to_world,
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

struct DispersiveMaterialUniform {
    tile_world_size: f32,
    grid_size: f32,
    _pad0: f32,
    _pad1: f32,
}

/// Matches flat-bed `init_basin` rest depth. Mesh Y is free-surface η = h − rest (meters).
const D_REST_DEPTH: f32 = 4.0;

@group(constants::MATERIAL_BIND_GROUP) @binding(100) var<uniform> d_ext: DispersiveMaterialUniform;
@group(constants::MATERIAL_BIND_GROUP) @binding(101) var state_texture: texture_2d<f32>;
@group(constants::MATERIAL_BIND_GROUP) @binding(102) var state_sampler: sampler;

/// Texel-centered UV like `ocean.wgsl` so linear sampling lines up with the sim grid.
fn d_uv_texel(mesh_uv: vec2<f32>) -> vec2<f32> {
    let n_tex = max(d_ext.grid_size, 1.0);
    return mesh_uv * (n_tex - 1.0) / n_tex + vec2<f32>(0.5 / n_tex);
}

fn d_surface_height_at_uv(mesh_uv: vec2<f32>) -> f32 {
    let uv_tex = d_uv_texel(clamp(mesh_uv, vec2<f32>(0.0), vec2<f32>(1.0)));
    let st = textureSampleLevel(state_texture, state_sampler, uv_tex, 0.0);
    return st.r - D_REST_DEPTH;
}

/// World-space dh/dx and dh/dz from filtered height; matches smooth fragment normals.
fn d_world_height_gradients(mesh_uv: vec2<f32>) -> vec2<f32> {
    let n_tex = max(d_ext.grid_size, 1.0);
    let du = 1.0 / n_tex;
    let dv = du;
    let tile = max(d_ext.tile_world_size, 1e-6);
    let h_l = d_surface_height_at_uv(mesh_uv - vec2<f32>(du, 0.0));
    let h_r = d_surface_height_at_uv(mesh_uv + vec2<f32>(du, 0.0));
    let h_d = d_surface_height_at_uv(mesh_uv - vec2<f32>(0.0, dv));
    let h_u = d_surface_height_at_uv(mesh_uv + vec2<f32>(0.0, dv));
    let dhd_u = (h_r - h_l) / (2.0 * du);
    let dhd_v = (h_u - h_d) / (2.0 * dv);
    return vec2<f32>(dhd_u / tile, dhd_v / tile);
}

struct DisplacedSurface {
    clip_position: vec4<f32>,
    world_position: vec4<f32>,
    world_normal: vec3<f32>,
}

fn d_displace_surface(instance_index: u32, position_local: vec3<f32>, mesh_uv: vec2<f32>) -> DisplacedSurface {
    let surface = d_surface_height_at_uv(mesh_uv);
    let dhd = d_world_height_gradients(mesh_uv);
    let deta_dx = dhd.x;
    let deta_dz = dhd.y;

    var displaced_position = position_local;
    displaced_position.y = surface;

    let tu = vec3<f32>(1.0, deta_dx, 0.0);
    let tv = vec3<f32>(0.0, deta_dz, 1.0);
    let n_raw = cross(tv, tu);
    let len2 = dot(n_raw, n_raw);
    let local_n = select(normalize(n_raw), vec3(0.0, 1.0, 0.0), len2 < 1e-12);

    let world_from_local = get_world_from_local(instance_index);
    var out: DisplacedSurface;
    out.clip_position = mesh_position_local_to_clip(world_from_local, vec4<f32>(displaced_position, 1.0));
    out.world_position = mesh_position_local_to_world(world_from_local, vec4<f32>(displaced_position, 1.0));
    out.world_normal = mesh_normal_local_to_world(local_n, instance_index);
    return out;
}

fn d_highres_world_normal(instance_index: u32, mesh_uv: vec2<f32>) -> vec3<f32> {
    let dhd = d_world_height_gradients(mesh_uv);
    let tu = vec3<f32>(1.0, dhd.x, 0.0);
    let tv = vec3<f32>(0.0, dhd.y, 1.0);
    let n_raw = cross(tv, tu);
    let len2 = dot(n_raw, n_raw);
    let local_n = select(normalize(n_raw), vec3(0.0, 1.0, 0.0), len2 < 1e-12);
    return mesh_normal_local_to_world(local_n, instance_index);
}

/// Keeps [`StandardMaterial`] base color (ocean tint, roughness) and only softens the very shallow band toward sand.
fn d_tint_albedo(_mesh_uv: vec2<f32>, base_color: vec4<f32>) -> vec4<f32> {
    return base_color;
}

@vertex
fn vertex(vertex: Vertex) -> VertexOutput {
    let surf = d_displace_surface(vertex.instance_index, vertex.position, vertex.uv);

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
@if(VERTEX_OUTPUT_INSTANCE_INDEX && VERTEX_UVS_A) {
    let double_sided =
        (pbr_input.material.flags & STANDARD_MATERIAL_FLAGS_DOUBLE_SIDED_BIT) != 0u;
    let n_world = d_highres_world_normal(in.instance_index, in.uv);
    pbr_input.world_normal = prepare_world_normal(n_world, double_sided, is_front);
    pbr_input.N = normalize(pbr_input.world_normal);
    pbr_input.material.base_color = d_tint_albedo(in.uv, pbr_input.material.base_color);
}
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
@if(VERTEX_OUTPUT_INSTANCE_INDEX && VERTEX_UVS_A) {
    let double_sided =
        (pbr_input.material.flags & STANDARD_MATERIAL_FLAGS_DOUBLE_SIDED_BIT) != 0u;
    let n_world = d_highres_world_normal(in.instance_index, in.uv);
    pbr_input.world_normal = prepare_world_normal(n_world, double_sided, is_front);
    pbr_input.N = normalize(pbr_input.world_normal);
    pbr_input.material.base_color = d_tint_albedo(in.uv, pbr_input.material.base_color);
}
    return deferred_output(in, pbr_input);
}
}
