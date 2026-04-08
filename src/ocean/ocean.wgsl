#import bevy_pbr::mesh_functions::{
    get_world_from_local,
    get_visibility_range_dither_level,
    mesh_normal_local_to_world,
    mesh_position_local_to_clip,
    mesh_position_local_to_world,
}

struct OceanExtensionUniform {
    amplitude: f32,
    choppiness: f32,
    ocean_tile_world_size: f32,
    grid_size: f32,
    wind_direction: f32,
}

@group(#{MATERIAL_BIND_GROUP}) @binding(100) var<uniform> ocean_ext: OceanExtensionUniform;
@group(#{MATERIAL_BIND_GROUP}) @binding(101) var displacement_texture: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(102) var displacement_sampler: sampler;

fn ocean_load_disp(tex: texture_2d<f32>, cx: i32, cz: i32, dims: i32) -> vec4<f32> {
    let ii = clamp(cx, 0, dims - 1);
    let jj = clamp(cz, 0, dims - 1);
    return textureLoad(tex, vec2<i32>(ii, jj), 0);
}

struct DisplacedSurface {
    clip_position: vec4<f32>,
    world_position: vec4<f32>,
    world_normal: vec3<f32>,
}

/// `spatial_output`: R/G/B = height slopes X and Z and elevation; A = wind-along chop scalar (before amplitude and choppiness).
fn ocean_displace_surface(instance_index: u32, position_local: vec3<f32>, mesh_uv: vec2<f32>) -> DisplacedSurface {
    let size = ocean_ext.grid_size;
    // Plane UVs match the undistorted patch; using them avoids chop / interpolation pushing lookups past the tile edge.
    let x = mesh_uv.x * (size - 1.0);
    let z = mesh_uv.y * (size - 1.0);

    let dims = i32(size);
    let ix = clamp(i32(round(x)), 0, dims - 1);
    let iz = clamp(i32(round(z)), 0, dims - 1);
    let disp = ocean_load_disp(displacement_texture, ix, iz, dims);

    let amp = ocean_ext.amplitude;
    let tile = ocean_ext.ocean_tile_world_size;

    let deta_dx = disp.r * amp;
    let deta_dz = disp.g * amp;
    let eta = disp.b * amp;
    // Same linear scale as height; scalar field matches `((k dot w_hat)/|k|)(i*h)` in `ocean_spectrum`.
    let chop = disp.a * amp * ocean_ext.choppiness;
    let wx = cos(ocean_ext.wind_direction);
    let wz = sin(ocean_ext.wind_direction);

    var displaced_position = position_local;
    displaced_position.x += chop * wx;
    displaced_position.z += chop * wz;
    displaced_position.y = eta;

    // Tangents use spectral height slopes only (horizontal chop does not tilt the normal here).
    let dx_cell = tile / max(size - 1.0, 1.0);
    let dz_cell = dx_cell;
    let tu = vec3<f32>(2.0 * dx_cell, deta_dx * 2.0 * dx_cell, 0.0);
    let tv = vec3<f32>(0.0, deta_dz * 2.0 * dz_cell, 2.0 * dz_cell);

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

/// Slopes from bilinear fetch on the FFT grid (mip 0), then the same local normal as the vertex path.
/// `mesh_uv` is the plane attribute UV (0..1); we map to texel-centered coordinates so clamp-to-edge never pins the seam.
fn ocean_highres_world_normal(instance_index: u32, mesh_uv: vec2<f32>) -> vec3<f32> {
    let size = ocean_ext.grid_size;
    let amp = ocean_ext.amplitude;
    let tile = ocean_ext.ocean_tile_world_size;

    let n_tex = max(size, 1.0);
    let uv_tex = mesh_uv * (n_tex - 1.0) / n_tex + vec2<f32>(0.5 / n_tex);
    let disp = textureSampleLevel(displacement_texture, displacement_sampler, uv_tex, 0.0);

    let deta_dx = disp.r * amp;
    let deta_dz = disp.g * amp;

    let dx_cell = tile / max(size - 1.0, 1.0);
    let dz_cell = dx_cell;
    let tu = vec3<f32>(2.0 * dx_cell, deta_dx * 2.0 * dx_cell, 0.0);
    let tv = vec3<f32>(0.0, deta_dz * 2.0 * dz_cell, 2.0 * dz_cell);
    let n_raw = cross(tv, tu);
    let len2 = dot(n_raw, n_raw);
    let local_n = select(normalize(n_raw), vec3(0.0, 1.0, 0.0), len2 < 1e-12);
    return mesh_normal_local_to_world(local_n, instance_index);
}

// `PREPASS_PIPELINE` is only set for actual prepass draw pipelines. The main mesh pipeline can still
// define `DEFERRED_PREPASS` (for example when sampling prepass normals) without
// `NORMAL_PREPASS_OR_DEFERRED_PREPASS`, in which case `prepass_io::VertexOutput` has no `world_normal`.
// Match Bevy's `prepass.wgsl`: use `prepass_io` only when `PREPASS_PIPELINE` is set, otherwise `forward_io`.
#ifdef PREPASS_PIPELINE
#import bevy_pbr::prepass_io::{Vertex, VertexOutput};
#else
#import bevy_pbr::forward_io::{Vertex, VertexOutput};
#endif

@vertex
fn vertex(vertex: Vertex) -> VertexOutput {
    let surf = ocean_displace_surface(vertex.instance_index, vertex.position, vertex.uv);

#ifdef PREPASS_PIPELINE
#ifdef NORMAL_PREPASS_OR_DEFERRED_PREPASS
    var out: VertexOutput;
    out.position = surf.clip_position;
    out.world_position = surf.world_position;
#ifdef VERTEX_UVS_A
    out.uv = vertex.uv;
#endif
#ifdef VERTEX_UVS_B
    out.uv_b = vertex.uv_b;
#endif
    out.world_normal = surf.world_normal;
#ifdef VERTEX_OUTPUT_INSTANCE_INDEX
    out.instance_index = vertex.instance_index;
#endif
#ifdef MOTION_VECTOR_PREPASS
    out.previous_world_position = surf.world_position;
#endif
#ifdef VERTEX_COLORS
    out.color = vertex.color;
#endif
#ifdef VISIBILITY_RANGE_DITHER
    out.visibility_range_dither = get_visibility_range_dither_level(
        vertex.instance_index,
        get_world_from_local(vertex.instance_index)[3],
    );
#endif
    return out;
#else
    var out: VertexOutput;
    out.position = surf.clip_position;
    out.world_position = surf.world_position;
#ifdef VERTEX_UVS_A
    out.uv = vertex.uv;
#endif
#ifdef VERTEX_UVS_B
    out.uv_b = vertex.uv_b;
#endif
#ifdef VERTEX_OUTPUT_INSTANCE_INDEX
    out.instance_index = vertex.instance_index;
#endif
#ifdef MOTION_VECTOR_PREPASS
    out.previous_world_position = surf.world_position;
#endif
#ifdef VERTEX_COLORS
    out.color = vertex.color;
#endif
#ifdef UNCLIPPED_DEPTH_ORTHO_EMULATION
    out.unclipped_depth = surf.clip_position.z;
    out.position.z = min(out.position.z, 1.0);
#endif
#ifdef VISIBILITY_RANGE_DITHER
    out.visibility_range_dither = get_visibility_range_dither_level(
        vertex.instance_index,
        get_world_from_local(vertex.instance_index)[3],
    );
#endif
    return out;
#endif
#else
    var out: VertexOutput;
    out.position = surf.clip_position;
    out.world_position = surf.world_position;
    out.world_normal = surf.world_normal;
#ifdef VERTEX_UVS_A
    out.uv = vertex.uv;
#endif
#ifdef VERTEX_UVS_B
    out.uv_b = vertex.uv_b;
#endif
#ifdef VERTEX_OUTPUT_INSTANCE_INDEX
    out.instance_index = vertex.instance_index;
#endif
#ifdef VERTEX_COLORS
    out.color = vertex.color;
#endif
#ifdef VISIBILITY_RANGE_DITHER
    out.visibility_range_dither = get_visibility_range_dither_level(
        vertex.instance_index,
        get_world_from_local(vertex.instance_index)[3],
    );
#endif
    return out;
#endif
}

#ifndef PREPASS_PIPELINE
#import bevy_pbr::{
    pbr_fragment::pbr_input_from_standard_material,
    pbr_functions::alpha_discard,
    pbr_functions::prepare_world_normal,
    pbr_types,
    pbr_types::STANDARD_MATERIAL_FLAGS_DOUBLE_SIDED_BIT,
    pbr_types::STANDARD_MATERIAL_FLAGS_UNLIT_BIT,
    forward_io::{VertexOutput as FragVertexOutput, FragmentOutput},
    pbr_functions::{apply_pbr_lighting, main_pass_post_lighting_processing},
}
#endif

#ifdef PREPASS_PIPELINE
#ifdef PREPASS_FRAGMENT
#import bevy_pbr::prepass_io::{VertexOutput as FragVertexOutput, FragmentOutput};
#ifdef NORMAL_PREPASS_OR_DEFERRED_PREPASS
#import bevy_pbr::{
    pbr_fragment::pbr_input_from_standard_material,
    pbr_functions::alpha_discard,
    pbr_functions::prepare_world_normal,
    pbr_deferred_functions::deferred_output,
    pbr_types::STANDARD_MATERIAL_FLAGS_DOUBLE_SIDED_BIT,
}
#endif
#endif
#endif

#ifndef PREPASS_PIPELINE
@fragment
fn fragment(in: FragVertexOutput, @builtin(front_facing) is_front: bool) -> FragmentOutput {
    var pbr_input = pbr_input_from_standard_material(in, is_front);
    pbr_input.material.base_color = alpha_discard(pbr_input.material, pbr_input.material.base_color);
#ifdef VERTEX_OUTPUT_INSTANCE_INDEX
#ifdef VERTEX_UVS_A
    let double_sided =
        (pbr_input.material.flags & STANDARD_MATERIAL_FLAGS_DOUBLE_SIDED_BIT) != 0u;
    let n_world = ocean_highres_world_normal(in.instance_index, in.uv);
    pbr_input.world_normal = prepare_world_normal(n_world, double_sided, is_front);
    pbr_input.N = normalize(pbr_input.world_normal);
#endif
#endif
    var out: FragmentOutput;
    if (pbr_input.material.flags & STANDARD_MATERIAL_FLAGS_UNLIT_BIT) == 0u {
        out.color = apply_pbr_lighting(pbr_input);
    } else {
        out.color = pbr_input.material.base_color;
    }
    out.color = main_pass_post_lighting_processing(pbr_input, out.color);
    return out;
}
#endif

#ifdef PREPASS_PIPELINE
#ifdef PREPASS_FRAGMENT
@fragment
fn fragment(in: FragVertexOutput, @builtin(front_facing) is_front: bool) -> FragmentOutput {
#ifndef NORMAL_PREPASS_OR_DEFERRED_PREPASS
    var out: FragmentOutput;
#ifdef UNCLIPPED_DEPTH_ORTHO_EMULATION
    out.frag_depth = in.position.z;
#endif
    return out;
#else
    var pbr_input = pbr_input_from_standard_material(in, is_front);
    pbr_input.material.base_color = alpha_discard(pbr_input.material, pbr_input.material.base_color);
#ifdef VERTEX_OUTPUT_INSTANCE_INDEX
#ifdef VERTEX_UVS_A
    let double_sided =
        (pbr_input.material.flags & STANDARD_MATERIAL_FLAGS_DOUBLE_SIDED_BIT) != 0u;
    let n_world = ocean_highres_world_normal(in.instance_index, in.uv);
    pbr_input.world_normal = prepare_world_normal(n_world, double_sided, is_front);
    pbr_input.N = normalize(pbr_input.world_normal);
#endif
#endif
    return deferred_output(in, pbr_input);
#endif
}
#endif
#endif
