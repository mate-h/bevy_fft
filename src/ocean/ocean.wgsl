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
    foam_intensity: f32,
    foam_cutoff: f32,
    foam_falloff: f32,
    foam_color: vec4<f32>,
    crest_scatter_intensity: f32,
    crest_scatter_view_power: f32,
    crest_scatter_rim_power: f32,
    crest_scatter_slope_scale: f32,
    crest_scatter_tint: vec4<f32>,
    crest_light_dir_to_light_ws: vec4<f32>,
    crest_light_radiance: vec4<f32>,
    foam_trail_decay: f32,
}

@group(#{MATERIAL_BIND_GROUP}) @binding(100) var<uniform> ocean_ext: OceanExtensionUniform;
@group(#{MATERIAL_BIND_GROUP}) @binding(101) var displacement_texture: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(102) var displacement_sampler: sampler;
@group(#{MATERIAL_BIND_GROUP}) @binding(103) var foam_mask_texture: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(104) var foam_mask_sampler: sampler;

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

/// `mesh_uv` is the plane attribute UV (0..1); texel-centered coordinates so clamp-to-edge never pins the seam.
/// Returns `(d_eta/dx, d_eta/dz, |grad eta|)` in world-scale height units (after amplitude).
fn ocean_displacement_slopes_steepness(mesh_uv: vec2<f32>) -> vec3<f32> {
    let size = ocean_ext.grid_size;
    let amp = ocean_ext.amplitude;

    let n_tex = max(size, 1.0);
    let uv_tex = mesh_uv * (n_tex - 1.0) / n_tex + vec2<f32>(0.5 / n_tex);
    let disp = textureSampleLevel(displacement_texture, displacement_sampler, uv_tex, 0.0);

    let deta_dx = disp.r * amp;
    let deta_dz = disp.g * amp;
    let steep = length(vec2<f32>(deta_dx, deta_dz));
    return vec3<f32>(deta_dx, deta_dz, steep);
}

/// Bilinear slopes on the FFT grid (mip 0), same local normal construction as the vertex path.
fn ocean_world_normal_from_slopes(instance_index: u32, deta_dx: f32, deta_dz: f32) -> vec3<f32> {
    let size = ocean_ext.grid_size;
    let tile = ocean_ext.ocean_tile_world_size;

    let dx_cell = tile / max(size - 1.0, 1.0);
    let dz_cell = dx_cell;
    let tu = vec3<f32>(2.0 * dx_cell, deta_dx * 2.0 * dx_cell, 0.0);
    let tv = vec3<f32>(0.0, deta_dz * 2.0 * dz_cell, 2.0 * dz_cell);
    let n_raw = cross(tv, tu);
    let len2 = dot(n_raw, n_raw);
    let local_n = select(normalize(n_raw), vec3(0.0, 1.0, 0.0), len2 < 1e-12);
    return mesh_normal_local_to_world(local_n, instance_index);
}

fn ocean_foam_coverage(mesh_uv: vec2<f32>) -> f32 {
    let size = ocean_ext.grid_size;
    let n_tex = max(size, 1.0);
    let uv_tex = mesh_uv * (n_tex - 1.0) / n_tex + vec2<f32>(0.5 / n_tex);
    return textureSampleLevel(foam_mask_texture, foam_mask_sampler, uv_tex, 0.0).r;
}

fn ocean_mix_foam_into_base_color(mesh_uv: vec2<f32>, base_color: vec4<f32>) -> vec4<f32> {
    let foam_cov = ocean_foam_coverage(mesh_uv);
    let foam_mix = saturate(foam_cov * ocean_ext.foam_intensity);
    let foam_rgb = ocean_ext.foam_color.xyz;
    return vec4<f32>(mix(base_color.rgb, foam_rgb, foam_mix), base_color.a);
}

/// Additive highlight on sharp crests from the key directional light (`crest_light_*` in the uniform), rim, and slope; fades where foam is strong.
/// `N_shade` is the double-sided corrected normal used with lighting. `N_geom` is the FFT slope normal before `prepare_world_normal`, so rim uses true silhouette (see fragment).
fn ocean_crest_scatter_emissive(
    N_shade: vec3<f32>,
    N_geom: vec3<f32>,
    V: vec3<f32>,
    slope_steepness: f32,
    mesh_uv: vec2<f32>,
) -> vec3<f32> {
    let intensity = ocean_ext.crest_scatter_intensity;
    let light_rgb = ocean_ext.crest_light_radiance.xyz;
    if (intensity <= 0.0 || dot(light_rgb, light_rgb) < 1e-12) {
        return vec3(0.0);
    }

    let L = normalize(ocean_ext.crest_light_dir_to_light_ws.xyz);
    if (dot(L, L) < 1e-12) {
        return vec3(0.0);
    }

    let Nn = normalize(N_shade);
    let Ng = normalize(N_geom);
    let Vn = normalize(V);
    let slope_term = saturate(slope_steepness * ocean_ext.crest_scatter_slope_scale);

    let foam_cov = ocean_foam_coverage(mesh_uv);
    let foam_mix = saturate(foam_cov * ocean_ext.foam_intensity);
    let foam_suppress = 1.0 - foam_mix;

    // Not front-on to the light (side faces and silhouettes), not only when N·L < 0.
    let ndotl = dot(Nn, L);
    let scatter_facing = pow(saturate(1.0 - saturate(ndotl)), 0.65);

    // `V` points fragment→camera; `L` fragment→sun. Back-lit “see-through” crests make V and L roughly
    // opposed, so use -V·L instead of saturating V·L (which goes to zero there).
    let through_view = pow(saturate(-dot(Vn, L)), ocean_ext.crest_scatter_view_power);
    let same_side = pow(saturate(dot(Vn, L)), ocean_ext.crest_scatter_view_power);
    let toward_light = max(through_view, 0.12 * same_side);

    // Rim from geometric N·V only. `N_shade` is flipped on back faces for double-sided PBR, which
    // forces N·V > 0 everywhere and makes (1-N·V)^p peak on interior trough walls instead of wave silhouettes.
    let nv_geom = dot(Ng, Vn);
    let rim_grazing = pow(saturate(1.0 - nv_geom), ocean_ext.crest_scatter_rim_power);
    let rim = select(rim_grazing, 0.0, nv_geom <= 0.0);

    let tint = ocean_ext.crest_scatter_tint.xyz;
    return light_rgb * tint * scatter_facing * toward_light * rim * slope_term * intensity * foam_suppress;
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
    let disp_slopes = ocean_displacement_slopes_steepness(in.uv);
    let n_world = ocean_world_normal_from_slopes(in.instance_index, disp_slopes.x, disp_slopes.y);
    let n_geom = normalize(n_world);
    pbr_input.world_normal = prepare_world_normal(n_world, double_sided, is_front);
    pbr_input.N = normalize(pbr_input.world_normal);
    pbr_input.material.base_color = ocean_mix_foam_into_base_color(in.uv, pbr_input.material.base_color);
    let crest_e = ocean_crest_scatter_emissive(pbr_input.N, n_geom, pbr_input.V, disp_slopes.z, in.uv);
    pbr_input.material.emissive = vec4(pbr_input.material.emissive.rgb + crest_e, pbr_input.material.emissive.a);
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
    let disp_slopes = ocean_displacement_slopes_steepness(in.uv);
    let n_world = ocean_world_normal_from_slopes(in.instance_index, disp_slopes.x, disp_slopes.y);
    let n_geom = normalize(n_world);
    pbr_input.world_normal = prepare_world_normal(n_world, double_sided, is_front);
    pbr_input.N = normalize(pbr_input.world_normal);
    pbr_input.material.base_color = ocean_mix_foam_into_base_color(in.uv, pbr_input.material.base_color);
    let crest_e = ocean_crest_scatter_emissive(pbr_input.N, n_geom, pbr_input.V, disp_slopes.z, in.uv);
    pbr_input.material.emissive = vec4(pbr_input.material.emissive.rgb + crest_e, pbr_input.material.emissive.a);
#endif
#endif
    return deferred_output(in, pbr_input);
#endif
}
#endif
#endif
