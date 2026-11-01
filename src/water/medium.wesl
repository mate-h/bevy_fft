// The water body as a participating medium for pixels whose view ray starts under water, after
// Monzon et al. (2024, §4.2). Runs after the main pass, before tonemapping, on exposed linear HDR.
//
// The waterline mask says where the near plane lies under water. The water surface draws its own
// underside together with the medium below it, so pixels that land on the surface are left alone.
// Everything else seen from under water fades with the extinction along the ray, gains the
// multiple-scattered light of Eq. 9 and the single-scattered sunlight of Eq. 5, and is lit as if
// by the downwelling irradiance at its depth, which is the `exp(-K_d y_S)` factor of Eq. 10.
import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput;
import bevy_pbr::render::mesh_view_bindings::{view, lights};
import bevy_pbr::render::view_transformations::position_ndc_to_world;
import bevy_fft::water::lod::{WaterLodSlices, displaced_height, lod_coverage, sample_lod_filtered};
import bevy_fft::water::underside::{
    MEDIUM_MAX_PATH,
    downwelling_irradiance,
    fresnel_dielectric,
    medium,
    medium_multiple_scattering,
    single_scattering,
};

@group(constants::MATERIAL_BIND_GROUP) @binding(0) var source: texture_2d<f32>;
@group(constants::MATERIAL_BIND_GROUP) @binding(1) var scene_depth: texture_depth_2d;
/// Height of each pixel's near-plane point above the water, from the waterline pass.
@group(constants::MATERIAL_BIND_GROUP) @binding(2) var waterline_mask: texture_2d<f32>;
@group(constants::MATERIAL_BIND_GROUP) @binding(3) var<uniform> lod: WaterLodSlices;
@group(constants::MATERIAL_BIND_GROUP) @binding(4) var lod_displacement: texture_2d_array<f32>;
@group(constants::MATERIAL_BIND_GROUP) @binding(5) var lod_foam: texture_2d_array<f32>;
/// Linear, clamped to edge.
@group(constants::MATERIAL_BIND_GROUP) @binding(6) var lod_sampler: sampler;
/// Height of the water mesh plane in `x`.
@group(constants::MATERIAL_BIND_GROUP) @binding(7) var<uniform> water_plane: vec4<f32>;
/// Caustic intensity on the receiver plane, laid out like one LOD slice.
@group(constants::MATERIAL_BIND_GROUP) @binding(8) var caustic_map: texture_2d<f32>;
/// One in `x` when the map was traced this frame, the receiver depth in `y`, the slice in `z`.
@group(constants::MATERIAL_BIND_GROUP) @binding(9) var<uniform> caustic_params: vec4<f32>;

/// Meters a scene point may sit below the displaced surface and still count as the surface, which
/// covers the difference between the mesh and the per-pixel height.
const SURFACE_TOLERANCE: f32 = 0.05;
/// Depth reported past the coarsest slice, deep enough that no bed light returns.
const OPEN_WATER_DEPTH: f32 = 1000.0;

fn surface_height(xz: vec2<f32>) -> f32 {
    return water_plane.x + displaced_height(lod_displacement, lod_foam, lod_sampler, lod, xz);
}

/// Depth of the water column under `xz`, from the green lane of the LOD foam data.
fn water_column(xz: vec2<f32>) -> f32 {
    let c = lod_coverage(lod, xz);
    return mix(OPEN_WATER_DEPTH, sample_lod_filtered(lod_foam, lod_sampler, lod, xz, 0.0).g, c);
}

/// Bilinear caustic intensity at world `xz` on the receiver, one outside the map.
fn caustic_at_receiver(xz: vec2<f32>) -> f32 {
    let s = lod.slices[u32(caustic_params.z)];
    let n = vec2<f32>(textureDimensions(caustic_map));
    let x = ((xz - s.xy) / s.w + 0.5) * n - 0.5;
    let i0 = vec2<i32>(floor(x));
    if any(i0 < vec2<i32>(0)) || any(i0 + 1 >= vec2<i32>(n)) {
        return 1.0;
    }
    let f = x - floor(x);
    let a = textureLoad(caustic_map, i0, 0).r;
    let b = textureLoad(caustic_map, i0 + vec2<i32>(1, 0), 0).r;
    let c = textureLoad(caustic_map, i0 + vec2<i32>(0, 1), 0).r;
    let d = textureLoad(caustic_map, i0 + vec2<i32>(1, 1), 0).r;
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

/// Scale on the light reaching scene point `p` from the caustic map. The first directional light
/// refracts at a level surface, and the map is read where that beam through `p` crosses the
/// receiver plane. The map holds the pattern at the receiver depth only, so points far above or
/// below it see that pattern rather than their own focus. Only the sun's share of the downwelling
/// irradiance `e0` follows the map.
fn caustic_lighting(p: vec3<f32>, e0: vec3<f32>) -> vec3<f32> {
    if caustic_params.x == 0.0 || lights.n_directional_lights == 0u {
        return vec3<f32>(1.0);
    }
    let light = &lights.directional_lights[0];
    let L = (*light).direction_to_light.xyz;
    if L.y <= 0.0 {
        return vec3<f32>(1.0);
    }
    let eta = 1.0 / medium.ior;
    let beam = refract(-L, vec3<f32>(0.0, 1.0, 0.0), eta);
    let depth = water_plane.x - p.y;
    let back = (depth - caustic_params.y) / max(-beam.y, 1e-3);
    let intensity = caustic_at_receiver(p.xz - beam.xz * back);
    let sun = (*light).color.rgb * L.y * (1.0 - fresnel_dielectric(L.y, eta));
    let share = select(sun / e0, vec3<f32>(0.0), e0 <= vec3<f32>(0.0));
    return 1.0 + (intensity - 1.0) * saturate(share);
}

@fragment
fn water_medium(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let pixel = vec2<i32>(in.position.xy);
    let color = textureLoad(source, pixel, 0);
    let near_height = textureLoad(waterline_mask, pixel, 0).r;
    if near_height >= 0.0 {
        return color;
    }
    let ndc = in.uv * vec2<f32>(2.0, -2.0) + vec2<f32>(-1.0, 1.0);
    // Reversed depth puts the near plane at one.
    let near = position_ndc_to_world(vec3<f32>(ndc, 1.0));
    let direction = normalize(position_ndc_to_world(vec3<f32>(ndc, 0.5)) - near);
    let descent = -direction.y;
    let depth_near = -near_height;
    let floor_depth = max(water_column(near.xz), depth_near);

    var path = MEDIUM_MAX_PATH;
    var lighting = vec3<f32>(1.0);
    let e0 = downwelling_irradiance();
    let depth = textureLoad(scene_depth, pixel, 0);
    if depth > 0.0 {
        let far = position_ndc_to_world(vec3<f32>(ndc, depth));
        let h = surface_height(far.xz);
        if far.y > h - SURFACE_TOLERANCE {
            return color;
        }
        path = distance(near, far);
        lighting = exp(-max(medium.diffuse_attenuation, vec3<f32>(0.0)) * (h - far.y))
            * caustic_lighting(far, e0);
    } else if descent < 0.0 {
        // A ray that rises without meeting the surface mesh stops at the surface.
        path = depth_near / max(-descent, 1e-3);
    } else {
        path = min(path, (floor_depth - depth_near) / max(descent, 1e-3));
    }

    let sigma_t = max(medium.absorption + medium.scattering, vec3<f32>(0.0));
    let transmittance = exp(-sigma_t * path);
    let scattered = medium_multiple_scattering(depth_near, descent, path, floor_depth, e0)
        + single_scattering(near, direction, path, water_plane.x, in.position.xy);
    return vec4<f32>(color.rgb * lighting * transmittance + scattered * view.exposure, color.a);
}
