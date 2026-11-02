// The water body as a participating medium for pixels whose view ray starts under water, after
// Monzon et al. (2024, §4.2). Runs after the main pass, before tonemapping, on exposed linear HDR.
//
// The waterline mask says where the near plane lies under water. The water surface draws its own
// underside together with the medium below it, so pixels that land on the surface are left alone.
// Everything else seen from under water fades with the extinction along the ray, gains the
// multiple-scattered light of Eq. 9 and the single-scattered sunlight of Eq. 5, and is lit as if
// by the downwelling irradiance at its depth, which is the `exp(-K_d y_S)` factor of Eq. 10. The
// sun's share fades over its refracted beam's slant path, the same way beds seen from above do.
import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput;
import bevy_pbr::render::mesh_view_bindings::{view, lights};
import bevy_pbr::render::view_transformations::position_ndc_to_world;
import bevy_fft::water::lod::{WaterLodSlices, displaced_height, lod_coverage, sample_lod_filtered};
import bevy_fft::water::underside::{
    MEDIUM_MAX_PATH,
    downwelling_irradiance,
    medium,
    medium_multiple_scattering,
    single_scattering,
    underwater_lighting,
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

/// Meters a scene point may sit below the displaced surface and still count as the surface, which
/// covers the difference between the mesh and the per-pixel height.
const SURFACE_TOLERANCE: f32 = 0.05;
/// Depth reported past the coarsest slice, deep enough that no bed light returns.
const OPEN_WATER_DEPTH: f32 = 1000.0;

/// Darkening per meniscus sample that lands on the other side of the waterline, after Crest's
/// `ComputeMeniscusWeight`.
const MENISCUS_DARKENING: f32 = 0.9;
/// Pixels along the horizon normal that the meniscus looks across.
const MENISCUS_SAMPLES: i32 = 3;

fn near_height_at(pixel: vec2<i32>) -> f32 {
    let size = vec2<i32>(textureDimensions(waterline_mask));
    return textureLoad(waterline_mask, clamp(pixel, vec2<i32>(0), size - 1), 0).r;
}

/// Dark line where the near plane cuts the water, the zero crossing of the waterline mask. It
/// stands in for the water film a real lens gathers at that line. No paper models it, so it is an
/// appearance choice after Crest's underwater meniscus: a pixel darkens by `MENISCUS_DARKENING`
/// for each of the next `MENISCUS_SAMPLES` pixels along the horizon normal that lies on the other
/// side, so the line is a few pixels wide on both sides. The horizon normal is the screen gradient
/// of the mask. Crest also fades the line out by 15 m for water volumes whose edge can lie at a
/// distance. This line always lies on the near plane, so it never fades.
fn meniscus_weight(pixel: vec2<i32>, height: f32) -> f32 {
    let gradient = vec2<f32>(
        near_height_at(pixel + vec2<i32>(1, 0)) - near_height_at(pixel - vec2<i32>(1, 0)),
        near_height_at(pixel + vec2<i32>(0, 1)) - near_height_at(pixel - vec2<i32>(0, 1)),
    );
    let length_g = length(gradient);
    if length_g <= 0.0 {
        return 1.0;
    }
    let below = height < 0.0;
    // Toward the other side of the waterline.
    let across = select(-gradient, gradient, below) / length_g;
    var weight = 1.0;
    for (var i = 1; i <= MENISCUS_SAMPLES; i++) {
        let sample = pixel + vec2<i32>(round(across * f32(i)));
        if (near_height_at(sample) < 0.0) != below {
            weight *= MENISCUS_DARKENING;
        }
    }
    return weight;
}

fn surface_height(xz: vec2<f32>) -> f32 {
    return water_plane.x + displaced_height(lod_displacement, lod_foam, lod_sampler, lod, xz);
}

/// Depth of the water column under `xz`, from the green lane of the LOD foam data.
fn water_column(xz: vec2<f32>) -> f32 {
    let c = lod_coverage(lod, xz);
    return mix(OPEN_WATER_DEPTH, sample_lod_filtered(lod_foam, lod_sampler, lod, xz, 0.0).g, c);
}

@fragment
fn water_medium(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let pixel = vec2<i32>(in.position.xy);
    let color = textureLoad(source, pixel, 0);
    let near_height = textureLoad(waterline_mask, pixel, 0).r;
    let meniscus = meniscus_weight(pixel, near_height);
    if near_height >= 0.0 {
        return vec4<f32>(color.rgb * meniscus, color.a);
    }
    let ndc = in.uv * vec2<f32>(2.0, -2.0) + vec2<f32>(-1.0, 1.0);
    // Reversed depth puts the near plane at one.
    let near = position_ndc_to_world(vec3<f32>(ndc, 1.0));
    let direction = normalize(position_ndc_to_world(vec3<f32>(ndc, 0.5)) - near);
    let descent = -direction.y;
    // Depths run from the mesh plane, as in the underside, since the light field of Monzon et al.
    // falls off below a level surface. Waves passing overhead leave it in place.
    let depth_near = max(water_plane.x - near.y, 0.0);
    let floor_depth = max(water_column(near.xz), depth_near);

    var path = MEDIUM_MAX_PATH;
    var lighting = vec3<f32>(1.0);
    let e0 = downwelling_irradiance();
    let depth = textureLoad(scene_depth, pixel, 0);
    if depth > 0.0 {
        let far = position_ndc_to_world(vec3<f32>(ndc, depth));
        let h = surface_height(far.xz);
        if far.y > h - SURFACE_TOLERANCE {
            return vec4<f32>(color.rgb * meniscus, color.a);
        }
        path = distance(near, far);
        lighting = underwater_lighting(far, e0, water_plane.x, water_plane.x - far.y);
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
    let lit = color.rgb * lighting * transmittance + scattered * view.exposure;
    return vec4<f32>(lit * meniscus, color.a);
}
