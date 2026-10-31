// Terrain-aware optical composition over the opaque scene, in exposed linear HDR.
import bevy_pbr::render::mesh_view_bindings::{view, view_transmission_texture, view_transmission_sampler};
import bevy_pbr::render::pbr_types::PbrInput;
import bevy_pbr::render::pbr_functions::apply_pbr_lighting;
import bevy_pbr::render::view_transformations::position_ndc_to_world;

struct WaterOpticsUniform {
    extinction: vec3<f32>,
    refraction_strength: f32,
    scattering: vec3<f32>,
    contact_fade: f32,
    max_distance: f32,
    ssr_strength: f32,
    debug_view: u32,
    enabled: u32,
}
@group(constants::MATERIAL_BIND_GROUP) @binding(106) var opaque_depth: texture_depth_2d;
@group(constants::MATERIAL_BIND_GROUP) @binding(107) var<uniform> optics: WaterOpticsUniform;

struct Receiver { position: vec3<f32>, valid: bool }
struct RayHit { uv: vec2<f32>, distance: f32, confidence: f32 }

fn receiver_at(uv: vec2<f32>) -> Receiver {
    var result: Receiver;
    if any(uv <= vec2<f32>(0.0)) || any(uv >= vec2<f32>(1.0)) { return result; }
    let size = vec2<f32>(textureDimensions(opaque_depth));
    let pixel = vec2<i32>(uv * size);
    let depth = textureLoad(opaque_depth, pixel, 0);
    if depth <= 0.0 { return result; }
    let viewport_uv = ((vec2<f32>(pixel) + 0.5) - view.viewport.xy) / view.viewport.zw;
    let ndc = vec3<f32>(viewport_uv * vec2<f32>(2.0, -2.0) + vec2<f32>(-1.0, 1.0), depth);
    result.position = position_ndc_to_world(ndc);
    result.valid = true;
    return result;
}

// Screen UV and reverse-Z depth of a clip-space point. Clip space is linear along a world ray,
// so the march interpolates two clip positions instead of transforming every sample.
fn clip_to_uv_depth(clip: vec4<f32>) -> vec3<f32> {
    if clip.w <= 0.0 { return vec3<f32>(-1.0); }
    let viewport_uv = clip.xy / clip.w * vec2<f32>(0.5, -0.5) + 0.5;
    let uv = (viewport_uv * view.viewport.zw + view.viewport.xy) / vec2<f32>(textureDimensions(opaque_depth));
    return vec3<f32>(uv, clip.z / clip.w);
}

// True where the opaque surface at the projected point is at or in front of it. Sky never is.
fn behind_scene(projected: vec3<f32>) -> bool {
    if any(projected.xy <= vec2<f32>(0.0)) || any(projected.xy >= vec2<f32>(1.0)) { return false; }
    let pixel = vec2<i32>(projected.xy * vec2<f32>(textureDimensions(opaque_depth)));
    let depth = textureLoad(opaque_depth, pixel, 0);
    return depth > 0.0 && depth >= projected.z;
}

// Bracket a crossing of the opaque depth surface, then refine it. Hits outside the screen,
// across a depth discontinuity, or on the wrong side of the water interface are rejected.
fn trace_scene(origin: vec3<f32>, direction: vec3<f32>, limit: f32, reflection: bool) -> RayHit {
    var result: RayHit;
    let clip_origin = view.clip_from_world * vec4<f32>(origin, 1.0);
    let clip_direction = view.clip_from_world * vec4<f32>(direction, 0.0);
    var previous = 0.025;
    for (var i = 1u; i <= 20u; i++) {
        let fraction = f32(i) / 20.0;
        let distance = 0.025 + fraction * fraction * limit;
        let projected = clip_to_uv_depth(clip_origin + clip_direction * distance);
        if projected.z < 0.0 || any(projected.xy <= vec2<f32>(0.0)) || any(projected.xy >= vec2<f32>(1.0)) { break; }
        if behind_scene(projected) {
            var lo = previous;
            var hi = distance;
            for (var j = 0u; j < 6u; j++) {
                let mid = (lo + hi) * 0.5;
                if behind_scene(clip_to_uv_depth(clip_origin + clip_direction * mid)) { hi = mid; }
                else { lo = mid; }
            }
            let ray_point = origin + direction * hi;
            let hit_uv = clip_to_uv_depth(clip_origin + clip_direction * hi).xy;
            let hit = receiver_at(hit_uv);
            if !hit.valid { return result; }
            let tolerance = max(0.08, distance - previous);
            if length(hit.position - ray_point) > tolerance { return result; }
            if reflection && hit.position.y < origin.y + 0.025 { return result; }
            if !reflection && hit.position.y > origin.y - 0.001 { return result; }
            let edge = min(min(hit_uv.x, 1.0 - hit_uv.x), min(hit_uv.y, 1.0 - hit_uv.y));
            result.uv = hit_uv;
            result.distance = length(hit.position - origin);
            result.confidence = smoothstep(0.0, 0.04, edge);
            return result;
        }
        previous = distance;
    }
    return result;
}

fn scene_color(uv: vec2<f32>) -> vec3<f32> {
    return textureSampleLevel(view_transmission_texture, view_transmission_sampler, uv, 0.0).rgb;
}

fn shade_water(input: PbrInput, solver_depth: f32, foam: f32) -> vec4<f32> {
    if any(textureDimensions(opaque_depth) != textureDimensions(view_transmission_texture)) {
        return apply_pbr_lighting(input);
    }
    let full_size = vec2<f32>(textureDimensions(opaque_depth));
    let uv = input.frag_coord.xy / full_size;
    let receiver = receiver_at(uv);
    let background = scene_color(uv);
    var path = optics.max_distance;
    var column = solver_depth;
    if receiver.valid {
        path = length(receiver.position - input.world_position.xyz);
        column = max(0.0, input.world_position.y - receiver.position.y);
    }
    let contact_width = max(optics.contact_fade, min(fwidth(path), 0.25));
    let wet_width = max(fwidth(solver_depth), 0.0001);
    let coverage = smoothstep(0.0, contact_width, path) * smoothstep(0.0, wet_width, solver_depth);
    if solver_depth <= 0.0 || coverage <= 0.0 { discard; }
    var bottom = background;
    var refraction_confidence = 0.0;
    let thin_fade = smoothstep(0.0, 0.3, min(column, solver_depth));
    if receiver.valid && optics.refraction_strength > 0.0 && thin_fade > 0.0 {
        let ray = refract(-input.V, input.N, 1.0 / max(input.material.ior, 1.0));
        let hit = trace_scene(input.world_position.xyz, ray, min(optics.max_distance, path * 1.5 + 1.0), false);
        refraction_confidence = hit.confidence;
        let weight = clamp(optics.refraction_strength, 0.0, 1.0) * thin_fade * hit.confidence;
        if weight > 0.0 {
            bottom = mix(background, scene_color(hit.uv), weight);
            path = mix(path, hit.distance, weight);
        }
    }
    let transmittance = exp(-max(optics.extinction, vec3<f32>(0.0)) * min(path, optics.max_distance));
    let f0 = 0.16 * input.material.reflectance * input.material.reflectance;
    let fresnel = f0 + (vec3<f32>(1.0) - f0) * pow(1.0 - clamp(dot(input.N, input.V), 0.0, 1.0), 5.0);
    var reflection_input = input;
    reflection_input.material.specular_transmission = 0.0;
    reflection_input.material.base_color = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    reflection_input.material.perceptual_roughness = mix(input.material.perceptual_roughness, 0.65, foam);
    // Lighting the scattering body also yields the environment reflection as its specular part.
    // The reflection alone is lit separately only where a screen-space hit replaces it.
    var body_input = reflection_input;
    body_input.material.base_color = vec4<f32>(mix(optics.scattering * (1.0 - transmittance), vec3<f32>(0.85), foam), 1.0);
    var lit = apply_pbr_lighting(body_input).rgb;
    if optics.ssr_strength > 0.0 && foam < 0.95 {
        let hit = trace_scene(input.world_position.xyz, reflect(-input.V, input.N), optics.max_distance, true);
        let weight = hit.confidence * optics.ssr_strength * (1.0 - foam);
        if weight > 0.0 {
            let environment_reflection = apply_pbr_lighting(reflection_input).rgb;
            lit += (scene_color(hit.uv) * fresnel - environment_reflection) * weight;
        }
    }
    let transmitted = (vec3<f32>(1.0) - fresnel) * transmittance * bottom * (1.0 - foam);
    var color = mix(background, lit + transmitted, coverage);
    if optics.debug_view == 1u { color = vec3<f32>(clamp(path / 5.0, 0.0, 1.0)); }
    if optics.debug_view == 2u { color = transmittance; }
    if optics.debug_view == 3u { color = vec3<f32>(coverage); }
    if optics.debug_view == 4u { color = vec3<f32>(refraction_confidence); }
    return vec4<f32>(color, 1.0);
}
