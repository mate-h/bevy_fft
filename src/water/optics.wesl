// Terrain-aware optical composition over the opaque scene, in exposed linear HDR.
import bevy_pbr::render::mesh_view_bindings::{lights, view, view_transmission_texture, view_transmission_sampler};
@if(ATMOSPHERE) import bevy_pbr::render::mesh_view_bindings::atmosphere;
import bevy_fft::water::underside::{
    downwelling_irradiance,
    environment_irradiance,
    environment_radiance,
    underwater_lighting,
};
import bevy_pbr::render::pbr_types::PbrInput;
import bevy_pbr::render::pbr_functions::apply_pbr_lighting;
import bevy_pbr::render::mesh_types::MESH_FLAGS_SHADOW_RECEIVER_BIT;
import bevy_pbr::render::mesh_view_types::DIRECTIONAL_LIGHT_FLAGS_SHADOWS_ENABLED_BIT;
import bevy_pbr::render::pbr_lighting::{D_GGX, F_AB, V_SmithGGXCorrelated};
@if(ATMOSPHERE) import bevy_pbr::render::pbr_lighting::sample_transmittance_lut;
@if(ATMOSPHERE) import bevy_pbr::atmosphere::functions::{calculate_visible_sun_ratio, clamp_to_surface};
import bevy_pbr::render::shadows::fetch_directional_shadow;
import bevy_pbr::render::view_transformations::position_ndc_to_world;
import bevy_render::maths::PI;

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

/// Reflectance of dense foam. Matches `FOAM_ALBEDO` in `surface.wesl`.
const FOAM_ALBEDO: f32 = 0.55;
/// Reflectance of the bubble cloud that breaking drives under active foam, before the water above
/// it absorbs. An appearance choice, since the sources here give no measured value.
const BUBBLE_REFLECTANCE: f32 = 0.2;
/// Depth in meters of the bubble cloud, which sets how much the water tints its light on the way
/// down and back up.
const BUBBLE_DEPTH: f32 = 0.5;

struct Receiver { position: vec3<f32>, valid: bool }
struct RayHit { uv: vec2<f32>, distance: f32, confidence: f32, position: vec3<f32> }

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

// A world ray in homogeneous screen space: `xy` is the depth texture pixel times `w`, `z` the
// reverse-Z depth times `w`. Projection is linear in homogeneous coordinates, so the march
// interpolates two of these instead of transforming every sample, and each sample takes one
// division.
struct ScreenRay { origin: vec4<f32>, direction: vec4<f32>, size: vec2<f32> }

fn screen_ray(origin: vec3<f32>, direction: vec3<f32>) -> ScreenRay {
    var ray: ScreenRay;
    ray.size = vec2<f32>(textureDimensions(opaque_depth));
    // pixel = (ndc.xy * (0.5, -0.5) + 0.5) * viewport size + viewport offset, applied before the
    // division by w.
    let scale = vec2<f32>(0.5, -0.5) * view.viewport.zw;
    let offset = 0.5 * view.viewport.zw + view.viewport.xy;
    let clip_origin = view.clip_from_world * vec4<f32>(origin, 1.0);
    let clip_direction = view.clip_from_world * vec4<f32>(direction, 0.0);
    ray.origin = vec4<f32>(clip_origin.xy * scale + clip_origin.w * offset, clip_origin.zw);
    ray.direction = vec4<f32>(clip_direction.xy * scale + clip_direction.w * offset, clip_direction.zw);
    return ray;
}

// Depth texture pixel in `xy` and reverse-Z depth in `z` of the ray point `distance` along it.
// `z` is negative behind the eye.
fn screen_point(ray: ScreenRay, distance: f32) -> vec3<f32> {
    let h = ray.origin + ray.direction * distance;
    if h.w <= 0.0 { return vec3<f32>(-1.0); }
    return h.xyz / h.w;
}

fn on_screen(point: vec3<f32>, size: vec2<f32>) -> bool {
    return all(point.xy > vec2<f32>(0.0)) && all(point.xy < size);
}

// True where the opaque surface at the point is at or in front of it. Sky never is.
fn behind_scene(point: vec3<f32>, size: vec2<f32>) -> bool {
    if !on_screen(point, size) { return false; }
    let depth = textureLoad(opaque_depth, vec2<i32>(point.xy), 0);
    return depth > 0.0 && depth >= point.z;
}

/// March and bisection steps of the screen-space traces.
const TRACE_STEPS: u32 = 12u;
const TRACE_REFINE_STEPS: u32 = 5u;

// Bracket a crossing of the opaque depth surface, then refine it. Hits outside the screen,
// across a depth discontinuity, or on the wrong side of the water interface are rejected.
fn trace_scene(origin: vec3<f32>, direction: vec3<f32>, limit: f32, reflection: bool) -> RayHit {
    var result: RayHit;
    let ray = screen_ray(origin, direction);
    var previous = 0.025;
    for (var i = 1u; i <= TRACE_STEPS; i++) {
        let fraction = f32(i) / f32(TRACE_STEPS);
        let distance = 0.025 + fraction * fraction * limit;
        let point = screen_point(ray, distance);
        if point.z < 0.0 || !on_screen(point, ray.size) { break; }
        if behind_scene(point, ray.size) {
            var lo = previous;
            var hi = distance;
            for (var j = 0u; j < TRACE_REFINE_STEPS; j++) {
                let mid = (lo + hi) * 0.5;
                if behind_scene(screen_point(ray, mid), ray.size) { hi = mid; }
                else { lo = mid; }
            }
            let ray_point = origin + direction * hi;
            let hit_uv = screen_point(ray, hi).xy / ray.size;
            let hit = receiver_at(hit_uv);
            if !hit.valid { return result; }
            let tolerance = max(0.08, distance - previous);
            if length(hit.position - ray_point) > tolerance { return result; }
            if reflection && hit.position.y < origin.y + 0.025 { return result; }
            if !reflection && hit.position.y > origin.y - 0.001 { return result; }
            let edge = min(min(hit_uv.x, 1.0 - hit_uv.x), min(hit_uv.y, 1.0 - hit_uv.y));
            result.uv = hit_uv;
            result.position = hit.position;
            result.distance = length(hit.position - origin);
            result.confidence = smoothstep(0.0, 0.04, edge);
            return result;
        }
        previous = distance;
    }
    return result;
}

/// Light the water surface sends toward the eye: diffuse light from a layer of color `albedo`, and
/// the sun and the environment reflected. `environment_specular` is the part of `color` that
/// reflects the environment map, which a screen-space hit replaces. Radiance is exposed.
struct SurfaceLight { color: vec3<f32>, environment_specular: vec3<f32> }

/// Bevy's `apply_pbr_lighting` for an opaque dielectric, reduced to the light sources the water's
/// scenes have: directional lights with their shadows and the atmosphere's transmittance, the
/// view environment map, and ambient light. Point, spot and area lights and reflection probes do
/// not light the water. The full function holds so much state for those that the water program
/// ran out of registers.
fn surface_light(input: PbrInput, albedo: vec3<f32>) -> SurfaceLight {
    let N = input.N;
    let V = input.V;
    let perceptual_roughness = input.material.perceptual_roughness;
    let clamped_roughness = clamp(perceptual_roughness, 0.089, 1.0);
    let roughness = clamped_roughness * clamped_roughness;
    let NdotV = max(dot(N, V), 0.0001);
    let F0 = 0.16 * input.material.reflectance * input.material.reflectance;
    let F_ab = F_AB(perceptual_roughness, NdotV);
    // Fresnel at grazing angles, which Bevy also uses as its baked specular occlusion.
    let f90 = saturate(dot(F0, vec3<f32>(50.0 * 0.33)));
    let energy = 1.0 + F0 * (1.0 / (F_ab.x + F_ab.y) - 1.0);
    let view_z = dot(vec4<f32>(
        view.view_from_world[0].z,
        view.view_from_world[1].z,
        view.view_from_world[2].z,
        view.view_from_world[3].z,
    ), input.world_position);
    let receives_shadows = (input.flags & MESH_FLAGS_SHADOW_RECEIVER_BIT) != 0u;
    var direct = vec3<f32>(0.0);
    for (var i = 0u; i < lights.n_directional_lights; i++) {
        let light = &lights.directional_lights[i];
        let L = (*light).direction_to_light.xyz;
        let NdotL = saturate(dot(N, L));
        if NdotL <= 0.0 {
            continue;
        }
        let H = normalize(L + V);
        let NdotH = saturate(dot(N, H));
        let LdotH = saturate(dot(L, H));
        // Burley diffuse and the GGX lobe with Bevy's multiple-scattering compensation.
        let burley_f90 = 0.5 + 2.0 * roughness * LdotH * LdotH;
        let light_scatter = 1.0 + (burley_f90 - 1.0) * pow(1.0 - NdotL, 5.0);
        let view_scatter = 1.0 + (burley_f90 - 1.0) * pow(1.0 - NdotV, 5.0);
        let diffuse = albedo * (light_scatter * view_scatter / PI);
        let F = F0 + (f90 - F0) * pow(1.0 - LdotH, 5.0);
        let specular = D_GGX(roughness, NdotH) * V_SmithGGXCorrelated(roughness, NdotV, NdotL) * F * energy;
        var color = (diffuse + specular) * NdotL * (*light).color.rgb;
        @if(ATMOSPHERE) {
            let P = clamp_to_surface(atmosphere, (atmosphere.world_to_atmosphere * vec4<f32>(input.world_position.xyz, 1.0)).xyz);
            let r = length(P);
            let mu = dot(L, P / r);
            color *= sample_transmittance_lut(r, mu)
                * calculate_visible_sun_ratio(atmosphere, r, mu, (*light).sun_disk_angular_size);
        }
        if receives_shadows && ((*light).flags & DIRECTIONAL_LIGHT_FLAGS_SHADOWS_ENABLED_BIT) != 0u {
            color *= fetch_directional_shadow(i, input.world_position, input.world_normal, view_z, input.frag_coord.xy)
                * input.directional_shadow_factor;
        }
        direct += color;
    }
    // The environment map through the split-sum approximation, with Bevy's multiple scattering.
    let FssEss = (F0 * F_ab.x + F_ab.y) * f90;
    let Ems = 1.0 - (F_ab.x + F_ab.y);
    let Favg = F0 + (1.0 - F0) / 21.0;
    let FmsEms = FssEss * Favg / (1.0 - Ems * Favg) * Ems;
    let Edss = 1.0 - (FssEss + FmsEms);
    let radiance = environment_radiance(N, reflect(-V, N), perceptual_roughness);
    let irradiance = environment_irradiance(N);
    let environment_specular = FssEss * radiance * input.specular_occlusion;
    var indirect = (FmsEms + albedo * Edss) * irradiance * input.diffuse_occlusion + environment_specular;
    let diffuse_ab = F_AB(1.0, NdotV);
    let ambient = (albedo * diffuse_ab.x + diffuse_ab.y) + (F0 * F_ab.x + F_ab.y) * f90;
    indirect += ambient * lights.ambient_color.rgb * input.diffuse_occlusion;
    var out: SurfaceLight;
    out.color = view.exposure * (direct + indirect);
    out.environment_specular = view.exposure * environment_specular;
    return out;
}

fn scene_color(uv: vec2<f32>) -> vec3<f32> {
    return textureSampleLevel(view_transmission_texture, view_transmission_sampler, uv, 0.0).rgb;
}

/// Water over the opaque scene. `foam` is the opacity of the foam layer on top and `bubbles` the
/// coverage of active foam, whose bubble cloud brightens the water below it. `surface_y` is the
/// height of the undisplaced surface. Beds below it fade with depth and take the caustic map.
fn shade_water(input: PbrInput, solver_depth: f32, foam: f32, bubbles: f32, surface_y: f32) -> vec4<f32> {
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
    // The scene was lit as if no water were above it, so the bed takes the light that reaches its
    // depth, with the caustic pattern of the sun's refracted beam on the sun's share.
    let e0 = downwelling_irradiance();
    var bottom = background;
    if receiver.valid {
        bottom *= underwater_lighting(receiver.position, e0, surface_y, surface_y - receiver.position.y);
    }
    let direct = bottom;
    var refraction_confidence = 0.0;
    let thin_fade = smoothstep(0.0, 0.3, min(column, solver_depth));
    if receiver.valid && optics.refraction_strength > 0.0 && thin_fade > 0.0 {
        let ray = refract(-input.V, input.N, 1.0 / max(input.material.ior, 1.0));
        let hit = trace_scene(input.world_position.xyz, ray, min(optics.max_distance, path * 1.5 + 1.0), false);
        refraction_confidence = hit.confidence;
        let weight = clamp(optics.refraction_strength, 0.0, 1.0) * thin_fade * hit.confidence;
        if weight > 0.0 {
            let refracted = scene_color(hit.uv)
                * underwater_lighting(hit.position, e0, surface_y, surface_y - hit.position.y);
            bottom = mix(direct, refracted, weight);
            path = mix(path, hit.distance, weight);
        }
    }
    let transmittance = exp(-max(optics.extinction, vec3<f32>(0.0)) * min(path, optics.max_distance));
    let f0 = 0.16 * input.material.reflectance * input.material.reflectance;
    let fresnel = f0 + (vec3<f32>(1.0) - f0) * pow(1.0 - clamp(dot(input.N, input.V), 0.0, 1.0), 5.0);
    // Lighting the scattering body also yields the environment reflection as its specular part,
    // which a screen-space hit replaces.
    let extinction = max(optics.extinction, vec3<f32>(0.0));
    let bubble = saturate(bubbles) * BUBBLE_REFLECTANCE * exp(-2.0 * extinction * BUBBLE_DEPTH);
    let body = optics.scattering * (1.0 - transmittance) + bubble;
    let surface = surface_light(input, mix(body, vec3<f32>(FOAM_ALBEDO), foam));
    var lit = surface.color;
    if optics.ssr_strength > 0.0 && foam < 0.95 {
        let R = reflect(-input.V, input.N);
        let hit = trace_scene(input.world_position.xyz, R, optics.max_distance, true);
        let weight = hit.confidence * optics.ssr_strength * (1.0 - foam);
        if weight > 0.0 {
            lit += (scene_color(hit.uv) * fresnel - surface.environment_specular) * weight;
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
