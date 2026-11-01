// The water surface seen from below and the medium between it and the eye, after Monzon et al.
// (2024, §4.2). Radiance is in exposed linear HDR, like Bevy's lighting output.
//
// Depth `y` runs down from the surface. The model takes the downwelling irradiance to fall as
// `E_D(0) exp(-K_d y)` and lets an optional Lambertian bed send part of it back up (Eqs. 6 to 8).
import bevy_pbr::render::mesh_view_bindings::{view, lights, light_probes};
@if(ENVIRONMENT_MAP) import bevy_pbr::render::mesh_view_bindings as view_bindings;
import bevy_render::maths::{PI, quat_rotate};

struct WaterMediumUniform {
    absorption: vec3<f32>,
    floor_albedo: f32,
    scattering: vec3<f32>,
    ior: f32,
    diffuse_attenuation: vec3<f32>,
    _pad: f32,
}

@group(constants::MATERIAL_BIND_GROUP) @binding(112) var<uniform> medium: WaterMediumUniform;

/// Longest path in meters the medium terms integrate over.
const MEDIUM_MAX_PATH: f32 = 10000.0;

/// Specular radiance of the view environment map along `R`, as Bevy's lighting samples it.
fn environment_radiance(N: vec3<f32>, R: vec3<f32>, perceptual_roughness: f32) -> vec3<f32> {
    var radiance = vec3<f32>(0.0);
    @if(ENVIRONMENT_MAP) {
        if light_probes.view_cubemap_index >= 0 {
            let roughness = perceptual_roughness * perceptual_roughness;
            let smoothness = saturate(1.0 - roughness);
            var direction = quat_rotate(light_probes.view_rotation, mix(N, R, smoothness * (sqrt(smoothness) + roughness)));
            // Cube maps are left-handed.
            direction.z = -direction.z;
            let level = perceptual_roughness * f32(light_probes.smallest_specular_mip_level_for_view);
            @if(MULTIPLE_LIGHT_PROBES_IN_ARRAY) {
                radiance = textureSampleLevel(view_bindings::specular_environment_maps[light_probes.view_cubemap_index], view_bindings::environment_map_sampler, direction, level).rgb;
            }
            @if(!MULTIPLE_LIGHT_PROBES_IN_ARRAY) {
                radiance = textureSampleLevel(view_bindings::specular_environment_map, view_bindings::environment_map_sampler, direction, level).rgb;
            }
            radiance *= light_probes.intensity_for_view;
        }
    }
    return radiance;
}

/// Irradiance the sky sends onto a horizontal surface, from the view's diffuse environment map.
/// Bevy stores that map as irradiance over pi.
fn environment_irradiance_down() -> vec3<f32> {
    var irradiance = vec3<f32>(0.0);
    @if(ENVIRONMENT_MAP) {
        if light_probes.view_cubemap_index >= 0 {
            var direction = quat_rotate(light_probes.view_rotation, vec3<f32>(0.0, 1.0, 0.0));
            direction.z = -direction.z;
            @if(MULTIPLE_LIGHT_PROBES_IN_ARRAY) {
                irradiance = textureSampleLevel(view_bindings::diffuse_environment_maps[light_probes.view_cubemap_index], view_bindings::environment_map_sampler, direction, 0.0).rgb;
            }
            @if(!MULTIPLE_LIGHT_PROBES_IN_ARRAY) {
                irradiance = textureSampleLevel(view_bindings::diffuse_environment_map, view_bindings::environment_map_sampler, direction, 0.0).rgb;
            }
            irradiance *= PI * light_probes.intensity_for_view;
        }
    }
    return irradiance;
}

/// Unpolarized Fresnel reflectance for light arriving at `cos_i` inside a medium of relative index
/// `eta`, the index on the incident side over the index on the far side. One past the critical
/// angle, where all light reflects.
fn fresnel_dielectric(cos_i: f32, eta: f32) -> f32 {
    let c = saturate(cos_i);
    let sin_t2 = eta * eta * (1.0 - c * c);
    if sin_t2 >= 1.0 {
        return 1.0;
    }
    let cos_t = sqrt(1.0 - sin_t2);
    let rs = (eta * c - cos_t) / (eta * c + cos_t);
    let rp = (c - eta * cos_t) / (c + eta * cos_t);
    return 0.5 * (rs * rs + rp * rp);
}

/// Downwelling irradiance just below a level surface, E_D(0). Each directional light enters
/// through the Fresnel transmittance at its elevation. Sky light enters without Fresnel loss, which
/// slightly overstates it.
fn downwelling_irradiance() -> vec3<f32> {
    var e = environment_irradiance_down();
    for (var i = 0u; i < lights.n_directional_lights; i++) {
        let light = &lights.directional_lights[i];
        let cos_sun = (*light).direction_to_light.y;
        if cos_sun > 0.0 {
            let transmitted = 1.0 - fresnel_dielectric(cos_sun, 1.0 / medium.ior);
            e += (*light).color.rgb * cos_sun * transmitted;
        }
    }
    return e;
}

/// Integral of `exp(-x v)` for `v` from zero to `p`, with the limit `p` where `x` is zero.
fn medium_segment(x: vec3<f32>, p: f32) -> vec3<f32> {
    let small = abs(x * p) < vec3<f32>(1e-4);
    let safe = select(x, vec3<f32>(1.0), small);
    let integral = (1.0 - exp(min(-safe * p, vec3<f32>(80.0)))) / safe;
    return select(integral, vec3<f32>(p), small);
}

/// Multiple-scattered radiance gathered along a ray that starts `depth` meters below the surface,
/// runs `distance` meters, and moves down by `descent` meters per meter. `floor_depth` is the depth
/// of the bed. `e0` is E_D(0).
///
/// This is Monzon et al. (2024) Eq. 9 with an isotropic phase function. Integrating Eq. 8 along the
/// ray gives each of its two terms its own exponent: the downwelling term fades by
/// `sigma_t + K_d descent` per meter, the light from the bed by `sigma_t - K_d descent`, since the
/// bed term grows toward the bed. Eq. 9 as printed factors both under the first exponent.
fn medium_multiple_scattering(
    depth: f32,
    descent: f32,
    distance: f32,
    floor_depth: f32,
    e0: vec3<f32>,
) -> vec3<f32> {
    let kd = max(medium.diffuse_attenuation, vec3<f32>(0.0));
    let sigma_t = max(medium.absorption + medium.scattering, vec3<f32>(0.0));
    let p = min(distance, MEDIUM_MAX_PATH);
    let down = exp(-kd * depth) * medium_segment(sigma_t + kd * descent, p);
    let up = medium.floor_albedo / PI * exp(-kd * (2.0 * floor_depth - depth))
        * medium_segment(sigma_t - kd * descent, p);
    return medium.scattering / (4.0 * PI) * e0 * (down + up);
}

/// Radiance reaching the eye at `eye` from the underside of the surface at `position`, whose
/// upward normal is `n`. Inside Snell's window the sky refracts through the surface. Outside it,
/// total internal reflection shows the lit water body (Monzon et al. 2024, Eq. 11). The medium
/// between the surface and the eye then dims that and adds its own light. `floor_depth` is the
/// water column below the surface point.
fn shade_underside(position: vec3<f32>, eye: vec3<f32>, n: vec3<f32>, floor_depth: f32) -> vec3<f32> {
    let to_eye = eye - position;
    let distance = length(to_eye);
    let V = to_eye / max(distance, 1e-6);
    // Normal on the water side, facing the eye.
    var N = -n;
    if dot(N, V) < 0.0 {
        N = normalize(N - 2.0 * dot(N, V) * V);
    }
    let cos_i = dot(N, V);
    let e0 = downwelling_irradiance();
    let reflectance = fresnel_dielectric(cos_i, medium.ior);
    var sky = vec3<f32>(0.0);
    if reflectance < 1.0 {
        let refracted = refract(-V, N, medium.ior);
        sky = environment_radiance(-N, refracted, 0.0);
    }
    let reflected = reflect(-V, N);
    let descent = max(-reflected.y, 1e-3);
    let column = max(floor_depth, 0.0);
    let body = medium_multiple_scattering(0.0, descent, column / descent, column, e0);
    let surface = (1.0 - reflectance) * sky + reflectance * body;
    // The view ray runs from the eye up to the surface point.
    let eye_depth = max(position.y - eye.y, 0.0);
    let sigma_t = max(medium.absorption + medium.scattering, vec3<f32>(0.0));
    let rise = -to_eye.y / max(distance, 1e-6);
    let path = medium_multiple_scattering(eye_depth, -rise, distance, max(column, eye_depth), e0);
    return (exp(-sigma_t * distance) * surface + path) * view.exposure;
}
