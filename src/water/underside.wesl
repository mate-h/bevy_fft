// The water surface seen from below and the medium between it and the eye, after Monzon et al.
// (2024, §4.2). Radiance is in exposed linear HDR, like Bevy's lighting output.
//
// Depth `y` runs down from the surface. The model takes the downwelling irradiance to fall as
// `E_D(0) exp(-K_d y)` and lets an optional Lambertian bed send part of it back up (Eqs. 6 to 8).
import bevy_pbr::render::mesh_view_bindings::{view, lights, light_probes};
@if(ENVIRONMENT_MAP) import bevy_pbr::render::mesh_view_bindings as view_bindings;
import bevy_pbr::render::mesh_view_types::DIRECTIONAL_LIGHT_FLAGS_SHADOWS_ENABLED_BIT;
import bevy_pbr::render::shadows::fetch_directional_shadow;
import bevy_pbr::render::utils::interleaved_gradient_noise;
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

struct CausticsLookup {
    /// The covered slice: centre X, centre Z, texel width, and width in meters.
    rect: vec4<f32>,
    /// One when the map was traced this frame, then the receiver count.
    params: vec4<f32>,
    /// Depth of each receiver in meters, shallowest first.
    receiver_depths: vec4<f32>,
}

/// Caustic intensity on each level receiver, one layer per receiver, from the caustics pass.
@group(constants::MATERIAL_BIND_GROUP) @binding(113) var caustic_map: texture_2d_array<f32>;
@group(constants::MATERIAL_BIND_GROUP) @binding(114) var<uniform> caustics: CausticsLookup;

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

/// Bilinear caustic intensity at world `xz` on receiver `receiver`, one outside the map.
fn caustic_at_receiver(xz: vec2<f32>, receiver: i32) -> f32 {
    let n = vec2<f32>(textureDimensions(caustic_map));
    let x = ((xz - caustics.rect.xy) / caustics.rect.w + 0.5) * n - 0.5;
    let i0 = vec2<i32>(floor(x));
    if any(i0 < vec2<i32>(0)) || any(i0 + 1 >= vec2<i32>(n)) {
        return 1.0;
    }
    let f = x - floor(x);
    let a = textureLoad(caustic_map, i0, receiver, 0).r;
    let b = textureLoad(caustic_map, i0 + vec2<i32>(1, 0), receiver, 0).r;
    let c = textureLoad(caustic_map, i0 + vec2<i32>(0, 1), receiver, 0).r;
    let d = textureLoad(caustic_map, i0 + vec2<i32>(1, 1), receiver, 0).r;
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

/// Caustic intensity on receiver `receiver` where the refracted beam `beam` through a point at
/// `xz`, `depth` meters down, crosses it.
fn caustic_along_beam(xz: vec2<f32>, beam: vec3<f32>, depth: f32, receiver: i32) -> f32 {
    let back = (depth - caustics.receiver_depths[receiver]) / max(-beam.y, 1e-3);
    return caustic_at_receiver(xz - beam.xz * back, receiver);
}

/// Caustic intensity of the refracted beam `beam` at point `p` under a level surface at height
/// `surface_y`. The map holds the pattern on a few receivers, so `p` reads the two receivers
/// around its depth where the beam through it crosses them and blends them linearly in depth.
/// Points above the shallowest or below the deepest receiver see that receiver's pattern rather
/// than their own focus.
fn caustic_scale(p: vec3<f32>, beam: vec3<f32>, surface_y: f32) -> f32 {
    if caustics.params.x == 0.0 {
        return 1.0;
    }
    let count = max(i32(caustics.params.y), 1);
    let depth = surface_y - p.y;
    var lower = 0;
    for (var i = 1; i < count; i++) {
        if caustics.receiver_depths[i] <= depth {
            lower = i;
        }
    }
    let upper = min(lower + 1, count - 1);
    let shallow = caustic_along_beam(p.xz, beam, depth, lower);
    let gap = caustics.receiver_depths[upper] - caustics.receiver_depths[lower];
    if upper == lower || gap <= 0.0 {
        return shallow;
    }
    let f = saturate((depth - caustics.receiver_depths[lower]) / gap);
    if f == 0.0 {
        return shallow;
    }
    return mix(shallow, caustic_along_beam(p.xz, beam, depth, upper), f);
}

/// Scale on the light reaching scene point `p`, `depth` meters under the water, relative to the
/// unattenuated light the scene was lit with. `e0` is the downwelling irradiance and `surface_y`
/// the height of the level surface the caustic map refracts through. Sky light fades as
/// `exp(-K_d y)` with depth `y`, the downwelling attenuation of Monzon et al. (2024, Eq. 10). Each
/// directional light's share instead fades over its refracted beam's slant path `y / cos(theta_t)`,
/// as their single scattering does (§4.1), and the first light's share follows the caustic map.
/// Points at or above the surface keep their light.
fn underwater_lighting(p: vec3<f32>, e0: vec3<f32>, surface_y: f32, depth: f32) -> vec3<f32> {
    if depth <= 0.0 {
        return vec3<f32>(1.0);
    }
    let kd = max(medium.diffuse_attenuation, vec3<f32>(0.0));
    let diffuse = exp(-kd * depth);
    let eta = 1.0 / medium.ior;
    var suns = vec3<f32>(0.0);
    var sun_light = vec3<f32>(0.0);
    for (var i = 0u; i < lights.n_directional_lights; i++) {
        let light = &lights.directional_lights[i];
        let L = (*light).direction_to_light.xyz;
        if L.y <= 0.0 {
            continue;
        }
        let beam = refract(-L, vec3<f32>(0.0, 1.0, 0.0), eta);
        let sun = (*light).color.rgb * L.y * (1.0 - fresnel_dielectric(L.y, eta));
        var intensity = 1.0;
        if i == 0u && p.y < surface_y {
            intensity = caustic_scale(p, beam, surface_y);
        }
        suns += sun;
        sun_light += sun * intensity * exp(-kd * depth / max(-beam.y, 1e-3));
    }
    let sky = max(e0 - suns, vec3<f32>(0.0));
    return select((sky * diffuse + sun_light) / e0, diffuse, e0 <= vec3<f32>(0.0));
}

/// Mean cosine of the Henyey-Greenstein phase function for single scattering, from Monzon et al.
/// (2024, §4.1).
const PHASE_G: f32 = 0.8;
/// Samples along each ray for single scattering.
const SINGLE_SCATTERING_STEPS: u32 = 24u;
/// Optical depths along the ray that single scattering covers, measured in the clearest channel.
/// Past that the eye receives less than e^-8 of the light scattered there.
const SINGLE_SCATTERING_REACH: f32 = 8.0;
/// Longest stretch of a ray single scattering covers, in meters.
const SINGLE_SCATTERING_MAX_DISTANCE: f32 = 128.0;

fn henyey_greenstein(cos_theta: f32) -> f32 {
    let g2 = PHASE_G * PHASE_G;
    return (1.0 - g2) / (4.0 * PI * pow(1.0 + g2 - 2.0 * PHASE_G * cos_theta, 1.5));
}

/// Sunlight scattered once toward the eye along `distance` meters of the ray from `start` in
/// `direction`, under a level surface at height `surface_y`, after Monzon et al. (2024) Eq. 5. The
/// result is not yet exposed. Each directional light enters through the level
/// surface with its Fresnel transmittance and bends by Snell's law, as Monzon et al. aim a second,
/// refracted sun into the water. On its way down it fades over the slant path `y / cos(theta)` to
/// depth `y`. §4.1 prints that path as `y cos(theta)`, which would shorten it as the sun sinks.
/// The sum over the ray takes evenly spaced samples, jittered per pixel, and reads the sun's
/// shadow map at each, so geometry under water casts shafts. The first light also follows the
/// caustic map, which draws the shafts below focusing waves, the god rays Papadopoulos and
/// Papaioannou (2009, §3.4) build from the same refracted rays.
fn single_scattering(
    start: vec3<f32>,
    direction: vec3<f32>,
    distance: f32,
    surface_y: f32,
    frag_xy: vec2<f32>,
) -> vec3<f32> {
    let sigma_s = max(medium.scattering, vec3<f32>(0.0));
    if all(sigma_s <= vec3<f32>(0.0)) {
        return vec3<f32>(0.0);
    }
    let sigma_t = max(medium.absorption + medium.scattering, vec3<f32>(0.0));
    let clearest = max(min(sigma_t.x, min(sigma_t.y, sigma_t.z)), 1e-3);
    let reach = min(min(distance, SINGLE_SCATTERING_MAX_DISTANCE), SINGLE_SCATTERING_REACH / clearest);
    let dt = reach / f32(SINGLE_SCATTERING_STEPS);
    let jitter = interleaved_gradient_noise(frag_xy, 0u);
    var sum = vec3<f32>(0.0);
    for (var i = 0u; i < lights.n_directional_lights; i++) {
        let light = &lights.directional_lights[i];
        let L = (*light).direction_to_light.xyz;
        if L.y <= 0.0 {
            continue;
        }
        let eta = 1.0 / medium.ior;
        let transmitted = 1.0 - fresnel_dielectric(L.y, eta);
        let beam_direction = refract(-L, vec3<f32>(0.0, 1.0, 0.0), eta);
        let cos_beam = max(-beam_direction.y, 1e-3);
        // Irradiance across the refracted beam. The level surface spreads the light of a sun at
        // `L.y` over a beam that crosses depth at `cos_beam`.
        let beam = (*light).color.rgb * L.y * transmitted / cos_beam;
        let phase = henyey_greenstein(dot(beam_direction, -direction));
        let shadows = ((*light).flags & DIRECTIONAL_LIGHT_FLAGS_SHADOWS_ENABLED_BIT) != 0u;
        for (var step = 0u; step < SINGLE_SCATTERING_STEPS; step++) {
            let v = (f32(step) + jitter) * dt;
            let p = start + direction * v;
            let depth = max(surface_y - p.y, 0.0);
            var shadow = 1.0;
            if shadows {
                let view_z = (view.view_from_world * vec4<f32>(p, 1.0)).z;
                shadow = fetch_directional_shadow(i, vec4<f32>(p, 1.0), L, view_z, frag_xy);
            }
            if i == 0u {
                shadow *= caustic_scale(p, beam_direction, surface_y);
            }
            sum += exp(-sigma_t * (v + depth / cos_beam)) * shadow * beam * phase;
        }
    }
    return sum * sigma_s * dt;
}

/// Radiance reaching the eye at `eye` from the underside of the surface at `position`, whose
/// upward normal is `n`. Inside Snell's window the sky refracts through the surface. Outside it,
/// total internal reflection shows the lit water body (Monzon et al. 2024, Eq. 11), with both its
/// multiple-scattered and single-scattered light. The medium between the surface and the eye then
/// dims that and adds its own light. `floor_depth` is the water column below the surface point.
fn shade_underside(
    position: vec3<f32>,
    eye: vec3<f32>,
    n: vec3<f32>,
    floor_depth: f32,
    frag_xy: vec2<f32>,
) -> vec3<f32> {
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
    let body = medium_multiple_scattering(0.0, descent, column / descent, column, e0)
        + single_scattering(position, reflected, column / descent, position.y, frag_xy);
    let surface = (1.0 - reflectance) * sky + reflectance * body;
    // The view ray runs from the eye up to the surface point.
    let eye_depth = max(position.y - eye.y, 0.0);
    let sigma_t = max(medium.absorption + medium.scattering, vec3<f32>(0.0));
    let rise = -to_eye.y / max(distance, 1e-6);
    let path = medium_multiple_scattering(eye_depth, -rise, distance, max(column, eye_depth), e0)
        + single_scattering(eye, -V, distance, position.y, frag_xy);
    return (exp(-sigma_t * distance) * surface + path) * view.exposure;
}
