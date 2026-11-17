// Draws each live water particle as a camera-facing quad, six vertices per slot in the draw list
// the simulation fills, with an indirect draw of that many vertices.
//
// A fast drop is stretched along its velocity across the screen, as a camera exposure smears it,
// keeping its total opacity. Its size on screen is clamped: a drop smaller than
// `look.min_pixels` is drawn at that size with its opacity cut by the area it lacks, so distant
// spray thins into a haze instead of aliasing away, and one nearer than `look.max_pixels` is
// capped to bound the fill. Each drop scatters the sun with a Henyey-Greenstein phase function,
// bright toward the light, plus the sky's ambient light, and fades out where it nears the
// geometry behind it in the depth prepass, so it never cuts a hard line into the water or the
// shore.

import bevy_pbr::render::mesh_view_bindings::{view, lights};
import bevy_pbr::render::view_transformations::depth_ndc_to_view_z;
@if(DEPTH_PREPASS) import bevy_pbr::prepass::utils::prepass_depth;

struct Particle {
    position: vec3<f32>,
    size: f32,
    velocity: vec3<f32>,
    life: f32,
}

struct ParticleLook {
    /// Albedo of a drop in `.rgb` and its opacity at the centre in `.a`.
    color: vec4<f32>,
    /// Sky light on the drops in lux, per channel.
    ambient: vec4<f32>,
    /// Seconds over which a particle fades out before it dies.
    fade: f32,
    /// Seconds of motion a drop is smeared over.
    stretch_time: f32,
    /// Henyey-Greenstein asymmetry of the scattering, from -1 to 1. Positive throws the light
    /// forward.
    anisotropy: f32,
    /// Distance in meters over which a drop fades into the geometry behind it.
    soft_distance: f32,
    min_pixels: f32,
    max_pixels: f32,
    /// Opacity of mist, marked by a negative size, relative to a drop's.
    mist_opacity: f32,
    _pad: f32,
}

@group(constants::MATERIAL_BIND_GROUP) @binding(0) var<storage, read> particles: array<Particle>;
@group(constants::MATERIAL_BIND_GROUP) @binding(1) var<uniform> look: ParticleLook;
/// The indirect draw's arguments, then the live slots, six vertices each.
@group(constants::MATERIAL_BIND_GROUP) @binding(2) var<storage, read> draw: array<u32>;

struct Vertex {
    @builtin(vertex_index) index: u32,
    @location(0) position: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip: vec4<f32>,
    /// Quad coordinate from -1 to 1.
    @location(0) corner: vec2<f32>,
    @location(1) opacity: f32,
    /// Direction from the camera to the drop.
    @location(2) forward: vec3<f32>,
};

const PI: f32 = 3.14159265;

fn henyey_greenstein(cos_theta: f32, g: f32) -> f32 {
    let denom = 1.0 + g * g - 2.0 * g * cos_theta;
    return (1.0 - g * g) / (4.0 * PI * denom * sqrt(denom));
}

@vertex
fn vertex(in: Vertex) -> VertexOutput {
    var out: VertexOutput;
    // Past the list only when the mesh draw stands in for the indirect one.
    if in.index >= draw[0] {
        out.clip = vec4<f32>(2.0, 2.0, 2.0, 1.0);
        return out;
    }
    let p = particles[draw[4u + in.index / 6u]];
    if p.life <= 0.0 {
        out.clip = vec4<f32>(2.0, 2.0, 2.0, 1.0);
        return out;
    }
    let corners = array<vec2<f32>, 6>(
        vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(1.0, 1.0),
        vec2(-1.0, -1.0), vec2(1.0, 1.0), vec2(-1.0, 1.0),
    );
    let corner = corners[in.index % 6u];
    let camera = view.world_position;
    let to_drop = p.position - camera;
    let distance = max(length(to_drop), 1e-3);
    let forward = to_drop / distance;

    // Pixels per meter at this distance.
    let pixels_per_meter = view.clip_from_view[1][1] * 0.5 * view.viewport.w / distance;
    var width = abs(p.size);
    var opacity = look.color.a * saturate(p.life / max(look.fade, 1e-4));
    if p.size < 0.0 {
        opacity *= look.mist_opacity;
    }
    let pixels = width * pixels_per_meter;
    if pixels < look.min_pixels {
        opacity *= (pixels / look.min_pixels) * (pixels / look.min_pixels);
        width = look.min_pixels / pixels_per_meter;
    } else if pixels > look.max_pixels {
        width = look.max_pixels / pixels_per_meter;
    }

    // Smear along the velocity across the view, spreading the same opacity over a longer quad.
    let across = p.velocity - dot(p.velocity, forward) * forward;
    let smear = length(across) * look.stretch_time;
    var axis = view.world_from_view[0].xyz;
    if smear > 1e-4 {
        axis = across / length(across);
    }
    // Up on screen when the smear runs right, so the quad keeps the winding back-face culling
    // passes.
    let side = normalize(cross(axis, forward));
    let length_along = width + smear;
    opacity *= width / length_along;
    let world = p.position + axis * corner.x * 0.5 * length_along + side * corner.y * 0.5 * width;
    out.clip = view.clip_from_world * vec4<f32>(world, 1.0);
    out.corner = corner;
    out.opacity = opacity;
    out.forward = forward;
    return out;
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let r2 = dot(in.corner, in.corner);
    var alpha = saturate(1.0 - r2) * in.opacity;
    @if(DEPTH_PREPASS) {
        let scene = -depth_ndc_to_view_z(prepass_depth(in.clip, 0u));
        let drop = -depth_ndc_to_view_z(in.clip.z);
        alpha *= saturate((scene - drop) / max(look.soft_distance, 1e-4));
    }
    // Sunlight scattered toward the camera, and the sky's. The lights are bound for fragments
    // only.
    var light = look.ambient.rgb / PI;
    for (var i = 0u; i < lights.n_directional_lights; i++) {
        let sun = lights.directional_lights[i];
        let cos_theta = dot(in.forward, -sun.direction_to_light);
        light += sun.color.rgb * henyey_greenstein(cos_theta, look.anisotropy);
    }
    return vec4<f32>(look.color.rgb * light * view.exposure * alpha, alpha);
}
