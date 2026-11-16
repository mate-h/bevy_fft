// Draws each water particle as a camera-facing quad, six vertices per pool slot, read from the
// pool by vertex index. Free slots collapse to a point outside the view. The fragment is a soft
// round dot with premultiplied alpha.

import bevy_pbr::render::mesh_view_bindings::view;

struct Particle {
    position: vec3<f32>,
    size: f32,
    velocity: vec3<f32>,
    life: f32,
}

struct ParticleLook {
    /// Premultiplied colour and opacity at the dot's centre.
    color: vec4<f32>,
    /// Seconds over which a particle fades out before it dies.
    fade: f32,
    _pad: vec3<f32>,
}

@group(constants::MATERIAL_BIND_GROUP) @binding(0) var<storage, read> particles: array<Particle>;
@group(constants::MATERIAL_BIND_GROUP) @binding(1) var<uniform> look: ParticleLook;

struct Vertex {
    @builtin(vertex_index) index: u32,
    @location(0) position: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip: vec4<f32>,
    /// Quad coordinate from -1 to 1.
    @location(0) corner: vec2<f32>,
    @location(1) opacity: f32,
};

@vertex
fn vertex(in: Vertex) -> VertexOutput {
    var out: VertexOutput;
    let p = particles[in.index / 6u];
    if p.life <= 0.0 {
        out.clip = vec4<f32>(2.0, 2.0, 2.0, 1.0);
        return out;
    }
    let corners = array<vec2<f32>, 6>(
        vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(1.0, 1.0),
        vec2(-1.0, -1.0), vec2(1.0, 1.0), vec2(-1.0, 1.0),
    );
    let corner = corners[in.index % 6u];
    let right = view.world_from_view[0].xyz;
    let up = view.world_from_view[1].xyz;
    let world = p.position + (right * corner.x + up * corner.y) * (0.5 * p.size);
    out.clip = view.clip_from_world * vec4<f32>(world, 1.0);
    out.corner = corner;
    out.opacity = saturate(p.life / max(look.fade, 1e-4));
    return out;
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let r2 = dot(in.corner, in.corner);
    let dot_alpha = saturate(1.0 - r2) * in.opacity;
    return look.color * dot_alpha;
}
