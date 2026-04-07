#import bevy_pbr::mesh_functions::{get_world_from_local, mesh_position_local_to_clip}

struct OceanMaterial {
    amplitude: f32,
    choppiness: f32,
    ocean_size: f32,
    // Keeps the uniform block at 16 bytes for stable std140-style layout.
    _pad: f32,
};

@group(2) @binding(0) var<uniform> material: OceanMaterial;
@group(2) @binding(1) var height_texture: texture_2d<f32>;
@group(2) @binding(2) var height_sampler: sampler;

struct Vertex {
    @builtin(instance_index) instance_index: u32,
    @location(0) position: vec3<f32>,
    @location(1) original_position: vec3<f32>,
    @location(2) normal: vec3<f32>,
    @location(3) uv: vec2<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
};

@vertex
fn vertex(vertex: Vertex) -> VertexOutput {
    var out: VertexOutput;

    let orig_pos = vertex.original_position;

    let size = 256.0;
    let x = (orig_pos.x / material.ocean_size + 0.5) * (size - 1.0);
    let z = (orig_pos.z / material.ocean_size + 0.5) * (size - 1.0);

    let uv = vec2<f32>(x, z) / size;
    let height_sample = textureSample(height_texture, height_sampler, uv);
    let uv_dx = vec2<f32>(uv.x + 1.0 / size, uv.y);
    let uv_dz = vec2<f32>(uv.x, uv.y + 1.0 / size);
    let height_dx = textureSample(height_texture, height_sampler, uv_dx).r * 2.0 - 1.0;
    let height_dz = textureSample(height_texture, height_sampler, uv_dz).r * 2.0 - 1.0;

    let height = (height_sample.r * 2.0 - 1.0) * material.amplitude * 5.0;

    var displaced_position = orig_pos;
    displaced_position.y = height;

    if (material.choppiness > 0.0) {
        let gradient_x = height_dx - height_sample.r * 2.0 + 1.0;
        let gradient_z = height_dz - height_sample.r * 2.0 + 1.0;

        displaced_position.x = orig_pos.x - gradient_x * material.choppiness;
        displaced_position.z = orig_pos.z - gradient_z * material.choppiness;
    }

    let epsilon = 0.01;
    let normal = normalize(vec3<f32>(
        height_dx - height_sample.r,
        epsilon,
        height_dz - height_sample.r
    ));

    let world_from_local = get_world_from_local(vertex.instance_index);
    out.clip_position = mesh_position_local_to_clip(
        world_from_local,
        vec4<f32>(displaced_position, 1.0)
    );

    out.world_position = (world_from_local * vec4<f32>(displaced_position, 1.0)).xyz;
    out.world_normal = normalize((world_from_local * vec4<f32>(normal, 0.0)).xyz);
    out.uv = vertex.uv;

    return out;
}
