// Poly Haven ground materials selected by the co-registered coastline classes.
import bevy_terrain::shaders::bindings::{terrain_data};
@if(TERRAIN_SHADOW) import bevy_terrain::shaders::bindings::{shadow_map, shadow_map_sampler, terrain_shadow};
@if(TERRAIN_SHADOW) import bevy_terrain::shaders::terrain_shadow::terrain_shadow_factor_at;
import bevy_terrain::shaders::attachments::sample_surface_gradient;
import bevy_terrain::shaders::render::fragment::{FragmentInput, FragmentOutput, fragment_info};
import bevy_terrain::shaders::functions::{lookup_tile, apply_height};
import bevy_pbr::render::pbr_types::{pbr_input_new, STANDARD_MATERIAL_FLAGS_UNLIT_BIT};
import bevy_pbr::render::pbr_functions::{calculate_view, apply_pbr_lighting};
import bevy_pbr::render::mesh_types::MESH_FLAGS_SHADOW_RECEIVER_BIT;
@if(DEFERRED_PREPASS) import bevy_pbr::deferred::functions::deferred_gbuffer_from_pbr_input;

struct Settings { repeat_a: vec4<f32>, repeat_b: vec4<f32>, domain: vec4<f32> }
@group(4) @binding(0) var albedo: texture_2d_array<f32>;
@group(4) @binding(1) var material_sampler: sampler;
@group(4) @binding(2) var normal_roughness: texture_2d_array<f32>;
@group(4) @binding(3) var surface_classes: texture_2d<f32>;
@group(4) @binding(4) var<storage, read> ground_wetness: array<f32>;
@group(4) @binding(5) var<uniform> settings: Settings;

struct Surface { color: vec3<f32>, normal: vec3<f32>, roughness: f32 }

fn class_at(cell: vec2<i32>) -> u32 {
    let hi = vec2<i32>(textureDimensions(surface_classes)) - 1;
    return u32(round(textureLoad(surface_classes, clamp(cell, vec2<i32>(0), hi), 0).r * 255.0));
}

fn moisture_at(cell: vec2<i32>) -> f32 {
    let n = i32(settings.domain.y);
    let c = clamp(cell, vec2<i32>(0), vec2<i32>(n - 1));
    return ground_wetness[u32(c.y * n + c.x)];
}

// Land and submerged continuations share a layer. IDs remain categorical at every lookup.
fn layer_for_class(id: u32) -> u32 {
    if id == 4u || id == 9u { return 0u; }
    if id == 5u { return 2u; }
    if id == 6u { return 3u; }
    if id == 7u { return 4u; }
    if id == 8u { return 5u; }
    if id == 11u || id == 12u { return 6u; }
    if id == 1u { return 7u; }
    // Exposed rock, submerged hardbottom, and the ground beneath mapped structures.
    return 1u;
}

// Blend translated scans on a triangular lattice to hide repeated patches without changing scale.
struct TexturePair { color: vec4<f32>, detail: vec4<f32> }
fn tile_hash(cell: vec2<f32>) -> vec2<f32> {
    return fract(sin(vec2<f32>(dot(cell, vec2<f32>(127.1, 311.7)), dot(cell, vec2<f32>(269.5, 183.3)))) * 43758.5453);
}
fn value_noise(p: vec2<f32>) -> vec2<f32> {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    let a = tile_hash(i);
    let b = tile_hash(i + vec2<f32>(1.0, 0.0));
    let c = tile_hash(i + vec2<f32>(0.0, 1.0));
    let d = tile_hash(i + vec2<f32>(1.0, 1.0));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y) * 2.0 - 1.0;
}

// Offset in class texels that bends the raster's axis-aligned label edges into organic outlines.
fn class_warp(xz: vec2<f32>) -> vec2<f32> {
    return value_noise(xz / 7.0) * 1.6 + value_noise(xz / 2.3 + 17.0) * 0.6;
}

fn ground_texture(uv: vec2<f32>, layer: i32, dx: vec2<f32>, dy: vec2<f32>) -> TexturePair {
    let cell = floor(uv);
    let f = fract(uv);
    var offsets = array<vec2<f32>, 3>(vec2<f32>(0.0), vec2<f32>(1.0, 0.0), vec2<f32>(0.0, 1.0));
    var weights = vec3<f32>(1.0 - f.x - f.y, f.x, f.y);
    if f.x + f.y > 1.0 {
        offsets = array<vec2<f32>, 3>(vec2<f32>(1.0), vec2<f32>(0.0, 1.0), vec2<f32>(1.0, 0.0));
        weights = vec3<f32>(f.x + f.y - 1.0, 1.0 - f.x, 1.0 - f.y);
    }
    weights = weights * weights;
    weights /= dot(weights, vec3<f32>(1.0));
    var out: TexturePair;
    for (var i = 0u; i < 3u; i++) {
        let shifted = uv + tile_hash(cell + offsets[i]);
        out.color += textureSampleGrad(albedo, material_sampler, shifted, layer, dx, dy) * weights[i];
        out.detail += textureSampleGrad(normal_roughness, material_sampler, shifted, layer, dx, dy) * weights[i];
    }
    return out;
}

fn material_at(id: u32, position: vec3<f32>, dx: vec3<f32>, dy: vec3<f32>, normal: vec3<f32>, wetness: f32) -> Surface {
    let layer = layer_for_class(id);
    var repeat = settings.repeat_a[min(layer, 3u)];
    if layer >= 4u { repeat = settings.repeat_b[layer - 4u]; }
    let p = position / repeat;
    let x = dx / repeat;
    let y = dy / repeat;
    let side = sign(normal + vec3<f32>(1e-5));
    // Projections under this weight are skipped. Flat ground then reads one projection, not three.
    var weights = pow(abs(normal), vec3<f32>(4.0));
    weights = select(vec3<f32>(0.0), weights, weights > vec3<f32>(0.02 * max(weights.x, max(weights.y, weights.z))));
    weights /= max(dot(weights, vec3<f32>(1.0)), 1e-5);
    var color = vec3<f32>(0.0);
    var roughness = 0.0;
    var detail = vec3<f32>(0.0);
    if weights.x > 0.0 {
        let s = ground_texture(vec2<f32>(-p.z * side.x, p.y), i32(layer), vec2<f32>(-x.z * side.x, x.y), vec2<f32>(-y.z * side.x, y.y));
        let t = (s.detail.xy * 2.0 - 1.0) / max(s.detail.z * 2.0 - 1.0, 0.3);
        color += s.color.rgb * weights.x;
        roughness += s.detail.a * weights.x;
        detail += vec3<f32>(0.0, t.y, -t.x * side.x) * weights.x;
    }
    if weights.y > 0.0 {
        let s = ground_texture(vec2<f32>(p.x, -p.z * side.y), i32(layer), vec2<f32>(x.x, -x.z * side.y), vec2<f32>(y.x, -y.z * side.y));
        let t = (s.detail.xy * 2.0 - 1.0) / max(s.detail.z * 2.0 - 1.0, 0.3);
        color += s.color.rgb * weights.y;
        roughness += s.detail.a * weights.y;
        detail += vec3<f32>(t.x, 0.0, -t.y * side.y) * weights.y;
    }
    if weights.z > 0.0 {
        let s = ground_texture(vec2<f32>(p.x * side.z, p.y), i32(layer), vec2<f32>(x.x * side.z, x.y), vec2<f32>(y.x * side.z, y.y));
        let t = (s.detail.xy * 2.0 - 1.0) / max(s.detail.z * 2.0 - 1.0, 0.3);
        color += s.color.rgb * weights.z;
        roughness += s.detail.a * weights.z;
        detail += vec3<f32>(t.x * side.z, t.y, 0.0) * weights.z;
    }
    var out: Surface;
    out.color = color;
    out.roughness = roughness;
    if layer == 1u { out.color *= vec3<f32>(0.18, 0.20, 0.22); }
    if layer == 3u { out.color *= vec3<f32>(0.45, 0.80, 0.40); }
    if layer == 4u { out.color *= vec3<f32>(0.28, 0.55, 0.22); }
    let sand = id == 4u || id == 9u;
    let wet = select(0.0, wetness, sand);
    out.color *= mix(1.0, 0.45, wet);
    out.roughness = mix(clamp(out.roughness, 0.45, 1.0), 0.23, wet);
    detail *= 0.35 * (1.0 - 0.45 * wet);
    out.normal = normalize(normal + detail - normal * dot(normal, detail));
    return out;
}


fn class_color(id: u32) -> vec3<f32> {
    let colors = array<vec3<f32>, 13>(
        vec3<f32>(0.1), vec3<f32>(0.84, 0.19, 0.15), vec3<f32>(0.55, 0.18, 0.02),
        vec3<f32>(0.45), vec3<f32>(0.96, 0.91, 0.63), vec3<f32>(0.65, 0.85, 0.42),
        vec3<f32>(0.40, 0.65, 0.12), vec3<f32>(0.11, 0.47, 0.22), vec3<f32>(0.17, 0.55, 0.75),
        vec3<f32>(0.78, 0.70, 0.44), vec3<f32>(0.30, 0.36, 0.43), vec3<f32>(0.88, 0.40, 0.64), vec3<f32>(0.53, 0.34, 0.65));
    return colors[min(id, 12u)];
}

@fragment
fn fragment(input: FragmentInput) -> FragmentOutput {
    let info = fragment_info(input);
    let tile = lookup_tile(info.coordinate, info.blend);
    // The Kauai tiles carry no height mask. Without a discard, Apple GPUs keep hidden surface
    // removal and shade each covered pixel once.
    let world = apply_height(info.world_coordinate, info.height);
    let gradient = sample_surface_gradient(tile, info.tangent_space);
    let normal = normalize(info.world_coordinate.normal - gradient);
    let dx = dpdx(world);
    let dy = dpdy(world);
    let grid = (world.xz / settings.domain.x + 0.5) * settings.domain.y - 0.5;
    let i0 = vec2<i32>(floor(grid));
    let f = fract(grid);
    let wet = mix(mix(moisture_at(i0), moisture_at(i0 + vec2<i32>(1, 0)), f.x), mix(moisture_at(i0 + vec2<i32>(0, 1)), moisture_at(i0 + vec2<i32>(1, 1)), f.x), f.y);
    let class_grid = grid + class_warp(world.xz);
    let c0 = vec2<i32>(floor(class_grid));
    let cf = fract(class_grid);
    let ida = class_at(c0);
    let idb = class_at(c0 + vec2<i32>(1, 0));
    let idc = class_at(c0 + vec2<i32>(0, 1));
    let idd = class_at(c0 + vec2<i32>(1, 1));
    let a = material_at(ida, world, dx, dy, normal, wet);
    var b = a;
    if idb != ida { b = material_at(idb, world, dx, dy, normal, wet); }
    var c = a;
    if idc == idb { c = b; } else if idc != ida { c = material_at(idc, world, dx, dy, normal, wet); }
    var d = a;
    if idd == idb { d = b; } else if idd == idc { d = c; } else if idd != ida { d = material_at(idd, world, dx, dy, normal, wet); }
    var color = mix(mix(a.color, b.color, cf.x), mix(c.color, d.color, cf.x), cf.y);
    if settings.domain.z > 0.5 && settings.domain.z < 1.5 { color = class_color(class_at(vec2<i32>(floor(class_grid + 0.5)))); }
    if settings.domain.z > 1.5 {
        let id = class_at(vec2<i32>(floor(class_grid + 0.5)));
        color = vec3<f32>(select(0.0, wet, id == 4u || id == 9u));
    }
    let shading_normal = normalize(mix(mix(a.normal, b.normal, cf.x), mix(c.normal, d.normal, cf.x), cf.y));
    let roughness = mix(mix(a.roughness, b.roughness, cf.x), mix(c.roughness, d.roughness, cf.x), cf.y);
    var pbr = pbr_input_new();
    pbr.material.base_color = vec4<f32>(color, 1.0);
    pbr.material.perceptual_roughness = roughness;
    pbr.material.reflectance = vec3<f32>(0.5);
    pbr.frag_coord = info.clip_position;
    pbr.world_position = vec4<f32>(world, 1.0);
    pbr.world_normal = normal;
    pbr.N = shading_normal;
    pbr.flags = MESH_FLAGS_SHADOW_RECEIVER_BIT;
    pbr.V = calculate_view(pbr.world_position, pbr.is_orthographic);
@if(TERRAIN_SHADOW)
    pbr.directional_shadow_factor = terrain_shadow_factor_at(world, terrain_shadow, shadow_map, shadow_map_sampler);
    var output: FragmentOutput;
@if(DEFERRED_PREPASS) {
    if settings.domain.z < 0.5 { pbr.material.base_color = apply_pbr_lighting(pbr); }
    pbr.material.flags |= STANDARD_MATERIAL_FLAGS_UNLIT_BIT;
    output.deferred = deferred_gbuffer_from_pbr_input(pbr);
    output.deferred_lighting_pass_id = pbr.material.deferred_lighting_pass_id;
}
@if(!DEFERRED_PREPASS) {
    output.color = vec4<f32>(color, 1.0);
    if settings.domain.z < 0.5 { output.color = apply_pbr_lighting(pbr); }
}
    return output;
}
