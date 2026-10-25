// Ground material for the Kauai beach example, drawn by bevy_terrain.
//
// Colors come from height alone. Below sea level the bed is wet sand that darkens toward reef
// rock with depth, since the water surface is drawn on top. Above it, sand gives way to vegetation
// and then rock.

import bevy_terrain::shaders::types::AtlasTile;
import bevy_terrain::shaders::bindings::terrain_data;
import bevy_terrain::shaders::attachments::{sample_height, sample_height_mask, sample_surface_gradient};
import bevy_terrain::shaders::render::fragment::{FragmentInput, FragmentOutput, fragment_info, fragment_output};
@if(!DEFERRED_PREPASS) import bevy_terrain::shaders::render::fragment::fragment_debug;
import bevy_terrain::shaders::functions::lookup_tile;

fn ground_color(height: f32) -> vec4<f32> {
    let sand = vec3<f32>(0.78, 0.71, 0.52);
    if height < 0.0 {
        let wet = vec3<f32>(0.55, 0.49, 0.36);
        let reef = vec3<f32>(0.24, 0.26, 0.22);
        let depth = clamp(-height / max(-terrain_data.terrain.min_height, 1.0), 0.0, 1.0);
        return vec4<f32>(mix(wet, reef, smoothstep(0.1, 0.7, depth)), 1.0);
    }
    let t = clamp(height / max(terrain_data.terrain.max_height, 1.0), 0.0, 1.0);
    let vegetation = vec3<f32>(0.20, 0.40, 0.17);
    let rock = vec3<f32>(0.42, 0.40, 0.37);
    let low = mix(sand, vegetation, smoothstep(0.015, 0.06, t));
    return vec4<f32>(mix(low, rock, smoothstep(0.5, 0.95, t)), 1.0);
}

@fragment
fn fragment(input: FragmentInput) -> FragmentOutput {
    var info = fragment_info(input);

    let tile = lookup_tile(info.coordinate, info.blend);
    let mask = sample_height_mask(tile);
    let color = ground_color(sample_height(tile));
    let surface_gradient = sample_surface_gradient(tile, info.tangent_space);

    if mask {
        discard;
    }

    var output: FragmentOutput;
    fragment_output(&info, &output, color, surface_gradient);
    @if(!DEFERRED_PREPASS)
    fragment_debug(&info, &output, tile, surface_gradient);
    return output;
}
