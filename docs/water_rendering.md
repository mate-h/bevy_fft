# Water over coastal terrain

Transmissive water preserves the actual terrain material beneath the surface. Sand, basalt, and wetness remain visible through thin water. Increasing the optical path attenuates the bottom color and adds a simple scattering approximation. Fresnel reflection supplies the surface sheen.

The beach example uses this path. Its Ground view and Water view controls inspect the classified terrain and the water independently.

## Enable transmission

Add `WaterTransmission` to the camera that owns the shared water data. It requires `WaterViewer`, HDR, a depth prepass, and screen-space transmission. It defaults to `Msaa::Off`. The current depth snapshot supports one active water viewer with single-sample rendering.

The material must use forward rendering and positive `specular_transmission` so Bevy queues it after opaque geometry and leaves it out of the opaque depth and deferred prepasses.

```rust
use bevy::material::OpaqueRendererMethod;
use bevy::prelude::*;
use bevy_fft::prelude::*;

commands.spawn((Camera3d::default(), WaterTransmission, Msaa::Off));
commands.spawn((
    WaterQuadtree::default(),
    MeshMaterial3d(materials.add(WaterSurfaceMaterial {
        base: StandardMaterial {
            base_color: Color::WHITE,
            specular_transmission: 1.0,
            perceptual_roughness: 0.08,
            reflectance: 0.356,
            ior: 1.333,
            opaque_render_method: OpaqueRendererMethod::Forward,
            ..default()
        },
        extension: WaterSurfaceExtension {
            optics: WaterOpticsUniform { enabled: 1, ..default() },
            ..default()
        },
    })),
    bevy::light::NotShadowCaster,
));
```

The reflectance above gives a normal-incidence Fresnel reflectivity near 0.02. The water surface does not cast an opaque terrain shadow in this example. Other opaque or deferred water materials remain available with `optics.enabled` left at zero.

## Color and depth contract

The water plugin snapshots the main scene depth after the opaque pass and before the transmissive pass. This includes the rendered terrain and ordinary meshes. The snapshot excludes the transmissive water. Bevy's transmission pass provides the corresponding opaque scene color in exposed linear HDR.

The shader reconstructs world positions through the inverse projection, including viewport offsets. Optical distance comes from the water entry point to the opaque receiver. It is distinct from the shallow-water solver's vertical column depth.

The optical model combines Fresnel-weighted reflection, transmitted terrain color attenuated by `exp(-extinction * distance)`, and a lit RGB scattering approximation. Extinction coefficients are in inverse meters. This is a surface renderer with an approximate volume color, rather than a full spectral underwater lighting model.

Refraction traces the refracted viewing ray against the preserved opaque depth and refines a bracketed intersection. It rejects off-screen hits, large depth discontinuities, and receivers above the water entry point. Rejected hits use the direct terrain lookup. Refraction fades in over the first 0.3 m of water so the exposed ground remains continuous with a thin water film.

Water reflections use the same opaque scene snapshot. A failed reflection trace retains environment lighting. `WaterOpticsUniform::ssr_strength` controls these water reflections. The camera's `ScreenSpaceReflections` component continues to handle opaque deferred materials separately. Screen-space tracing cannot recover off-screen or hidden geometry and has no temporal history filter.

## Shoreline coverage

The solver depth controls simulated wet-region coverage. The rendered opaque depth supplies occlusion and optical thickness. Terrain in front of the water wins the ordinary depth test. Zero wet coverage discards the fragment, and contact color blends over a screen-space footprint with a small metric minimum. The beach disables the separate `dry_depth` cutoff.

The renderer does not lift the water mesh to hide intersections. Water geometry still samples the shared LOD data, so a solver grid and mesh that undersample the shoreline can limit the outline. The numerical wet/dry tolerance remains a solver setting, separate from optical attenuation and antialiasing.

Persistent moisture belongs to the ground material. Foam is separate coverage over the water and reduces transmitted light where present. Neither uses a replacement sand color inside the water shader.

## Inspect and validate

The Beach panel's Water view shows shaded water, optical distance, transmittance, coverage, or successful refraction hits. Optical distance is displayed from black at zero to white at five meters. Set Refraction to zero to inspect direct transmission without changing the depth measurement.

For headless captures, `BEVY_FFT_WATER_VIEW` accepts `depth`, `transmission`, `coverage`, or `refraction`. The usual screenshot variables still apply.

Run the GPU regression test explicitly on a machine with a graphics adapter:

```sh
cargo test --test water_transmission_gpu --features free_camera -- --ignored --nocapture
```

It checks that clear water preserves colored receivers, a 10 cm path produces the expected Beer-Lambert attenuation, refraction finds submerged geometry, the depth snapshot resizes with the render target, and emerged terrain occludes the water.

The rendering approach follows [Ojeda and Susín, section 5](https://web.mat.upc.edu/toni.susin/files/hybrid_lbmsw_rend_new.pdf). Path-dependent extinction is also discussed by [Monzon et al.](https://graphics.unizar.es/projects/EG24Underwater/). The RGB scattering approximation and the contact reconstruction here are implementation choices.
