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

## White water

Rapids, hydraulic jumps and the feet of waterfalls look white because the water is full of bubbles, not only because foam floats on top. The LOD data carries the air fraction of the water column for this: the volume of air over the volume of air and water, averaged over the depth.

Heightfield writers estimate how much air the flow can hold in each texel and store it in the `aeration` array of `WaterLodData`, which is emptied each frame. `src/water/aeration.wesl` has three sources, and a texel takes the largest:

- Jumps and bores. The flow slows across a jump, and that drop in speed is the same in every frame of reference, so it serves for a standing jump and a moving bore alike. The writer measures it over at least two cells either side of the texel, since the solver spreads a jump over two to four cells. It divides the drop by the wave speed on the shallow side and turns that into the inflow Froude number `Fr1` with Bélanger's conjugate depths (Chanson 2009, eq. 1). Below `Fr1` = 1.5 a jump is undular and has no roller (Chanson 2009, §2.3). Above it the air grows linearly to 33 percent at `Fr1` = 2, the median Vallé and Pasternack (2006) measured over the roller of a natural jump at that Froude number, and stops at 50 percent, their peak.
- Steep fast flow. Self-aerated flow down a chute holds 0.9 sin θ of air in equilibrium (Chanson's 1993 fit to Straub and Anderson 1958). It fades in from Froude number 1 to 2 and from 1 to 2 m/s.
- Waterfalls. The CMF10 waterfall test that throws spray off lips also marks the lip cell and the cell where the water lands with 20 percent air, the 15 to 25 percent Vallé and Pasternack found at the toe of a jump below a plunging jet. It fades in as the impact speed, `sqrt(2 g drop)`, goes from 1 to 3 m/s.

The foam sim keeps the air in the blue lane of `foam_state`, an `Rgba32Float` array. It carries the air along the drift with the same MacCormack step as the foam, but as a concentration, so it does not gather where the drift converges. Where the writers ask for more air than the column holds, the air rises toward that value over `WaterAerationSettings::formation_time`, 0.08 s. Elsewhere bubbles leave the column: the air falls by e every `depth / detrainment_speed` seconds, so a deep plunge pool stays white for longer than a riffle. The defaults come from the unsubmerged jump Vallé and Pasternack surveyed: 0.27 m of water at `Fr1` = 2 reached its peak air 0.25 m past the toe, 0.08 s at the inflow speed, and fell below 2 percent 2.8 m past it. At the tailwater's 1.4 m/s over about 0.64 m of water, that is a detrainment speed near 1 m/s. It is faster than a millimetre bubble rises alone, since the roller's air packets are larger and the probe saw only the top 0.1 m.

The air that leaves the column rises to the surface, and part of it stays there as foam. The sim adds `WaterAerationSettings::foam_yield`, 20, of residual foam coverage for each meter of air per unit area that surfaces. A layer of millimetre bubbles holds about a millimetre of air and most surfacing bubbles burst, so that is about one bubble in fifty surviving into lasting foam, an appearance choice. Water that keeps an air fraction `C` holds about `69 C` of residual foam with the default foam settings, so the lace covers the whole tail down to about 1.5 percent of air and then drifts on with the current until it fades. That is where white water trails foam past the end of its air.

The surface treats the air as a cloud of bubbles filling the solver's water column. Bubbles of radius r scatter 3 C / (2 r) per meter, and a mean cosine near 0.85 leaves 15 percent of that to turn light around. For a spread of sizes r is the Sauter mean radius, three times the air's volume over its surface. A jump roller holds bubbles from a fraction of a millimetre to centimetres, and air packets larger still. The small bubbles are the most common, but the large ones hold most of the air, so the Sauter radius sits well above the most common bubble; the same large packets explain the detrainment speed above. The surface takes 4 mm. Over 0.2 m of water, 5 percent of air hides 43 percent of the bed and 30 percent hides 97 percent. Bubbles of 2 mm would hide 68 percent at 5 percent of air, so every patch would be opaque to its edge and look like a decal. With 4 mm the edges grade through green-white water, where the air thins, to clear water. The layer's opacity is `1 - exp(-scattering * depth)`, and the refraction trace stops where the layer is opaque. Its color is Kubelka and Munk's reflectance of a deep layer with the water's extinction as absorption, so thinly aerated water takes the water's tint and dense white water goes white, scaled by the opacity and by light the surface reflects back down into it. Dense white water comes out near 60 percent, a little brighter than foam. Foam is drawn on top as before. Surfaces drawn without the water optics mix the same layer into their base color with the absorption of pure water.

The foam sim carries only the mean air fraction of each texel, but the air in a roller comes in packets and clusters, which a probe in the flow records as an intermittent signal. The surface spreads the mean into a pattern with `water_air_texture` in `surface.wesl`. It reads the foam texture's cells through the same advected texture coordinates as the foam (Neyret 2003), so the pattern moves with the water and starts over before the flow stretches it. Each sample averages four taps along the drift over 0.6 s of travel, up to 2.5 m, in texture space. The averaging window moves with the coordinates, so the streaks it draws ride the current rather than standing in it. A second, single sample at 2.7 times the scale gathers the air into patches several meters across that still show from afar. The air at a pixel is the mean times `exp(k z - k² / 2)`, with `z` the pattern at unit spread and `k` = 0.85, which keeps the mean air the same and never goes negative. Where a pixel covers several of the pattern's texels the pattern fades out and the mean remains. The pattern costs 10 to 20 texture reads, and only on pixels where some pixel of the quad holds air.

From below, the air diffuses the surface's light the way foam does, so the underside mixes toward the foam's look by the air's opacity over the column. The underwater medium pass reads the air at the eye and a few meters along the view ray and clouds the view with it: the ray's light fades over the air's transport scattering toward the diffuse light at that depth, `E_D(0) exp(-K_d y) / π`.

[`tests/water_aeration_gpu.rs`](../tests/water_aeration_gpu.rs) holds stationary jumps from 0.2 m of water. At `Fr1` = 2 the air peaks at 0.34 at the face, and at `Fr1` = 3 it reaches the 0.5 cap. Downstream it falls by e every 0.7 to 1 m, near the 0.6 to 0.8 m the tailwater speed times depth over the detrainment speed gives. Water pouring over a 3 m drop holds 0.2 at the lip and the landing. Still water, gently varying subcritical flow, and uniform supercritical flow hold none. The air that leaves the `Fr1` = 2 jump leaves residual foam that covers the tailwater 2 m past the face and then halves once per residual half-life as it drifts, as far as 10 m downstream.

## Waves too short to see

A slice of the water LOD data keeps only the waves its texels resolve, so distant water would lose the roughness of the short waves and its sun glitter would shrink to a thin line. The ocean integrates each cascade's mean square slope from its spectrum on the CPU, next to its Stokes drift, and the LOD writer stores in the foam lane's alpha the share of it each slice drops. The surface adds that variance to its GGX roughness, as Bruneton et al. (2010) widen the sea's reflection lobe by its unresolved slopes, so the glitter spreads into a wide, even path toward the horizon. Other writers keep the ocean's value. The deferred prepass writes the widened roughness into the G-buffer, so water drawn deferred shows the same glitter as forward water. That roughness also brings distant water into the range where Bevy's `ScreenSpaceReflections` act, and their single stochastic ray per pixel speckles on the waves without a temporal filter, so the `ocean` example starts them at a perceptual roughness of 0.45, above the water's.

## Breaker curl

A heightfield can't overhang, so the lip of a plunging breaker is drawn as a separate mesh. `WaterBreakerCurl`, next to `WaterBreakingFronts` on a heightfield, sweeps one animated cross-section along each traced front, forward along its travel, as Malan (2022) does for Horizon Forbidden West: a steep face, a lip thrown forward over it, the lip falling onto the water ahead, and a flat bore. A front's age plays it over `duration` seconds, so the steady front ids keep the curl playing through while the front breaks. The cross-section is hand-authored keyframes rather than Malan's baked simulation, and ends flat the same way. Its height is 0.78 of the depth at the front, McCowan's breaker index, within `height_range`. Foam grows over the lip as it breaks. The traced lines step from cell to cell, so positions, directions and heights are smoothed along them first, and the curl tapers to nothing at a line's ends. The mesh is rebuilt on the CPU at each readback. The beach draws it, and `BEACH_CURL=0` turns it off.

## Flow detail

Flowing water gets short ripples that move with it, too small for any solver grid. The foam texture holds a tiling field of ripple slopes, 12 to 60 cm long, in its green and blue channels. The surface reads them through the same advected texture coordinates as the foam pattern (Neyret 2003), so they ride the drift, start over unseen and never stretch. Their slopes add to the waves' and the foam relief's. `WaterSurfaceUniform::flow_detail` is their root mean square slope at full strength, 0.06 by default and zero for none. They grow with the drift speed from 0.15 to 1 m/s, so the ocean's Stokes drift stays clear of them, and with the strain the flow puts on them, from half to twice their strength, as Chentanez and Müller-Fischer (2010) weight their subgrid waves. They fade out where a pixel covers several texels of them. The deferred prepass adds them to the normal it writes into the G-buffer, as the forward pass does.

## Inspect and validate

The Beach panel's Water view shows shaded water, optical distance, transmittance, coverage, or successful refraction hits. Optical distance is displayed from black at zero to white at five meters. Set Refraction to zero to inspect direct transmission without changing the depth measurement.

For headless captures, `BEVY_FFT_WATER_VIEW` accepts `depth`, `transmission`, `coverage`, or `refraction`. The usual screenshot variables still apply.

Run the GPU regression test explicitly on a machine with a graphics adapter:

```sh
cargo test --test water_transmission_gpu --features free_camera -- --ignored --nocapture
```

It checks that clear water preserves colored receivers, a 10 cm path produces the expected Beer-Lambert attenuation, refraction finds submerged geometry, the depth snapshot resizes with the render target, and emerged terrain occludes the water.

The rendering approach follows [Ojeda and Susín, section 5](https://web.mat.upc.edu/toni.susin/files/hybrid_lbmsw_rend_new.pdf). Path-dependent extinction is also discussed by [Monzon et al.](https://graphics.unizar.es/projects/EG24Underwater/). The RGB scattering approximation and the contact reconstruction here are implementation choices.
