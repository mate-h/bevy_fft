# Shallow water in bevy_fft

The [shallow_water module](../src/shallow_water/mod.rs) runs a **staggered shallow water** solver on the GPU in the CMF10 spirit (Chentanez and Müller): MacCormack advection on staggered face velocities, upwind mass flux for depth `h`, η = H + h for the pressure term, wet–dry reflective faces, optional depth limiting (`h_avgmax`), velocity capping, **PML absorbing strips** after §2.1.4 and appendix 3.1 (auxiliary φ and ψ, quadratic σ / γ in the layer), and optional overshoot limiting near steep η. Brush interaction and passive particles match the style of [webgpu-shallow-water](https://github.com/mate-h/webgpu-shallow-water); the numerics are SWE, not that repo’s pipe network.

Textures: `bed_water` is `Rgba32Float` per cell (`.x` bed H, `.y` water h, η = sum). `flow_x` and `flow_y` are `R32Float` grids for face u and w. `mac_u_temps` and `mac_w_temps` are `Rgba32Float` at the same sizes as `flow_x` / `flow_y` (MacCormack uses `.x` and `.y`). `pml_state` is `Rgba32Float` on cells (`.x` = φ for x-moving damping, `.y` = ψ for y-moving; `.zw` unused). `velocity` stores cell-centered `.xy` for tracers. `ocean_eta` is `R32Float` on cells and holds the target surface for ocean edges. The compute bind group uses eight `read_write` 2D textures, which is Metal’s limit per kernel, so a new field needs to share a texture.

Domain edges: `applyDomainBoundaries` runs after pressure and integration, then again after PML face damping, so wall, source, drain, and waves edges are not overwritten by the sponge.

An ocean edge is closed at the face. Instead, `oceanBand` runs before advection and blends the water depth toward a target surface over `ocean_band_cells` cells just inside the PML, following Chentanez, Müller and Kim 2015, eq. 18. The weight is 1 next to the PML and falls linearly to 0 further in. Velocities are left to the solver. The PML relaxes toward the same target, so what passes the band is absorbed. The target is `pml_eta_rest`, or with `ocean_forcing` on, the FFT ocean's long waves, which `OceanPlugin` writes into `ocean_eta` each frame for a placed grid. On reset that write covers the whole grid, and the solver copies it into every wet cell with a wind-aligned velocity, so the interior starts in the long-wave field.

`loadPreset` builds the bed from a procedural preset, or from `bed_source` when `preset_index` is `SHALLOW_WATER_PRESET_BED_SOURCE`. That texture holds one bed height per cell in solver units, so a grid can start from a terrain. `bed_source_image` builds it from a slice of heights.

The example (`examples/shallow_water/main.rs`) adds lighting and egui controls and does not use the FFT graph; `ShallowWaterPlugin` is standalone.

## Bevy integration

Simulation WGSL lives in [`assets/shallow_water/simulator.wgsl`](../assets/shallow_water/simulator.wgsl). [`ShallowWaterController`](../src/shallow_water/mod.rs) is extracted to the render world; bind groups are built each frame in `prepare_shallow_water_gpu`. The example schedules egui on `EguiPrimaryContextPass`.

`Rg32Float` storage is unreliable on Metal, so this module uses `Rgba32Float` where multi-channel or packed storage is needed, as above.

[`ShallowWaterSurfaceMaterial`](../src/shallow_water/mod.rs) is an `ExtendedMaterial<StandardMaterial, ShallowWaterSurfaceExtension>`: the extension displaces Y from filtered bed + water height and updates normals from height samples.

Default `dt` is **0.08**. Optional PML: set `ShallowWaterController::pml_width` (for example 10), `pml_eta_rest` (target free-surface height; rest depth in the layer is `max(0, η_rest − H)` per cell), **`pml_sigma_exponent`** (**2** = quadratic σ/γ ramp per [Joh08], **3** = cubic), and **`pml_cosine_blend`** (**0** = off, **1** = scale x-strip damping by `|u|/|v|` and y-strip by `|w|/|v|` at cell centers, in the spirit of CMF10 §2.1.4 citing [Joh08]).

## Placing the grid in a larger scene

Set `ShallowWaterController::placement` to a `ShallowWaterPlacement` to put the grid in the world. The placement gives the grid centre, the cell width in meters, the datum that lifts or lowers the solver's heights to the surface mesh plane, and the width of the fade band along the edges. The cell width also becomes the solver's `dx`, so gravity, velocities, and heights are in SI units.

When `WaterPlugin` is in the app, which `OceanPlugin` adds on its own, a compute pass blends the grid into the shared water LOD data every frame. It runs after base writers such as the FFT ocean and before height queries. Inside the grid the solver replaces the long waves its cells can resolve. Shorter FFT cascades stay on top and fade out in the last half meter of water. Dry cells are drawn at the still-water level, so the mesh does not follow the bed onto land. Over the fade band the long waves hand back to the ocean. The foam texture's green channel receives the water depth, which the water surface uses to shade dry and very shallow cells as sand. The flow texture receives the cell velocity, plus the rate the surface rises from the divergence of depth times velocity, so `OceanQueries` report water velocity inside the grid.

`fill_level` fills each preset to a still water level on reset. Cells whose bed rises above it start dry.

## Try it

```bash
cargo run --example shallow_water --features free_camera
cargo run --example beach --features free_camera
```

The example requests `FLOAT32_FILTERABLE` where supported so linear sampling on simulation textures is valid.
