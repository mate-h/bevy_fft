# Shallow water in bevy_fft

The [shallow_water module](../src/shallow_water/mod.rs) runs a **staggered shallow water** solver on the GPU in the CMF10 spirit (Chentanez and Müller): MacCormack advection on staggered face velocities, upwind mass flux for depth `h`, η = H + h for the pressure term, wet–dry reflective faces, optional depth limiting (`h_avgmax`), velocity capping, **PML absorbing strips** after §2.1.4 and appendix 3.1 (auxiliary φ and ψ, quadratic σ / γ in the layer), and optional overshoot limiting near steep η. Brush interaction and passive particles match the style of [webgpu-shallow-water](https://github.com/mate-h/webgpu-shallow-water); the numerics are SWE, not that repo’s pipe network.

Textures: `bed_water` is `Rgba32Float` per cell (`.x` bed H, `.y` water h, η = sum). `flow_x` and `flow_y` are `R32Float` grids for face u and w. `mac_u_temps` and `mac_w_temps` are `Rgba32Float` at the same sizes as `flow_x` / `flow_y` (MacCormack uses `.x` and `.y`). `pml_state` is `Rgba32Float` on cells (`.x` = φ for x-moving damping, `.y` = ψ for y-moving; `.zw` unused). `velocity` stores cell-centered `.xy` for tracers. The compute bind group uses seven `read_write` 2D textures so it stays within Metal’s eight-texture limit per kernel.

Domain edges: `applyDomainBoundaries` runs after pressure and integration, then again after PML face damping, so wall / source / drain / waves are not overwritten by the sponge.

The example (`examples/shallow_water/main.rs`) adds lighting and egui controls and does not use the FFT graph; `ShallowWaterPlugin` is standalone.

## Bevy integration

Simulation WGSL lives in [`assets/shallow_water/simulator.wgsl`](../assets/shallow_water/simulator.wgsl). [`ShallowWaterController`](../src/shallow_water/mod.rs) is extracted to the render world; bind groups are built each frame in `prepare_shallow_water_gpu`. The example schedules egui on `EguiPrimaryContextPass`.

`Rg32Float` storage is unreliable on Metal, so this module uses `Rgba32Float` where multi-channel or packed storage is needed, as above.

[`ShallowWaterSurfaceMaterial`](../src/shallow_water/mod.rs) is an `ExtendedMaterial<StandardMaterial, ShallowWaterSurfaceExtension>`: the extension displaces Y from filtered bed + water height and updates normals from height samples.

Default `dt` is **0.08**. Optional PML: set `ShallowWaterController::pml_width` (for example 10), `pml_eta_rest` (target free-surface height; rest depth in the layer is `max(0, η_rest − H)` per cell), **`pml_sigma_exponent`** (**2** = quadratic σ/γ ramp per [Joh08], **3** = cubic), and **`pml_cosine_blend`** (**0** = off, **1** = scale x-strip damping by `|u|/|v|` and y-strip by `|w|/|v|` at cell centers, in the spirit of CMF10 §2.1.4 citing [Joh08]).

## Try it

```bash
cargo run --example shallow_water --features free_camera
```

The example requests `FLOAT32_FILTERABLE` where supported so linear sampling on simulation textures is valid.
