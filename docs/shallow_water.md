# Shallow water in bevy_fft

The [shallow_water module](../src/shallow_water/mod.rs) runs a **staggered shallow water** solver on the GPU in the CMF10 spirit (Chentanez and Müller): MacCormack advection on staggered face velocities, upwind mass flux for depth `h`, η = H + h for the pressure term, wet–dry reflective faces, optional depth limiting (`h_avgmax`), velocity capping, **PML absorbing strips** after §2.1.4 and appendix 3.1 (auxiliary φ and ψ, quadratic σ / γ in the layer), and optional overshoot limiting near steep η. Brush interaction and passive particles match the style of [webgpu-shallow-water](https://github.com/mate-h/webgpu-shallow-water); the numerics are SWE, not that repo’s pipe network.

Textures: `bed_water` is `Rgba32Float` per cell (`.x` bed H, `.y` water h, η = sum). `flow_x` and `flow_y` are `R32Float` grids for face u and w. `mac_u_temps` and `mac_w_temps` are `Rgba32Float` at the same sizes as `flow_x` / `flow_y` (MacCormack uses `.x` and `.y`). `pml_state` is `Rgba32Float` on cells (`.x` = φ for x-moving damping, `.y` = ψ for y-moving; `.zw` unused). `velocity` stores cell-centered `.xy` for tracers. `ocean_eta` is `R32Float` on cells and holds the target surface for ocean edges. The compute bind group uses eight `read_write` 2D textures, which is Metal’s limit per kernel, so a new field needs to share a texture.

Domain edges: `applyDomainBoundaries` runs after pressure and integration, then again after PML face damping, so wall, source, drain, and waves edges are not overwritten by the sponge.

An ocean edge is closed at the face. Instead, `oceanBand` runs before advection and blends the water depth toward a target surface over `ocean_band_cells` cells just inside the PML, following Chentanez, Müller and Kim 2015, eq. 18. The weight is 1 next to the PML and falls linearly to 0 further in. Velocities are left to the solver. The PML relaxes toward the same target, so what passes the band is absorbed. The target is `pml_eta_rest`, or with `ocean_forcing` on, the FFT ocean's long waves, which `OceanPlugin` writes into `ocean_eta` each frame for a placed grid. On reset that write covers the whole grid, and the solver copies it into every wet cell with a wind-aligned velocity, so the interior starts in the long-wave field.

`loadPreset` builds the bed from a procedural preset, or from `bed_source` when `preset_index` is `SHALLOW_WATER_PRESET_BED_SOURCE`. That texture holds one bed height per cell in solver units, so a grid can start from a terrain. `bed_source_image` builds it from a slice of heights.

The example (`examples/shallow_water/main.rs`) adds lighting and egui controls and does not use the FFT graph; `ShallowWaterPlugin` is standalone.

## Bevy integration

Simulation WGSL lives in [`src/shallow_water/simulator.wesl`](../src/shallow_water/simulator.wesl) and is embedded in the crate, so it loads from `embedded://bevy_fft/shallow_water/simulator.wesl` in any app. [`ShallowWaterController`](../src/shallow_water/mod.rs) is extracted to the render world, and `prepare_shallow_water_gpu` keeps a `ShallowWaterSolver` built from its images, rebuilding it when the images or the particle count change. The example schedules egui on `EguiPrimaryContextPass`.

`Rg32Float` storage is unreliable on Metal, so this module uses `Rgba32Float` where multi-channel or packed storage is needed, as above.

[`ShallowWaterSurfaceMaterial`](../src/shallow_water/mod.rs) is an `ExtendedMaterial<StandardMaterial, ShallowWaterSurfaceExtension>`: the extension displaces Y from filtered bed + water height and updates normals from height samples.

Default `dt` is **0.08**. Optional PML: set `ShallowWaterController::pml_width` (for example 10), `pml_eta_rest` (target free-surface height; rest depth in the layer is `max(0, η_rest − H)` per cell), **`pml_sigma_exponent`** (**2** = quadratic σ/γ ramp per [Joh08], **3** = cubic), and **`pml_cosine_blend`** (**0** = off, **1** = scale x-strip damping by `|u|/|v|` and y-strip by `|w|/|v|` at cell centers, in the spirit of CMF10 §2.1.4 citing [Joh08]).

## Placing the grid in a larger scene

Set `ShallowWaterController::placement` to a `ShallowWaterPlacement` to put the grid in the world. The placement gives the grid centre, the cell width in meters, the datum that lifts or lowers the solver's heights to the surface mesh plane, and the width of the fade band along the edges. The cell width also becomes the solver's `dx`, so gravity, velocities, and heights are in SI units.

When `WaterPlugin` is in the app, which `OceanPlugin` adds on its own, a compute pass blends the grid into the shared water LOD data every frame. It runs after base writers such as the FFT ocean and before height queries. Inside the grid the solver replaces the long waves its cells can resolve. Shorter FFT cascades stay on top and fade out in the last half meter of water. Dry cells are drawn at the still-water level, so the mesh does not follow the bed onto land. Over the fade band the long waves hand back to the ocean. The foam texture's green channel receives the water depth for wet-region coverage. Its red channel receives breaking foam where the solver surface is steep between wet columns less than about 3 m deep. A bore resolved over one or two cells is a near-vertical step that the surface normals cannot show, so foam is what marks it. The [transmissive surface](water_rendering.md) uses rendered scene depth for optical thickness and shows the actual terrain beneath shallow water. The flow texture receives the cell velocity, plus the rate the surface rises from the divergence of depth times velocity, so `OceanQueries` report water velocity inside the grid.

Two more controller fields feed the depth update. `manning_n` adds implicit Manning friction on top of the linear `friction`, and `evaporation` removes a fraction of the depth per second. `source` takes an optional water source texture, described in the next section.

`fill_level` fills each preset to a still water level on reset. Cells whose bed rises above it start dry.

## Driving the solver from another crate

The plugin steps the grid once per frame from the controller. A crate with its own loop, such as a terrain erosion model that moves the bed every frame and runs its own sediment kernels between steps, can drive the same solver directly. Add `ShallowWaterSolverPlugin` instead of `ShallowWaterPlugin`. It registers the embedded shader and queues `ShallowWaterPipelines` in the render world, and nothing else runs.

In the render world, `ShallowWaterSolver::new` takes the device, the pipelines, the pipeline cache, and a `ShallowWaterSolverDescriptor`. It returns `None` until every kernel has compiled, so call it again on later frames until it succeeds. The descriptor names the grid size in cells, the grid textures, the read-only inputs, and a particle count, which may be zero.

The grid textures are a `ShallowWaterTextures` of texture views. `ShallowWaterTextures::create` makes zeroed textures on the device. `ShallowWaterTextures::from_images` takes the views of main-world images from `ShallowWaterImages::new`, which suits a grid that is also drawn with `ShallowWaterSurfaceMaterial` or read back. `bed_water` holds the bed H in `.x` and the depth h in `.y`. `flow_x` is `R32Float` at `(nx + 1) × ny` with the face velocity u on vertical faces, and `flow_y` is `R32Float` at `nx × (ny + 1)` with w on horizontal faces. Both are positive along +x and +y in cells per second times the cell size. A caller can bind these views in its own kernels.

`ShallowWaterInputs` holds two optional sampled textures with one texel per cell. `bed` is `R32Float` bed height. `source` is `Rgba32Float`, where `.x` is depth added per second, negative to remove water, and `.y` is a sink weight between 0 and 1 that drains that fraction of the depth each step, so 1 keeps a cell dry. The depth never goes below zero. Missing inputs read as zero. `set_inputs` swaps them later. Both live in the settings bind group, so the main kernel stays at eight storage textures.

`write_settings` uploads a `ShallowWaterSettings`: time step, cell size, gravity, linear friction, Manning n, evaporation per second, the four borders, the PML and ocean parameters, the overshoot limiter, and a step counter. Write it before the frame's command buffer is submitted. Every step in that frame uses the same values.

Each `encode_*` method sets the solver's bind groups and pipelines and dispatches into a compute pass the caller has begun, so it can sit between the caller's own dispatches.

- `encode_sync_bed` copies the bed input into `bed_water.x` and leaves the depth alone.
- `encode_step` runs one full step: MacCormack advection of the face velocities, the depth update with sources, sinks and evaporation, pressure, domain edges, wet and dry faces, friction, the PML, and the overshoot limiter.
- `encode_reconstruct_velocity` averages the faces into the cell-centered `velocity` texture.

The plugin also uses `encode_clear`, `encode_load_preset`, `encode_seed_ocean`, `encode_interact`, `encode_ocean_band`, and `encode_particles`, which are public for callers that want them.

Manning friction slows each face implicitly by `u / (1 + g Δt n² |u| / h^(4/3))`, with h the mean depth of the two cells beside the face, floored at the wet threshold. It applies after the linear `friction` factor, and a zero `manning_n` turns it off.

[`tests/shallow_water_solver_gpu.rs`](../tests/shallow_water_solver_gpu.rs) drives the core this way over a bed that changes every frame, with a source texture and no presets, and checks that the bed matches the input and that water volume is conserved. Run it with `cargo test --test shallow_water_solver_gpu -- --ignored`, since it needs a graphics adapter.

## Try it

```bash
cargo run --example shallow_water --features free_camera
cargo run --example beach --features free_camera
```

The example requests `FLOAT32_FILTERABLE` where supported so linear sampling on simulation textures is valid.
