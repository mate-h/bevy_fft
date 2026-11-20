# Shallow water in bevy_fft

The [shallow_water module](../src/shallow_water/mod.rs) runs a **staggered shallow water** solver on the GPU in the CMF10 spirit (Chentanez and Müller): MacCormack advection on staggered face velocities, upwind mass flux for depth `h`, η = H + h for the pressure term, wet–dry reflective faces, optional depth limiting (`h_avgmax`), velocity capping, **PML absorbing strips** after §2.1.4 and appendix 3.1 (auxiliary φ and ψ, quadratic σ / γ in the layer), and optional overshoot limiting near steep η. Brush interaction and passive particles match the style of [webgpu-shallow-water](https://github.com/mate-h/webgpu-shallow-water); the numerics are SWE, not that repo’s pipe network.

Textures: `bed_water` is `Rgba32Float` per cell (`.x` bed H, `.y` water h, η = sum). `flow_x` and `flow_y` are `R32Float` grids for face u and w. `mac_u_temps` and `mac_w_temps` are `Rgba32Float` at the same sizes as `flow_x` / `flow_y` (MacCormack uses `.x` and `.y`). `pml_state` is `Rgba32Float` on cells (`.x` = φ for x-moving damping, `.y` = ψ for y-moving; `.zw` unused). `velocity` stores cell-centered `.xy` for tracers. `ocean_eta` is `R32Float` on cells and holds the target surface for ocean edges. The compute bind group uses eight `read_write` 2D textures, which is Metal’s limit per kernel, so a new field needs to share a texture.

Domain edges: `applyDomainBoundaries` runs after pressure and integration, then again after PML face damping, so wall, source, drain, and waves edges are not overwritten by the sponge.

An ocean edge is closed at the face. Instead, `oceanBand` runs before advection and blends the water depth toward a target surface over `ocean_band_cells` cells just inside the PML, following Chentanez, Müller and Kim 2015, eq. 18. The weight is 1 next to the PML and falls linearly to 0 further in. Velocities are left to the solver. The PML relaxes toward the same target, so what passes the band is absorbed. Both leave land alone: a cell whose bed is above the target surface holds no sea to absorb, so a river that runs in over a land edge keeps its water. The target is `pml_eta_rest`, or with `ocean_forcing` on, the FFT ocean's long waves, which `OceanPlugin` writes into `ocean_eta` each frame for a placed grid. On reset that write covers the whole grid, and the solver copies it into every wet cell with a wind-aligned velocity, so the interior starts in the long-wave field.

`loadPreset` builds the bed from a procedural preset, or from `bed_source` when `preset_index` is `SHALLOW_WATER_PRESET_BED_SOURCE`. That texture holds one bed height per cell in solver units, so a grid can start from a terrain. `bed_source_image` builds it from a slice of heights.

The example (`examples/shallow_water/main.rs`) adds lighting and egui controls and does not use the FFT graph; `ShallowWaterPlugin` is standalone.

## Bevy integration

Simulation WGSL lives in [`src/shallow_water/simulator.wesl`](../src/shallow_water/simulator.wesl) and is embedded in the crate, so it loads from `embedded://bevy_fft/shallow_water/simulator.wesl` in any app. [`ShallowWaterController`](../src/shallow_water/mod.rs) is extracted to the render world, and `prepare_shallow_water_gpu` keeps a `ShallowWaterSolver` built from its images, rebuilding it when the images or the particle count change. The example schedules egui on `EguiPrimaryContextPass`.

`Rg32Float` storage is unreliable on Metal, so this module uses `Rgba32Float` where multi-channel or packed storage is needed, as above.

[`ShallowWaterSurfaceMaterial`](../src/shallow_water/mod.rs) is an `ExtendedMaterial<StandardMaterial, ShallowWaterSurfaceExtension>`: the extension displaces Y from filtered bed + water height and updates normals from height samples.

Default `dt` is **0.08**. Optional PML: set `ShallowWaterController::pml_width` (for example 10), `pml_eta_rest` (target free-surface height; rest depth in the layer is `max(0, η_rest − H)` per cell), **`pml_sigma_exponent`** (**2** = quadratic σ/γ ramp per [Joh08], **3** = cubic), and **`pml_cosine_blend`** (**0** = off, **1** = scale x-strip damping by `|u|/|v|` and y-strip by `|w|/|v|` at cell centers, in the spirit of CMF10 §2.1.4 citing [Joh08]).

## Placing the grid in a larger scene

Set `ShallowWaterController::placement` to a `ShallowWaterPlacement` to put the grid in the world. The placement gives the grid centre, the cell width in meters, the datum that lifts or lowers the solver's heights to the surface mesh plane, and the width of the fade band along the edges. The cell width also becomes the solver's `dx`, so gravity, velocities, and heights are in SI units.

When `WaterPlugin` is in the app, which `OceanPlugin` adds on its own, a compute pass blends the grid into the shared water LOD data every frame. It runs after base writers such as the FFT ocean and before height queries. Inside the grid the solver replaces the cascades it can carry, and the rest stay FFT detail on top, fading out in the last half meter of water. A cascade's band runs from a sixth of its tile down to about an eighth of that, and the solver carries a wave only across eight cells or more, so the grid takes a cascade while its longest wave spans 16 cells and hands it back to the detail by 8. The ocean forcing feeds the grid only the cascades it takes, with the same split. The ocean's whitecaps stay wherever that detail does, under the grid's own breaking foam. Dry cells are drawn at the still-water level, so the mesh does not follow the bed onto land. A stream above that level can then sink under its banks wherever no mesh vertex lands inside it. `WaterHeightfieldDry::Extend` instead draws dry cells within two cells of water at that water's surface, which keeps a raised stream's surface up and lets the banks hide it ([`tests/water_heightfield_dry_gpu.rs`](../tests/water_heightfield_dry_gpu.rs)). Placed grids still use `Level` by default. Over the fade band the long waves hand back to the ocean. The foam texture's green channel receives the water depth for wet-region coverage. Its red channel receives breaking foam where the solver surface is steep between wet columns less than about 3 m deep. A bore resolved over one or two cells is a near-vertical step that the surface normals cannot show, so foam is what marks it. The [transmissive surface](water_rendering.md) uses rendered scene depth for optical thickness and shows the actual terrain beneath shallow water. The flow texture receives the cell velocity, plus the rate the surface rises from the divergence of depth times velocity, so `OceanQueries` report water velocity inside the grid.

Two more controller fields feed the depth update. `manning_n` adds implicit Manning friction on top of the linear `friction`, and `evaporation` removes a fraction of the depth per second. `source` takes an optional water source texture, described in the next section.

## Rivers

A river runs through the grid on per-face borders. `ShallowWaterController::border_profile` takes an image from `ShallowWaterBorderProfile`, which overrides single faces along the grid's edges and leaves the rest to their edge's `ShallowWaterBorder`. A face is one of these:

- `Inflow` lets in a fixed discharge per unit width, in m²/s. The depth update takes that flux exactly, whatever the depth beside the face, so a dry channel fills at the given rate. The face velocity is the discharge over the depth inside, which carries the inflow's momentum into the advection.
- `Outflow` lets water leave at Manning's normal-depth speed `h^(2/3) √S / n` for the bed slope `S` downstream, so the river holds the depth it would have in a long channel past the edge. It needs `manning_n`. With a zero slope it copies the outward speed of the face inside, a zero-gradient edge.
- `Wall` closes the face.

`ShallowWaterBorderProfile::add_river` places both ends from a `RiverSpline`: a Catmull-Rom centreline in world meters, a channel width, a discharge in m³/s and the outflow slope. Where the spline first enters the grid it spreads the discharge over the faces the channel covers, with a parabolic profile that peaks on the centreline and is wider along an edge the river crosses at a slant. Where it last leaves, those faces drain at normal depth. The spline should start and end outside the grid.

[`tests/shallow_water_river_gpu.rs`](../tests/shallow_water_river_gpu.rs) feeds a dry, 96 m channel on a 0.002 slope with 0.25 m²/s and Manning n 0.03. It checks that the flow settles within 5% of the normal depth of 0.343 m along the channel and that its volume then holds still, so what flows in flows out.

### Waterfalls

`ShallowWaterController::waterfalls` throws spray off waterfall lips into the [water particles](particles.md). Each frame a pass finds the faces that pass the CMF10 waterfall test: water flows across the face from the higher cell, the bed drops by more than `slope_cap` cell widths across it, 1.5 by default, and the lower cell isn't filled up to the higher cell's bed. Each such face throws `rate` particles per cubic meter that crosses it, from the water column at the lip and along the flow, through `WaterParticleQueue`. The spray is for the eye: the grid still carries the water down the drop. It needs `WaterParticlesPlugin` and a placement, and it runs on the controller's grid, not the nested one. [`tests/shallow_water_waterfall_gpu.rs`](../tests/shallow_water_waterfall_gpu.rs) checks that a settled river over a 3 m drop sprays within a quarter of `rate` times its discharge, and the same river without the drop doesn't spray.

The solver clamps face velocities to 0.5 Δx/Δt. Water falling a drop reaches √(2 g drop), 7.7 m/s for 3 m, so a fall on fine cells or with a long step is clipped and piles up at its foot. `clamped_falls` finds the lips in a settled grid where that happens, and the `river_banks` example prints them.

### Starting settled

A river takes minutes of simulated time to fill its channel and settle. `ShallowWaterSettled` caches a settled grid: the bed and depth of every cell and the velocity of every face, with the grid's size, cell width and centre. `capture_shallow_water_settled` reads a running grid back and hands over the cache a few frames later, and `save` and `load` write it to a file and read it back. Set `ShallowWaterController::settled` and every reset loads it over the preset and the ocean seed, when it fits the grid, before the nested grid fills from it. The beach example bakes and loads one with `BEACH_SETTLED`. [`tests/shallow_water_river_gpu.rs`](../tests/shallow_water_river_gpu.rs) checks that a dry channel started from the cache of a settled one holds normal depth from its first frame on.

### Checking the banks

Elevation models record the water surface over rivers, so a carved channel is only as good as the banks beside it. `check_river_banks` takes a river run to a steady state and read back as a `SettledGrid`, and reports two problems. A `BankLeak` is a region of wet cells outside the channel, which is half the river's width plus `RiverBankCheck::margin` from its centreline, that touches wet channel cells. A `LowBank` is a stretch where the highest bed across the bank, from half the width out to the margin, stays below the settled surface on the centreline plus `freeboard`. `along` limits both to a range of distances along the river, to leave out the mouth and the grid's edges.

The `river_banks` example runs the check on a terrain from `tools/bake_topobathy.py`. Give each river in the bake's `rivers.json` a `discharge` in m³/s. The tool writes the carved heights to `height.r32`, and the example fills a grid over the whole square from them, feeds every river with a discharge from its spline, runs it until it settles, and prints what it finds. It exits with status 1 when a bank fails.

```bash
cargo run --release --example river_banks -- <terrain> --cell 2 --seconds 600
cargo run --release --example river_banks -- --demo
```

`--demo` runs an 8 m channel in a 96 m valley with its left bank cut low over 8 m, and reports the low bank there and the floodplain it fills. A floodplain that fills from one breach meets the channel along much of its length, so a leak's range points to the low banks to look at. [`tests/shallow_water_river_gpu.rs`](../tests/shallow_water_river_gpu.rs) checks that a sound channel passes and the notched one fails at the notch.

`fill_level` fills each preset to a still water level on reset. Cells whose bed rises above it start dry.

## Moving objects

A `ShallowWaterMover` on an entity stirs the water as the entity moves through it, the way a wading character, a paddle or a dragged log does. Each frame the entity's velocity comes from how far it moved. Within `radius` of it, the grid and the nested grid pull each face velocity toward that velocity at `rate` per second, but only where the water is slower in that direction. Water already moving faster is left alone, and so is all water around an entity at rest or slower than `min_speed`, so a mover never holds the flow back. Rocks belong in the bed and floating bodies press on the water through the pressure head. The stir needs the entity's lowest point, `depth` below its position, under the water surface. Up to 16 movers stir each frame.

Floating bodies read the water velocity from the LOD data for their drag, so logs and other debris drift with the river and with the water a mover pushes. [`tests/shallow_water_mover_gpu.rs`](../tests/shallow_water_mover_gpu.rs) checks that an object wading through a pool sets the water behind it moving its way, that one standing still or passing above the water leaves it still, and that the water it pushes drags a crate it stops short of.

## Driving the solver from another crate

The plugin steps the grid once per frame from the controller. A crate with its own loop, such as a terrain erosion model that moves the bed every frame and runs its own sediment kernels between steps, can drive the same solver directly. Add `ShallowWaterSolverPlugin` instead of `ShallowWaterPlugin`. It registers the embedded shader and queues `ShallowWaterPipelines` in the render world, and nothing else runs.

In the render world, `ShallowWaterSolver::new` takes the device, the pipelines, the pipeline cache, and a `ShallowWaterSolverDescriptor`. It returns `None` until every kernel has compiled, so call it again on later frames until it succeeds. The descriptor names the grid size in cells, the grid textures, the read-only inputs, and a particle count, which may be zero.

The grid textures are a `ShallowWaterTextures` of texture views. `ShallowWaterTextures::create` makes zeroed textures on the device. `ShallowWaterTextures::from_images` takes the views of main-world images from `ShallowWaterImages::new`, which suits a grid that is also drawn with `ShallowWaterSurfaceMaterial` or read back. `bed_water` holds the bed H in `.x` and the depth h in `.y`. `flow_x` is `R32Float` at `(nx + 1) × ny` with the face velocity u on vertical faces, and `flow_y` is `R32Float` at `nx × (ny + 1)` with w on horizontal faces. Both are positive along +x and +y in cells per second times the cell size. A caller can bind these views in its own kernels.

`ShallowWaterInputs` holds optional sampled textures with one texel per cell, and the border profile. `bed` is `R32Float` bed height. `source` is `Rgba32Float`, where `.x` is depth added per second, negative to remove water, and `.y` is a sink weight between 0 and 1 that drains that fraction of the depth each step, so 1 keeps a cell dry. The depth never goes below zero. `pressure_head` is an `R32Float` head that presses the surface, and `border` the `Rgba32Float` image of a `ShallowWaterBorderProfile`, described under Rivers. Missing inputs read as zero. `set_inputs` swaps them later. They all live in the settings bind group, so the main kernel stays at eight storage textures.

`write_settings` uploads a `ShallowWaterSettings`: time step, cell size, gravity, linear friction, Manning n, evaporation per second, the four borders, the PML and ocean parameters, the overshoot limiter, and a step counter. Write it before the frame's command buffer is submitted. Every step in that frame uses the same values.

Each `encode_*` method sets the solver's bind groups and pipelines and dispatches into a compute pass the caller has begun, so it can sit between the caller's own dispatches.

- `encode_sync_bed` copies the bed input into `bed_water.x` and leaves the depth alone.
- `encode_step` runs one full step: advection of the face velocities (MacCormack, or momentum-conserving with `momentum_conserving`), the depth update with sources, sinks and evaporation, pressure, domain edges, wet and dry faces, friction, the PML, and the overshoot limiter.
- `encode_reconstruct_velocity` averages the faces into the cell-centered `velocity` texture.

The plugin also uses `encode_clear`, `encode_load_preset`, `encode_seed_ocean`, `encode_interact`, `encode_ocean_band`, and `encode_particles`, which are public for callers that want them.

By default the face velocities are advected with MacCormack's semi-Lagrangian steps, as in CMF10. That form is non-conservative, so a bore moves at the wrong speed: Stoker's dam break runs about a quarter short. `momentum_conserving`, on `ShallowWaterSettings` and `ShallowWaterController`, swaps it for Stelling and Duinmeijer's (2003) momentum-conserving advection. That form puts the bore within a cell and the depth behind it within 1%, and lets a flood front over a dry bed run further. It is two kernels instead of six, and a whole 1024² step ran 1.68 ms against MacCormack's 1.78 ms on an idle M5. It damps a free vortex somewhat faster: after 6 s it keeps 51% of its energy against MacCormack's 61%. [`tests/shallow_water_dam_break_gpu.rs`](../tests/shallow_water_dam_break_gpu.rs) checks both schemes against Stoker and Ritter, and the vortex.

The overshoot limiter blends a cell's depth toward its neighbours' where the surface steps down by more than 2 Δx, after CMF10 §2.2. It only lowers wet crests, so a dry boulder standing above the water stays dry. Lowering a crest still removes a little water, so tests that check volume turn it off.

Manning friction slows each face implicitly by `u / (1 + g Δt n² |u| / h^(4/3))`, with h the mean depth of the two cells beside the face, floored at the wet threshold. It applies after the linear `friction` factor, and a zero `manning_n` turns it off.

Before the depth update, `limitOutflow` scales each cell's outgoing fluxes so one step never drains more than the cell holds, so the depth stays non-negative without a clamp that creates water. The height kernels stage their result in `pml_state.z` and `commitHeight` copies it into `bed_water.y`, so every cell reads its neighbours' depth from before the kernel. By default the depth moves through each face first-order upwind, which diffuses about |u| Δx / 2. `second_order_depth` adds a flux-limited Lax-Wendroff correction with van Leer's limiter (Sweby 1984), which stays TVD up to Courant number 1 and lets roll waves grow at lower Froude numbers.

[`tests/shallow_water_solver_gpu.rs`](../tests/shallow_water_solver_gpu.rs) drives the core this way over a bed that changes every frame, with a source texture and no presets, and checks that the bed matches the input and that water volume is conserved. Run it with `cargo test --test shallow_water_solver_gpu -- --ignored`, since it needs a graphics adapter.

## Nesting

A fine grid can run inside a coarse one, coupled both ways at refinement 3. See [Nested shallow water grids](shallow_water_nesting.md).

## Try it

```bash
cargo run --example shallow_water --features free_camera
cargo run --example beach --features free_camera
```

The example requests `FLOAT32_FILTERABLE` where supported so linear sampling on simulation textures is valid.
