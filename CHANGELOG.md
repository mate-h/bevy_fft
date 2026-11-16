# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `ShallowWaterNest` runs a fine shallow water grid inside a coarse one at refinement 3, coupled both ways after Debreu and Blayo. The child's edge band takes the parent's state, the parent takes the child's interior back, and a reflux on the faces between them keeps the parent's water. See [docs/shallow_water_nesting.md](docs/shallow_water_nesting.md).
- Rivers through a shallow water grid. `ShallowWaterController::border_profile` sets single faces along the grid's edges to a fixed inflow discharge, a Manning normal-depth outflow or a wall, and `ShallowWaterBorderProfile::add_river` places a river's inflow and outflow where a `RiverSpline` crosses the grid.
- `bodies::voxelize` turns a closed mesh into voxels with the hull's volume, plus surface patches with their outward normal and area. `FloatingBodyPlugin` evaluates buoyancy, ITTC 1957 friction, pressure drag and slamming on each `FloatingBody` against the water LOD data on the GPU, and reads back one force and torque per body.
- `DispersiveController::follow` keeps a placed dispersive patch on a moving point, such as a boat, by moving it whole cells at a time. The water moves with it without resampling, the strip that comes in over the leading edge starts at rest, and floating bodies' pressure heads carry over. The `wake` example's patch follows its boat.
- `DispersiveController::enable_bodies` lets floating bodies act on a placed dispersive patch: their hulls fill Gilligan's obstruction map, so waves reflect off them, and in deep mode they press the surface down, so a moving body trails a wake. `BodyFootprintTarget` registers any such grid with `FloatingBodyPlugin`.
- `DispersiveController::deep_water_depth` runs a dispersive patch in Airy-only deep mode: no bulk shallow water, one Airy pass at one depth, for open-sea wakes. There the brush presses the surface down like a hull instead of adding water. `choose_mode` switches between deep mode and the hybrid from the depth under the patch and `kelvin_wavelength`, with hysteresis. The `wake` example runs in deep mode.
- The dispersive patch absorbs waves at its edges with Tessendorf's edge trim (Gilligan §3.6), so they no longer wrap around the periodic FFT domain. `DispersiveController::edge_trim_cells` sets the band, 12 cells by default, and 0 restores the periodic patch.
- Floating bodies press shallow water down through a pressure head (Kellomäki 2014): `FloatingBodyPlugin` writes each body's displaced volume into `ShallowWaterController::pressure_head`, the solver drives the flow with the surface plus that head, and buoyancy reads it back from the flow lane of the LOD data, so bodies push water aside and make waves without sinking into their own dent.
- The `floating` example drops crates, a log and a ball onto the ocean and moves them with Rapier from the GPU water forces. `FloatingBody::compensated` corrects the forces for the frames the readback takes.
- `WaterBreakingFronts` traces the breaking fronts of a `WaterHeightfield` into lines along the crest after Thuerey et al. 2007, with ids kept while a front is tracked. The `beach` panel draws them.
- `tools/bake_topobathy.py` bakes terrain tiles and a sea-floor depth texture for any bay or river area from Esri TopoBathy, with river channels carved along splines.
- Shared water LOD data in [`water`](src/water/mod.rs): camera-centred texture array slices that the FFT ocean, shallow water grids and dispersive patches all write into, drawn by one `WaterSurfaceMaterial`.
- Ocean cascades with finite-depth dispersion, TMA spectra, swell and full 2D horizontal displacement. Several FFT sources can run in one app.
- GPU height, slope and velocity queries on the water surface with async readback through `OceanQueries`.
- `OceanClipmap` and `WaterQuadtree` water meshes. The quadtree selects tiles on the GPU and draws them indirectly.
- Shallow water grids with a world placement, ocean forcing along their edges from the FFT long waves, and a solver core that other crates can step.
- Foam carried between frames on the LOD slices, whitecaps from the cascade Jacobian, breaking foam on steep shallow water fronts, and per-pixel foam advected with Neyret's flow layers.
- Water optics over opaque scenes: depth-aware transmission, screen-space refraction and reflection, Jerlov extinction, and caustics on the bed.
- An underwater view with Snell's window, total internal reflection, multiple and single scattering in the water body, and light shafts.
- The `beach` example on Kauai LiDAR terrain drawn by bevy_terrain, and the `wake` example with a dispersive patch on the FFT ocean.

### Changed

- The water surface lights itself with a reduced copy of Bevy's PBR model. Directional lights, their shadows, the view environment map and ambient light still apply, while point, spot and area lights and reflection probes no longer light the water.
- `WaterOpticsUniform::trace_steps` sets the march steps of the screen-space traces.

### Fixed

- Deep-mode dispersive patches now carry waves shorter than 2π h / 44, 14 m at 100 m deep. Metal evaluates `tanh(k h)` through an exponential that overflowed, and the NaN zeroed those waves, so a boat's wake lost its divergent waves and cusps and narrowed to about 9 degrees. A towed pressure now trails a Kelvin wake at 19 degrees. The ocean's directional spread clamps its `tanh` argument the same way.
- The dispersive hybrid's waves no longer drift toward -x and -y or fade. Its bulk solver now keeps the flux on each cell's +x and +y faces, where the merge reads it, and the Airy step shifts the surface half a cell onto the same faces. A standing wave that used to decay to nothing within 10 s now keeps most of its amplitude.
- A still hull on a deep-mode dispersive patch no longer radiates waves without end. The step's high-wavenumber damping now acts on the surface plus the heads that press it down, so it no longer smears a hull's dent onto the open water beside it. Hulls block flux on all four faces, and the head under a hull is measured at cell centres, so passing waves no longer move it.
- The dispersive hybrid gives the same result on every run. Its free-surface diffusion and bulk height update no longer overwrite cells their neighbours are still reading.

## [0.2.0] - 2026-10-14

### Added

- Dispersive surface waves: the [`dispersive`](src/dispersive/mod.rs) module combines bulk shallow water with Airy surface waves, with docs in [docs/dispersive.md](docs/dispersive.md) and the `dispersive` example.

### Changed

- Upgrade to Bevy 0.20 and Rust 1.97.1.
- Migrate internal shaders to WESL for Bevy 0.20 and naga_oil.

## [0.1.0] - 2026-07-01

Initial crates.io release with GPU FFT, ocean, shallow water, and eWave on Bevy 0.19.
