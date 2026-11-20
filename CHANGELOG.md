# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- The shallow water plugin nests grids more than one level deep. `ShallowWaterController::nested` is now a list, each grid nested in the one before it at refinement 3, and `nest_levels` gives where each sits and how it steps. Each grid follows its own point, takes its bed from its world bed or its parent, blends into the LOD data one `order` above its parent, and gets body footprints and mover stirs. `ShallowWaterNestedGrid::steps` set to `None` picks a grid's steps per parent step from its stability limit for `max_speed`, `max_depth` and `fall_speed`, and `momentum_conserving` defaults on for cells under 0.5 m. `ShallowWaterNest::encode_frame_with` chains nests, with the face volumes a nest's child adds up for its parent's reflux now in `.w`. The `beach` example adds a 2/9 m stream level around the camera with `BEACH_STREAM`.
- `ShallowWaterNestDescriptor::child_steps` sets how many steps a nested grid takes per parent step, instead of always three, so a fine grid can step as its own stability limit allows. The parent still holds its water to 1e-7 with 1, 2, 3 and 5.
- `tools/bake_topobathy.py` shapes a carved river's bed with step-pools or riffle-pools when its `rivers.json` entry has `"bedforms"`. Step-pools follow Abrahams, Li and Atkinson (1995): step height over spacing is 1.5 times the local slope, with a pool scoured below each step. The mean slope of the bed stays the same.
- `WaterHeightfieldDry::Extend` draws dry cells within two cells of water at that water's surface, and the rest at a still level. A stream above the still level then keeps its surface out into its banks, which hide it, instead of the mesh sinking to the still level wherever no vertex lands inside the stream. Placed shallow water grids use it.
- `clamped_falls` finds waterfall lips in a settled shallow water grid where free fall outruns the solver's velocity clamp of 0.5 Δx/Δt, and the `river_banks` example reports them.
- `ShallowWaterSettings::momentum_conserving` and `ShallowWaterController::momentum_conserving` advect shallow water velocity in momentum-conserving form, after Stelling and Duinmeijer (2003), instead of MacCormack's. A dam-break bore then lands within a cell of Stoker's exact solution instead of a quarter short. Off by default.
- `FloatingBody::footprint_scale` multiplies the volume a body presses the water down with. A planing hull, carried mostly by dynamic lift, can then press its wake with its whole weight. The `boat` example's wake waves at 15 knots grew from 5 to 17 cm with it.
- The `boat` example's chase camera has a wake view, toggled with `C` or started with `BOAT_VIEW=wake`: it looks down from above the starboard quarter at the whole boat and its wake.
- The `boat` example: a motor boat to drive across the FFT ocean with the keyboard.
  - The hull is lofted and voxelized in code, its water forces come from the GPU, and Rapier moves it.
  - A propeller, a rudder and a skeg drive it, as lifting surfaces in the local flow and the propeller wash.
  - A deep-mode dispersive patch with body maps follows it, so it leaves a Kelvin wake and a trail of foam.
  - The hull needed added mass in heave, pitch and roll, and a stand-in for wave radiation damping, to ride a sea. Headless runs wait for the GPU each frame, so the forces stay two frames behind instead of 24.
- Floating bodies press on a nested shallow water grid too: `ShallowWaterNestedGrid::pressure_head` gets their footprints while the parent has a pressure head. Body footprints now spread each voxel over the cells it overlaps, so grids finer than the voxels get an even dent instead of stripes.
- `WaterBreakerCurl` draws the overhanging lip of plunging breakers: one animated cross-section, after Malan (2022), swept along each traced breaking front and played by the front's age. It is only the lip, grown from the heightfield's crest and faded out once it falls, and drawn as dark, glossy, see-through water with foam at the tip. The beach draws it.
- The foam lane's alpha holds the mean square slope of the ocean waves each LOD slice is too coarse for, from each cascade's slope variance (`OceanSlopeVariance`, `slope_variance`), and the water surface adds it to its roughness, so distant glitter widens as it should.
- Flow detail normals: short ripples in the foam texture's green and blue channels, read through the foam's advected texture coordinates, roughen flowing water by drift speed and strain. `WaterSurfaceUniform::flow_detail` sets their slope, and the surface material gains a `lod_drift` binding.
- `ShallowWaterWaterfalls` throws water particles off waterfall lips in a shallow water grid, where the CMF10 test finds water pouring over a bed drop of more than 1.5 cell widths, in proportion to the water crossing the lip.
- `ShallowWaterSettled` caches a settled grid, `capture_shallow_water_settled` takes one from a running grid, and `ShallowWaterController::settled` starts every reset from it, so a river starts settled instead of filling its channel. The beach bakes and loads one with `BEACH_SETTLED` and `BEACH_SETTLED_AFTER`.
- `ShallowWaterMover` stirs shallow water with an entity's motion, only while it moves: faces within its radius on the grid and its nested grid are pulled toward its velocity where the water is slower in that direction. Floating bodies drift with the water it pushes.
- The `beach` example feeds the Lumahaʻi River in over the map's south edge along its OpenStreetMap centreline, 4 m³/s by default (`BEACH_RIVER`).
- [docs/particles.md](docs/particles.md) describes the water particles.
- Water particle budget: crest spray thins with distance from the viewer while its drops grow to keep the same coverage, `WaterParticleSettings::with_quality` sets the pool and spray budgets for `WaterParticleQuality`, `WaterCrestSpray::debug_potential` marks where spray spawns, and `WaterParticleStats` reads the pool's counters back when `read_counters` is on.
- Water particles look like spray: drops scatter the sun forward with a Henyey-Greenstein phase function and take the sky's light, smear along their motion, keep at least 1.5 pixels on screen with their opacity cut by the area they lack, and fade into the depth prepass behind them. `WaterParticleSettings` sets each.
- Floating bodies splash into the water particles: surface at a body's waterline that moves into the water faster than `FloatingBodySplash::min_speed` throws up particles from the force pass on the GPU, through `WaterParticleQueue`, a GPU spawn queue other passes can append to. The `floating` example splashes.
- `WaterCrestSpray` emits water particles from breaking crests on the GPU, from the breaking intensity the ocean, shallow water and dispersive patches write into the LOD foam lane, in proportion to texel area and how fast the surface rises. Nested slices emit each patch once. The `beach` and `ocean` examples have a Spray panel.
- Water particles land on the water surface in the LOD data and leave active foam where they hit, through `WaterFoamDeposit`, an atomic per-texel buffer the foam sim adds that same frame. Other passes can deposit foam there in `WaterFoamDepositSystems`.
- `WaterParticlesPlugin` keeps a GPU pool of water particles in world space: spawns from `WaterParticleSpawns` or a `WaterParticleEmitter` land at an atomic cursor in a ring, a compute pass moves them under gravity and quadratic air drag against the wind, and they are drawn as soft camera-facing dots. The `pool` example has a fountain.
- The `pool` example drops a crate into a shallow water channel with a river running through it, tows a log across it and lays one across it as a dam, with Rapier moving the bodies and their pressure heads pressing the water.
- `check_river_banks` flags water that left a settled river's channel and banks lower than its surface, and the `river_banks` example runs it on a baked terrain. `tools/bake_topobathy.py` also writes the carved heights to `height.r32`, and rivers take an optional discharge.
- `ShallowWaterController::nested` runs a `ShallowWaterNestedGrid` with a third of the cell width inside the plugin's grid, following a point such as the camera and drawn over its parent in the water LOD data. `WaterHeightfield::order` sets which heightfield draws over which. The `beach` example nests one around the camera.
- `ShallowWaterShift` moves a shallow water grid over the world by whole cells, filling the cells that come in from a world bed texture and from a parent grid's surface, so a nest can follow the camera. Its sampling keeps a parent's bed the mean of its child's.
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

- FFT lines up to 512 long run all their radix-2 stages in workgroup memory in one dispatch, `fft_dit_line`, instead of one dispatch and a round trip through the buffers per stage. Results are unchanged; longer lines keep the per-stage chain.
- Faster GPU passes with unchanged results: dispersive grid indices wrap without integer division, floating body water lookups share one LOD footprint per point, and the dispersive LOD write reads each stencil point once.
- `WaterBreakingFronts` keeps front ids steadier: lines match when a share of either line lies near the other, and a line missing from a readback is carried forward for `keep_lost` seconds, 0.5 by default. In the beach this cuts new front ids by 61% on the grid and 83% on the nested grid, and lost fronts had lived 2.5 s on average instead of 0.5 s. `WaterBreakingFronts::tracking` counts how lines were matched, and `BEACH_FRONTS_LOG` logs it.
- The water surface lights itself with a reduced copy of Bevy's PBR model. Directional lights, their shadows, the view environment map and ambient light still apply, while point, spot and area lights and reflection probes no longer light the water.
- `WaterOpticsUniform::trace_steps` sets the march steps of the screen-space traces.
- Water particles draw only the live particles, with an indirect draw from a list the simulation fills in `WaterParticles::draw`, instead of six vertices for every pool slot.

### Fixed

- The shallow water overshoot limiter no longer pours water onto dry cells. CMF10's test reads only surface heights, so a dry cell more than 2 Δx above wet water, such as a boulder, passed it, and the blend added an eighth of the wet neighbour's depth every step. The limiter now only lowers wet crests.
- A free floating body that presses shallow water settles instead of bobbing without end: `FloatingBodySettings::pressure_relaxation` defaults to 0.05 instead of 0.25, so the head follows the hull over about 20 frames instead of counting its heave twice. The `pool` example's crate presses the water all the time again.
- Baked terrain no longer rises to 0 m in a thin line around the map: `tools/pad_terrain_borders.py` replaces the edge tiles' outer texels, which the bevy_terrain processor fills with 0 m, with the data inside them, and both bake scripts run it. The `beach` example's water bed reads the whole map again.
- A shallow water grid's ocean band and PML no longer drain land along an ocean border, so a river that runs in over a land edge keeps its water.
- Shallow water grids keep the ocean's short waves and whitecaps. The split between the grid and the FFT detail counted a cascade as resolved by its longest wave, so a 2 m grid was fed the 48 m cascade's 1.5 to 8 m waves, damped them away, and drew no detail in their place. Grids now take a cascade only while its longest wave spans 16 cells. The `beach` example's water bed no longer reads the terrain's last texel rows, which the terrain processor fills toward 0 m, raising a ridge to sea level along the map edge.
- Water coming in over a shallow water grid's edge, from a river inflow or a source border, is no longer scaled down by the outflow limit of the cell it enters.
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
