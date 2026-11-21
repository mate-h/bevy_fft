# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0](https://github.com/mate-h/bevy_fft/compare/bevy_fft-v0.2.0...bevy_fft-v0.3.0) - 2026-11-21

### Other

- Lower the boat's wake view so the Kelvin arms show
- Break the dispersive patch's crests at deep-water steepness
- Give the ocean LOD writer a half-float copy of each cascade's displacement
- Store the ocean's cascade foam textures as half floats
- Wait for ten pipeline-free frames before a dispersive bodies test measures
- Put the sky pass aside while the water viewer is under water
- Say how to size a WaterQuadtree over a heightfield
- Add WaterQuadtree::extent to keep the mesh within a map
- Make quadtree dry culling opt-in and test it on a max pyramid
- Drop quadtree water tiles with no water under them
- Add a glint view to the boat example
- Keep the water surface out of the sun's shadow cascades
- Keep two frames in the foam state and flow arrays instead of copying them
- Build the ocean's Jacobian mips five levels per dispatch
- Write the ocean's per-frame displacement change in the foam pass
- Format the Linux headless plugin setup
- Discard dry water pixels before the foam and air textures
- Keep the ocean foam bind groups across frames
- Log per-pass render diagnostics in example runs with BEVY_FFT_LOG_DIAGNOSTICS
- Run headless screenshot runs without winit on Linux
- Grade the beach river's banks over 10 m
- Carve rivers into terrain tiles with river_geometry
- Route the beach river with river_geometry and compare it with the hand channel
- Add the river_geometry crate, moved from bevy_erosion's river brush
- Plan shared river geometry with bevy_erosion
- Add a tool that carves the beach river's channel into the baked terrain tiles
- Carve the beach river through the reach where the LiDAR holds its water surface
- Merge branch 'worktree-agent-a95f4e83f9410ede5'
- Modulate the short waves on a dispersive patch by its waves' strain
- Add Ferguson's variable-power bed friction from a D84 grain size
- Merge branch 'worktree-agent-a2cd5b164d26dd9a6'
- Give a moving hull's wake the height of a pressure patch of its weight
- Apply slope-variance roughness and flow detail in the deferred water path
- Test that a dispersive wake reaches every LOD slice over its patch
- Soften and texture white water
- Add an aeration field for white water
- Clear the root shallow water grid's face volumes for its parent each frame
- Nest shallow water grids more than one level deep
- Let a nested shallow water grid take any number of steps per parent step
- Decide on 2D grids for narrow streams
- Draw placed shallow water grids' dry cells beside water at its surface
- Carve step-pools and riffle-pools into baked river beds
- Add a dry mode that keeps a raised stream's surface out into its banks
- Time momentum-conserving advection against MacCormack
- Flag waterfall lips whose free fall outruns the velocity clamp
- Add momentum-conserving advection to the shallow water solver
- Test dam-break bores and dry fronts against Stoker and Ritter
- Add a wake view to the boat example's chase camera
- Wait for every pipeline to compile before the dispersive body tests start
- Keep the shallow water overshoot limiter from wetting dry cells
- Add the river solver, river meshing and river realism research spikes
- Test the GPU FFT against a CPU DFT and its own inverse
- Run every FFT stage of a line in workgroup memory
- Read each point of the dispersive LOD write's stencil once
- Stop refiltering the boat example's sky lighting once it settles
- Share one LOD footprint between the reads of a water lookup
- Wrap dispersive grid indices without an integer division
- Press the boat's wake with its whole weight on the plane
- Speed the boat up to 15 knots and raise the chase camera over its wake
- Test the boat's speed, course keeping and turning on the GPU
- Share the boat's physics between the example and its tests
- Trail foam behind the boat and spray from its hull; document the boat example
- Show the boat's speed, helm and heading, with a tuning panel
- Drive the boat with a propeller, a rudder and a skeg
- Trail a wake patch behind the boat
- Add a tow knob to the boat example for measuring hull drag
- Add the boat example: a floating hull with a chase camera
- Add a procedural boat hull for the boat demo
- Plan a driveable boat demo
- Draw the breaker curl as the lip of the wave, in water colours
- Take turns in the dispersive GPU tests, and drop a flaky one
- Stop the shared LOD helpers spilling registers
- Stop the foam kernels spilling registers
- Throw mist at the foot of waterfalls
- Record the nested grid's cost in the beach
- Let floating bodies press on the nested shallow water grid
- Add underwater repro knobs to the ocean example
- Add the breaker curl along breaking fronts
- Steady the breaking front ids
- Add the unresolved slope variance lane and widen far glitter
- Add advected flow detail normals
- Spray water particles off waterfall lips
- Start rivers from a settled cache
- Format the particle imports
- Stir shallow water with moving objects
- Draw only live water particles
- Cost out Water Surface Wavelets for long-lived wakes
- Let free floating bodies settle on shallow water
- Pad baked terrain tiles at the map's edge
- Feed the Lumahai River into the beach
- Test that ambient swell rests against a hull on the wake patch
- Document the water particles
- Budget the water particles
- Make the water particles look like spray
- Splash floating bodies into the water particles
- Emit spray from breaking crests on the GPU
- Land water particles on the surface and deposit foam
- Add a GPU pool of water particles
- Show bodies in shallow water in a pool example
- Keep the ocean's short waves and whitecaps on shallow water grids
- Skip LOD texture reads that never reach the output
- Stop scaling inflow by the edge cell's outflow limit
- Check a settled river's banks
- Nest a camera-following grid in the plugin and the beach
- Move a nested shallow water grid over world bathymetry
- Nest a fine shallow water grid inside a coarse one
- Let a river run through a shallow water grid
- Keep short waves alive in deep mode, and check the Kelvin angle
- Let a dispersive patch follow a moving point by whole cells
- Stop a still hull radiating waves on a deep-mode patch
- Put the dispersive hybrid's fluxes on the faces the merge reads
- Let floating bodies obstruct and press a dispersive patch
- Run the wake in deep mode with a hull-like brush
- Add an Airy-only deep mode to the dispersive patch
- Trim the dispersive patch's edges
- Couple floating bodies into shallow water
- Float bodies on the ocean with Rapier
- Evaluate water forces on floating bodies on the GPU
- Voxelize floating bodies on the CPU
- Trace breaking fronts into tracked lines
- Restructure the README
- Add a TopoBathy terrain bake tool
- Build the docs without warnings
- Describe the water work in the changelog and the beach switches in the README
- Let BEACH_CAMERA place the beach camera
- Pass fmt and clippy on every target
- Run the beach grid with 2 m cells by default
- Make the water traces' march step count a setting
- Pass the water optics only the surface fields they read
- Let the beach grid run with coarser cells
- Pick the LOD slices once per water pixel
- Light the water surface without Bevy's full PBR lighting
- Add beads issue tracking for Claude
- Dispatch the shallow water forcing over its edge bands only
- Stage each cell's movable depth once per shallow water step
- Skip the foam pattern where a pixel quad has no foam
- Let any solver's heightfield blend into the water LOD data
- Carry depth with an optional second-order flux, and limit each cell's outflow
- Take out shallow water changes that were committed by mistake
- Measure light under water from the mesh plane in both the underside and the medium pass
- Darken a meniscus line where the near plane cuts the water
- Draw foam seen from below as a diffusing layer that passes the sky it does not reflect
- Draw the sun through Snell's window with Walter's rough refraction
- Trace caustics onto several receiver depths and blend the two around each point
- Split the shallow water solver into a core other crates can step
- Dim sunlight along its refracted beam on the way down to beds seen from either side
- Take Jerlov absorption and scattering from Solonenko and Mobley's tables
- Break near-range waves on the Jacobian determinant like the far range
- Keep ocean query results paired when the GPU runs dozens of frames behind
- Light beds seen from above water with the caustic map
- Advect foam texture coordinates per texel with Neyret's deformation-weighted layers
- Draw light shafts under focusing waves from the caustic map
- Light scene points under water with the caustic map
- Add a caustic map that refracts sun rays through the surface onto a receiver
- Add shadowed single scattering of the refracted sun under water
- Fill the view from under water with Monzon et al.'s multiple-scattering medium
- Draw the water surface from below with Snell's window and total internal reflection
- Add a waterline mask and eye height for the water viewer
- Add far-range whitecaps from Dupuy and Bruneton's erf coverage
- Keep foam off where the sim has none
- Carry the foam texture with the drift in two flow phases
- Render foam per pixel by thresholding a procedural foam texture
- Break ocean waves on the summed cascade Jacobian and calibrate whitecaps against Monahan and Woolf
- Use field init shorthand for the hull speed
- Feed the foam sim with breaking rates from every solver
- Carry foam between frames on the water LOD slices
- Add a drift array to the water LOD data
- Give the ocean cascades full 2D horizontal displacement
- Take water extinction from Jerlov water types
- Fade ocean whitecaps with a half-life in seconds and start them from the smaller Jacobian eigenvalue
- Light the beach ground in bevy_terrain's deferred lighting pass
- Mesh the beach ground with coarser terrain tiles
- Filter the beach sky light once and keep it
- Share wrapped bilinear taps across each ocean cascade's textures
- Estimate the replaced water reflection from one environment map sample
- Shorten the water's screen-space traces to 12 march and 5 bisection steps
- Format the beach example imports
- Drop screen-space reflections from the beach camera
- Update bevy_terrain for the faster terrain shadow map
- Drop the unused height-mask discard so Apple GPUs keep hidden surface removal on the coast
- March water rays in clip space and light the surface once
- Skip near-zero triplanar projections in the coastal ground material
- Mark steep shallow water fronts with depth-limited breaking foam
- Warp coastal class lookups so label edges render as irregular outlines
- Render coastal water over opaque terrain with depth-aware transmission
- Shade the Kauai coast with classified PBR materials and persistent sand wetness
- Select water quadtree tiles on the GPU and draw them indirectly
- Enable beach SSR and terrain self shadows
- Stabilize water LOD rendering and seed shallow water from ocean waves
- Keep the water visible when switching between clipmap and quadtree
- Take example screenshots without opening a window
- Make water surface LOD reads about 5x cheaper
- Cut beach GPU time: edge-only ocean forcing and fewer water fetches
- Capture example screenshots from an offscreen image
- Simulate the whole Kauai map and fade water past the LOD data
- Put the beach example on Kauai LiDAR terrain drawn by bevy_terrain
- Force the shallow water deep edge from the FFT ocean's long waves
- Add a quadtree water mesh with distance LOD and frustum culling
- Add a flow channel to the water LOD data and move the surface material
- Write shallow water and dispersive patches into the water LOD data
- Add shared water LOD data and render the ocean through it
- Add camera-following ocean clipmap and fix query height lane
- Add GPU ocean height queries with async readback
- Support several FFT sources and add ocean cascades
- Add finite-depth dispersion, TMA and swell to the ocean spectrum

### Added

- `WaterQuadtree::extent` limits the quadtree to a world XZ rectangle, such as a heightfield's map, in both the GPU and the CPU selector. Without it, a heightfield-only scene draws open water past its LOD slices out to the horizon.

- The `carve_rivers` example carves rivers into a bevy_terrain height tile set with `river_geometry`: it routes each river in a JSON file over the uncarved tiles, cuts its long profile through recorded water surfaces, carves a channel sized for its bankfull discharge, and writes the change into every LOD. The beach's `CHANNEL` and `tools/carve_terrain_tiles.py` are gone; run it once on the Kauai tiles with `examples/beach/rivers.json`.
- `river_geometry`, a workspace crate shared with bevy_erosion: river routing by priority flood, path smoothing, breached long profiles, the bankfull channel model and the bed raster with step-pools, riffle-pools and point bars, moved unchanged from bevy_erosion's river brush. See `docs/river_ecosystem.md`.
- Ferguson's (2007) variable-power bed friction from a grain size D84, set for the whole grid (`ShallowWaterController::grain_d84`) or per cell (`ShallowWaterController::grain`). It replaces Manning's where set, so shallow water over cobbles drags as hard as measured.
- A dispersive patch in add mode modulates the short waves riding on it, as a long wave's strain does: the unresolved slope variance and the resolved ocean slopes shorter than the patch's waves are scaled by `exp(m |k| η)` and its square root, rougher on the patch's crests and smoother in its troughs, in both the forward and deferred water paths. `DispersivePlacement::short_wave_modulation` sets m, 4.5 by default after Alpers and Hasselmann (1978), and the `boat` example takes it from `BOAT_MODULATION`. The wake's height is unchanged.
- White water: the water column carries an air fraction that makes rapids, hydraulic jumps and the feet of waterfalls white and opaque. Heightfield writers store the air the flow can hold in the new `WaterLodData::aeration` array, from jumps and bores by their velocity drop and Bélanger's conjugate depths, from steep fast flow, and from waterfall lips and landings. The foam sim carries it in the blue lane of `foam_state`, raises it over `WaterAerationSettings::formation_time` and lets it leave the column at `detrainment_speed`, calibrated against the step-pool jumps of Vallé and Pasternack (2006). The surface scatters light back from it and hides the bed under it. `foam_state` is now `Rgba32Float`.

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

- `WaterQuadtree::dry_culling` (off by default) drops tiles with no water under them in GPU selection when the material sets `dry_depth`. Each frame a small pass reduces every LOD slice's water depth to a max pyramid over 8x8 texel blocks, and refinement skips a node whose cells, widened by `horizontal_margin`, are all shallower than `dry_depth`, since the surface would discard all of its pixels. Tiles beyond the LOD slices stay, as open water. On an RTX 5090 over bevy_erosion's mostly dry map it cut the water pass from 0.067 to 0.044 ms but added 0.035 ms to selection; it pays where water pixels cost more.

- `OceanClipmap` and `WaterQuadtree` require `NotShadowCaster`, so the water surface no longer draws itself into the sun's shadow cascades. Water lets most direct sunlight through; the surface still receives shadows. On an RTX 5090 the boat's `per_view_shadow_pass` fell from 0.164 to 0.065 ms with no visible change.

- `WaterLodData::foam_state` and `foam_flow` hold two frames, and the foam sim reads last frame's half while it writes this frame's, instead of copying both arrays every frame (the boat's `water_foam` fell from 0.178 to 0.073 ms on an RTX 5090). `WaterLodSlices::foam_half` (formerly padding) says which half is this frame's: its slices start at layer `foam_half * count` of `foam_state` and `foam_half * count * WATER_FOAM_FLOW_LAYERS_PER_SLICE` of `foam_flow`. Alpha of `foam_state` holds the sim's time when it wrote the texel, so readbacks can tell the newer half.

- The ocean builds each cascade's Jacobian mip chain five levels per dispatch, averaging in workgroup memory, and runs the foam and mip stages of all cascades side by side. On an RTX 5090 the boat's `ocean_foam` fell from 0.139 to 0.060 ms; far whitecap coverage is unchanged to three figures.

- The ocean's foam pass stores each cascade's change of displacement X, Z and height since the previous frame in the foam texture's R, B and A lanes, keeping the raw values in two new ping-pong textures, `OceanFoamMask::carry_a` and `carry_b`. The LOD writer reads its flow from those lanes and no longer keeps a copy of every cascade's output. On an RTX 5090 at 2560x1440 the boat's `ocean_lod_write` fell from 0.090 to 0.057 ms, and the flow is unchanged (new GPU test `ocean_flow_gpu`).
- The cascade foam textures (`OceanFoamMask::texture_a` and `texture_b`) are `Rgba16Float`, `OCEAN_FOAM_FORMAT`, instead of `Rgba32Float`. The LOD writer is bound by its texture reads, so halving one of its two inputs cut the boat's `ocean_lod_write` on the M5 at 2560x1440 from 1.90-2.12 to 1.43-1.54 ms (two alternated xctrace rounds). The flow still matches the change of displacement within about 1% RMS, and the boat renders the same apart from small shifts in the foam pattern.
- The foam pass also writes a half-float copy of each cascade's FFT spatial output, `OceanFoamMask::displacement`, and the LOD writer reads that copy instead of the `Rgba32Float` output. On the M5 at 2560x1440 the boat's `ocean_lod_write` fell from 1.42-1.60 to 1.00-1.03 ms (two alternated xctrace rounds), and its read bandwidth halved again. The ocean and boat GPU tests pass unchanged.
- The dispersive patch marks breaking crests with the CMF10 test at the slope band of a breaking deep-water wave, 0.35 to 0.55 (`breaking_crest`), instead of the bore band of the shallow solvers, 0.12 to 0.3. The diverging waves of the boat's wake no longer break along their whole length, so the foam behind a hull is its turbulent trail and the Kelvin arms around it stay clear.
- The boat example's wake view sits low, about 20 degrees above the water it looks at, where the Kelvin arms show in the sky's reflection; from the old height the water reflected a nearly uniform sky.

- White water looks softer and less like decals. The surface treats its air as bubbles of 4 mm Sauter radius instead of 2 mm, so a few percent of air leaves the water green-white and translucent and only dense white water hides the bed. The surface spreads each texel's air into packets and streaks from the foam texture, read through the foam's advected texture coordinates and averaged along the drift, so the pattern moves with the water. Air that leaves the column surfaces as residual foam, `WaterAerationSettings::foam_yield` per meter of air, which trails downstream after the air is gone. The underside of aerated water takes the look of foam, and the underwater medium pass clouds the view where the eye looks through air. The medium pass binds `foam_state` at binding 8.
- FFT lines up to 512 long run all their radix-2 stages in workgroup memory in one dispatch, `fft_dit_line`, instead of one dispatch and a round trip through the buffers per stage. Results are unchanged; longer lines keep the per-stage chain.
- Faster GPU passes with unchanged results: dispersive grid indices wrap without integer division, floating body water lookups share one LOD footprint per point, and the dispersive LOD write reads each stencil point once.
- `WaterBreakingFronts` keeps front ids steadier: lines match when a share of either line lies near the other, and a line missing from a readback is carried forward for `keep_lost` seconds, 0.5 by default. In the beach this cuts new front ids by 61% on the grid and 83% on the nested grid, and lost fronts had lived 2.5 s on average instead of 0.5 s. `WaterBreakingFronts::tracking` counts how lines were matched, and `BEACH_FRONTS_LOG` logs it.
- The water surface lights itself with a reduced copy of Bevy's PBR model. Directional lights, their shadows, the view environment map and ambient light still apply, while point, spot and area lights and reflection probes no longer light the water.
- `WaterOpticsUniform::trace_steps` sets the march steps of the screen-space traces.
- Water particles draw only the live particles, with an indirect draw from a list the simulation fills in `WaterParticles::draw`, instead of six vertices for every pool slot.

### Fixed

- Under water, the viewer's sky pass no longer lays the air's aerial perspective over the water's underside. It drew a bright line just above the underwater horizon, at the far edge of the surface mesh, and lifted the whole underwater image by about 30%. A `WaterUnderwater` camera's `AtmosphereSettings` is put aside while its eye is more than a meter under water and restored within half a meter of the surface; the environment map, and with it the sky through Snell's window, stays.

- Water drawn deferred, as in the `boat`, `ocean`, `wake` and `floating` examples, now gets the unresolved slope variance in its roughness and the flow detail in its normal. The deferred prepass left both out, so distant glitter stayed thin and flowing water stayed smooth there. The G-buffer now carries them to the deferred lighting, which matches the forward pass. The `ocean` example starts its screen-space reflections at a perceptual roughness of 0.45, so the roughened far water doesn't pick up their speckle.
- A moving hull's Kelvin wake has the height linear theory gives for its weight. The boat example's wake at 15 knots was 0.21 m crest to trough 10 m abeam and is now 0.40 m, where a pressure patch of her 4.5 t makes 0.34 to 0.46 m in deep water. Three things held it down. Gilligan's obstruction pinned the water under the hull to rest and stopped the flux there, against the pressure that makes the wake; it now treats a hull as a wall only while most of the flux of its own transverse waves, e^{-g d / U²} at draft d and speed U through the water, would be stopped by its keel, so a hull at rest still reflects waves and a fast one lets them pass under it. The pressure head's memory stayed put in the world while the hull moved on, so the head took 2 m of travel to build under the bow at 7 m/s and lingered as far behind the transom; the memory now moves with the hull, and the head is zero where no hull is. And bodies now read the water under their own hull without the patch's response to their pressure, as the obstruction had made them do, since the voxel forces can't ride their own wave field and the boat's drag is calibrated without it. The footprint buffers gain lanes for the unscaled volume and the hull's velocity, and the head memory is double-buffered.
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
