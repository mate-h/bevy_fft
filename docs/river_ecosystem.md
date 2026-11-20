# bevy_fft and bevy_erosion: shared river geometry

bevy_erosion's `swe-water` branch places rivers on eroded terrain and carves their beds. bevy_fft
carves rivers into baked LiDAR terrain and runs them through its shallow water grids. The two
crates solve the same geometry problem twice, in different units and with different tools. This
note compares them and proposes how they grow together. Nothing here is implemented. A review
against both codebases (2026-11-20) corrected the first draft; its findings are folded in.

## What each crate has

| Piece | bevy_erosion (`swe-water`) | bevy_fft (`main`) |
|---|---|---|
| River path | `river::route`: priority flood from a source cell over the heights. It runs down the valley, fills hollows to their rim, stops at the sea, the map edge, standing water or another river (`taken`, so tributaries join) | `RiverSpline`: a hand-given polyline (the beach's from OpenStreetMap). The beach's `CHANNEL` thalweg was found by a one-off minimax flood in a scratch script |
| Path smoothing | `smooth_path`: two box passes, resampled one texel apart | Catmull-Rom in `tools/bake_topobathy.py`; a 7-sample box in the scratch script |
| Long profile | `long_profile`: a running minimum; a rise up to `breach_depth` is cut through with a steady slope, a higher one is followed and becomes a lake | None. The beach river's LiDAR water-surface plateau (zog.2.5) was found by hand and carved with a hand ramp |
| Channel size | `ChannelModel`: bankfull Q = k·A^0.8 from drainage area, W = 4.4·Q^0.5, H = 0.3·Q^0.4, widened to 3 texels at constant Manning conveyance; `set_flow` keeps Q monotone downstream | A width per river in `rivers.json`, a depth below the model's surface; the beach river is 20 m wide and runs 4 m³/s by hand |
| Drainage area | `drainage`: Barnes priority-flood fill, then MFD accumulation, with edge inlets | None |
| Cross-section and banks | `rasterize`: Shape Beds' profile, quintic bank grading, nearest sample per texel, merged with the armour band in one `Vec4` per texel | Parabola from the nearest centreline sample, only ever lowering (`carve` in the bake, `carve_channel` in the beach) |
| Bedforms | Hash-jittered steps spaced from depth and slope: step-pools above 2–4 % slope, riffle-pools above 0.1–0.4 % (`STEP_POOL_SLOPE`, `RIFFLE_POOL_SLOPE`), wider pools, point bars in bends | Periodic step-pools at Abrahams' H/L/S = 1.5 and sine riffle-pools that keep the mean slope, offset below the model surface, in the Python bake only |
| Where it runs | CPU, in-app, on GPU readbacks; the raster goes to `river_carve` in `river.wgsl` | Offline Python bake, a load-time carve in one example, a tile patcher (`carve_terrain_tiles.py`) over a 4-LOD tile pyramid with a 1-texel border |
| Checks | `inspect_river_brush` and `docs/VALIDATION.md` (H/L/S, spacing) | `check_river_banks` (leaks, low banks), `clamped_falls` (velocity clamp at falls) |
| Water | bevy_fft's `ShallowWaterSolver` at `cell_size` 1 in texel units and a rescaled time unit, from bevy_fft's `water-foundations` branch (locked at 96cf596 in `Cargo.lock`) | The same solver, plus nests, momentum-conserving advection, Ferguson grain friction, aeration, a settled-state cache (`settled.rs`), waterfall lips (`waterfalls.rs`), normal-depth outflow borders |
| Bed material | Two grain sizes with hiding, an active layer, a boulder fraction; boulders as GPU bodies that bump the solver's bed | One D84 per cell or per grid for friction, no transport |

## Can erosion's routing carve the beach?

Not as written. Three things stand in the way:
- **Heights below zero.** `route` and `drainage::fill` order their heaps by `heights.max(0.0)`,
  so all bathymetry sorts as 0, and the sea only stops a route when `sea_level > 0`. The beach's
  heights are metres about sea level, negative offshore.
- **Hydro-flattened pools.** On a flat the route follows flood order, not the centreline.
- **The plateau is not a breach case by itself.** `long_profile` is a running minimum, so on
  noisy LiDAR it makes a staircase that the step-pool test then reads as steep reaches.

So the first use on the beach is a comparison, not a replacement: route and profile the
Lumahaʻi with fixed ordering and see how close it comes to the hand `CHANNEL`.

## Where the shared code lives

bevy_erosion depends on bevy_fft. A dependency the other way would close a cycle, so shared code
goes into bevy_fft's repo or a repo of its own.

**Decision: a `river_geometry` crate as a workspace member of bevy_fft's repo.** bevy_fft is
upstream of both, so:
- bevy_fft uses it by path; bevy_erosion already pulls bevy_fft's repo by git and takes
  `river_geometry` from the same source, so both see one crate (two sources of one crate, path in
  one and git in the other, would be two distinct types without `[patch]`);
- release-plz publishes it with bevy_fft, so a git-only normal dependency never blocks crates.io;
- one push and one lock bump per API change, not two.

It moves to its own repo if a third consumer wants it without bevy_fft's repo.

**No glam in its public API.** bevy_fft's lock holds four glam versions, and a `Vec2` from the
wrong one is a different type. Public signatures take `[f32; 2]`, `[u32; 2]`, slices and the
crate's own small structs; glam stays an internal detail.

## The shared crate's API

On a regular grid of `metres_per_texel` with row-major `&[f32]` heights in metres, which may be
negative.

- **`Grid`**: size, texel size, bilinear sampling, and an optional nodata mask.
- **`route(grid, heights, source, cost, stop) -> Vec<[u32; 2]>`**:
  - total ordering over negative heights and NaN;
  - `stop` is a closure, so each caller says what ends a river (the sea or a tide level,
    standing water, another river, the edge);
  - `cost` is an optional per-texel penalty (a burned OSM line), so a route can follow a known
    channel across flats and under bridges.
- **`smooth_path`**, with the radius in metres and an output spacing parameter.
- **`long_profile(ground, breach_depth, end_level)`**, with these changes:
  - the ground is smoothed or fitted first, so noise doesn't make a staircase;
  - the profile ends at a given stage, as the beach's ends at the tide;
  - the follow branch drops its O(n²) scan.
- **`ChannelModel`**, its `discharge(area)` and `section(q, metres_per_texel)`. Floors such as
  the minimum width are in metres, set by the caller.
- **`River { path, profile, flow, seed }`** and **`rasterize(rivers, shape, bedforms, lower_only)
  -> Vec<RiverCell>`**:
  - `RiverCell` is a target bed and a bank blend; erosion keeps its armour by running its own
    merge over the same nearest-sample index, which the crate exposes;
  - `lower_only` makes the result never raise the ground, as bevy_fft's carves need.
- **`Bedforms`**: erosion's rules (jittered steps, pools, point bars), checked against Abrahams'
  1 < (H/L)/S < 2 and spacing in 1–4 widths with `inspect_river_brush`'s numbers. This replaces
  the Python `bedform_offsets`. Baked beds change, and that is accepted.
- **Windows**: route and rasterize work in a window around the river, so a 2 km mosaic at 1 m
  doesn't allocate full-grid arrays per river.
- **Determinism**: same input, same output on one platform. Tie order and the hash are exact;
  `powf` and `exp` may differ across platforms in the last bits.

**Kept out:**
- **`clamped_falls`** encodes the solver's 0.5 Δx/Δt velocity clamp, so it stays with the solver.
- **`check_river_banks`** takes a Bevy `RiverSpline` and settled grids in metres, while erosion's
  solver runs in texel units. It stays in bevy_fft. Erosion can call it after converting units.
- **Drainage** stays in bevy_erosion until bevy_fft needs it.
- **Armour, KTT steering, bed load, boulders and landslides** are erosion's.

## LiDAR input (bevy_fft's side)

Before routing, bevy_fft's carving tool prepares the tiles:
- **Nodata**: fill or mask it.
- **Bridges and culverts**: decks stand higher than any breach depth, so the route would make a
  lake behind them. Burn the OSM river line through them, or cut known decks from an OSM bridge
  layer.
- **Water-surface plateaus**: these are what the breach depth cuts. Set it from the expected
  depth, about 1–2 m for the Lumahaʻi.
- **Tides**: the profile ends at a tide level (the beach's `CHANNEL` ends at −0.70 m).
- **Tiles and LODs**: carve the finest mosaic, then rebuild each coarser LOD and the 1-texel
  borders from it, rather than carving every LOD analytically as `carve_terrain_tiles.py` does.
- **Channel size**: the beach river's 4 m³/s is a low flow, not bankfull. The channel model gives
  8.8 m × 0.52 m for it, against the hand 14–20 m. The tool takes a bankfull discharge or a width
  override per river.

## What each crate does with it

**bevy_fft**
- A Rust carving tool (an example or a `bin`) that reads the tile mosaic, prepares it as above,
  routes and profiles each river, rasterizes it, and writes the tiles and the solver bed. It
  replaces the Python carve, the beach's `CHANNEL` and `carve_terrain_tiles.py` once it matches
  them. The Python keeps the fetch and mosaic.
- `RiverSpline::from_path`, so an inflow border sits on a routed channel.
- Then a settled bake (`settled.rs`), giving the chain: erode, carve, settle.

**bevy_erosion**
- Uses the crate's route, profile, model, rasterize and bedforms. Keeps drainage, armour, KTT
  steering, bed load, boulders and landslides.
- Moves its bevy_fft dependency to `main`, pinned with `rev =`. The compile fix is one
  `..default()` in `ShallowWaterInputs { .. }` (main added `pressure_head`, `border`, `grain`). The
  behaviour change is larger:
  - `simulator.wesl` changed by about 430 lines;
  - `bed_water.z` is now scratch;
  - erosion writes the solver's textures from its own kernels (`swe_export`, `swe_import`) and
    derives its time unit from the limiter formula.

  Its tolerance-based `water_swe_gpu` tests decide whether the move holds. Worth trying once
  it's on `main`:
  - the overshoot limiter, which now only lowers wet cells (e279a13);
  - momentum-conserving advection;
  - Ferguson grain friction, fed from its bed's grain sizes as a D84 texture in texel units;
  - aeration.

## Concepts to borrow without sharing code

From bevy_erosion into bevy_fft:
- **Breach depth** as the test for a recorded water surface.
- **Boulders as bumps in the solver's bed**, for the stream scenes (zog.4): flow splitting and
  step-pools.
- **Springs at half the bankfull flow**, so placed rivers run inside their banks.
- **`set_flow`'s monotone discharge** and the `taken` confluences, for maps with tributaries.
- **The headless file runner's shape** (`--height in.f32 --grid in.json`, raw f32 out, a
  `run.json`), for the carving tool, so erode → carve → settle chains through files.
- **A frame-time cap on the water** (`MAX_WATER_FRAME`): a slow frame running more solver steps
  spirals.

From bevy_fft into bevy_erosion:
- **A finer water grid around the player.** Erosion widens narrow rivers instead. bevy_fft's
  camera-following nest (N steps of Δt/N, reflux) exists now. A grid along a river is
  `docs/stream_grid.md`'s option 1 and doesn't exist yet.
- **The bank checker** on brushed rivers, after unit conversion.
- **The settled cache and waterfall lips**, for scenes that start with water already in place.

## The water surface cost erosion measured

bevy_erosion measured bevy_fft's water surface fragment at about 46 ms a frame over its 5 km map,
the same with refraction and reflection off. The surface mesh covers the map, every pixel samples
the LOD data before the dry-pixel `discard`, and the `discard` turns off hidden-surface removal.
That is bevy_fft's to fix: skip the mesh over dry tiles (zog.2.3 bounds water tiles from LOD
heights) and keep `discard` out of the opaque path.

## Order of work

1. **Move bevy_erosion to bevy_fft `main`**, pinned by `rev`, alone, and rerun `water_swe_gpu`.
2. **Extract unchanged.** Move `route`, `smooth_path`, `long_profile`, `ChannelModel`, `River`
   and `rasterize` into `river_geometry` in texel units with the public types changed. bevy_erosion
   switches to it, and its rivers must come out bit-identical.
3. **Route the beach as a comparison.** Fix the negative-height ordering and add `stop`, `cost`
   and `end_level`. Route and profile the Lumahaʻi from its inflow and compare it with `CHANNEL`.
   Write no tiles yet.
4. **Metres and spacing parameters.** Make the per-texel floors and sample spacing parameters, so
   the crate works at 1 m and at 10 m. Erosion's rivers then agree within a stated tolerance.
5. **Bedforms**: erosion's rules with the H/L/S and spacing checks, and `lower_only`.
6. **The carving tool**: LiDAR preparation, a windowed rasterize, the tile pyramid rebuild, and a
   settled bake. Retire the Python carve, `CHANNEL` and `carve_terrain_tiles.py`.
7. **Grain sizes to friction**: erosion writes D84 for bevy_fft's solver.

## Open questions

- **bevy_fft versions**: does bevy_erosion follow `main` by `rev`, or does bevy_fft tag releases
  for it to move to?
- **Bed surface standard**: should the crate own a "bed surface" type (heights plus nodata plus
  datum), so LiDAR tiles and erosion maps meet one input?
