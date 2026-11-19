# Meshing rivers: a research spike on concurrent binary trees

The integrated water design kept concurrent binary trees (Benyoub and Dupuy 2024) as a research spike for one case: adaptively meshing an irregular river or lake outline from a polygon base mesh. This note checks what the method does, how river water is meshed today, where that fails, and whether CBT is the fix. Nothing here is implemented.

## What the papers do

A concurrent binary tree (Dupuy 2020) is a binary heap holding a sum-reduction tree over a bitfield. Each set bit is a leaf, a thread finds the i-th leaf in O(D) steps, and leaves split or merge by setting bits. Dupuy bisects a square terrain to depth 27. On an RTX 2080 the sum reduction takes 1.484 ms of a 1.54 ms update.

Benyoub and Dupuy (2024) make two changes. First, each halfedge of any polygon mesh becomes a root triangle, a bisector, from its two vertices to its face's centroid. Bisectors refine by newest-vertex bisection along a compatibility chain, so the mesh stays conforming, and merge in fours, or in twos at a border. Second, the tree stops encoding the subdivision and manages a pool of 2^D bisectors instead, each with three neighbour pointers, a 64-bit index, a command word and four reserved slots. At D = 17 that is 128k bisectors in 7 MiB, up to 64 levels deep. An update is 9 kernels, each bisector changes one level at most, and triangles aim for about 49 px of screen area.

On an AMD 6800 XT the update takes 0.1 ms for an Earth-sized ocean (a dodecahedron, 60 halfedges), 0.084 ms for the Moon and 0.085 ms for a 21,399-halfedge asset at D = 18. The ocean shot draws 64k triangles. Planet positions need doubles. Neither paper describes a geomorph, so triangles pop when they change level.

## How river water is drawn today

All water, river included, is one mesh: `WaterQuadtree` in `src/water/quadtree.rs`, Strugar's CDLOD with GPU selection in `src/water/quadtree_gpu.rs`. Tiles have 8 × 8 cells and leaves are 2 m, so vertices are 0.25 m apart nearest the camera. At distance d the spacing lies between d/32 and d/16, which is 40 to 80 px per edge at 2560 px and a 90° field of view. Vertices morph into the coarser level, so nothing pops.

The vertex shader samples the `WaterLodData` slices (`src/water/lod_data.rs`): 10 slices of 256² texels, from 3.1 cm to 16 m per texel. A shallow water grid reaches the mesh only through these slices, as a `WaterHeightfield` blended by `src/water/heightfield.wesl`. The plugin sets `WaterHeightfieldDry::Level(fill_level)` (`src/shallow_water/mod.rs`), so dry texels sit at the still-water level. The fragment shader discards pixels with zero coverage from the bilinear water depth, so the outline is cut per pixel, not per triangle (`docs/water_rendering.md`). At a waterfall the CMF10 lip test only throws spray (`src/shallow_water/waterfalls.rs`).

A CPU copy of the selection (default quadtree, 13 levels, 90° field of view, 16:9, a straight river running away from the camera) gives these counts:

| View | Tiles | Triangles | Over a 2 m stream | 8 m | 20 m |
|---|---|---|---|---|---|
| On the bank, 1.7 m up, 15° down | 293 | 37.5k | 3.8% | 15% | 34% |
| 6 m up, 30° down | 252 | 32.3k | 4.5% | 18% | 35% |
| 40 m up, 35° down | 102 | 13.1k | 0.8% | 3.3% | 8.3% |

So 65% to 99% of the water triangles lie over dry ground, but they are cheap. Kühnert (2025) measured about 0.9 ns per added terrain vertex on a 20-core M4 Pro. Assuming a 10-core M5 at half that rate and a water vertex four times dearer for its LOD samples, a vertex costs about 7 ns, and 24k vertices about 0.15 ms. Dry fragments are mostly rejected by depth or discarded before the optics. The waste is large in count and small in time.

## Where it fails

Geometry problems:

- **Narrow streams at a distance.** With spacing d/32 to d/16, no vertex lands inside a 1.5 m stream beyond about 25 to 50 m. The vertices on both banks read dry texels at `fill_level`. On any river above the still level, the triangle across it then dips toward that level and passes under the banks, so the stream vanishes with distance. Per-pixel coverage can't bring back a buried surface. The LOD data still holds the stream there, at d/128 to d/64 per texel, so the mesh is about four times coarser than its data.
- **Banks.** At every wet and dry edge of a raised river the last triangle falls from the water to `fill_level`. Near the camera the band is 0.25 m and the bank hides it; further away it shows as a notch. `WaterHeightfieldDry::Bed` lifts narrow channels to bank height instead.
- **Waterfalls.** A heightfield can't fold. The face is a ramp across one cell, 2 m in `beach` and 0.25 m in `pool`, with no air gap and no throw: water leaving a lip at 1 m/s over a 3 m drop lands about 0.8 m out. No triangle density fixes this, CBT included.
- **Elevation.** Tiles are culled against `height_bounds`, −24 to 24 m by default, so water well above the mesh plane is culled unless the bounds widen. Heights are f16 (`WATER_LOD_SURFACE_FORMAT`), which steps 6 cm between 64 and 128 m.

Simulation-grid problems, which no mesh fixes:

- **Stream width in cells.** `beach` runs 2 m cells, with a 2/3 m nest only near the camera. A 1.5 m stream is under one parent cell, which the staggered solver with reflective wet and dry faces can't carry. 0.25 m cells, as in `pool`, would be 64 M cells over 2 km.
- **The drop.** The solver sees a waterfall as a one-cell step.

## CBT against the alternatives

A CBT over a river-corridor base mesh would give near-equal screen-size triangles inside the corridor only. Against this project:

- **Outline.** Coverage already cuts the outline per pixel. A CBT base mesh is a fixed halfedge buffer, so following a flood or a drying bar means a rebuild that drops the incremental state. A static corridor is what `RiverSpline` (`src/shallow_water/river.rs`) already describes.
- **Density.** The 49 px target means 7 to 10 px edges, five to ten times denser than today's tiles. More `tile_cells` gets the quadtree there, and triangles aren't where the frame goes.
- **Pops.** CBT has no morph; CDLOD's is built and tested.
- **Waterfalls.** A base mesh could hold a vertical face, but the curtain is ballistic and changes with discharge. A ribbon rebuilt each frame does it in about 1k triangles, as `WaterBreakerCurl` (`src/water/curl.rs`) does for breakers.
- **Cost on the M5.** Nine dependent kernels took 0.085 to 0.1 ms on the 6800 XT. Assuming three to five times lower throughput and a dispatch-bound floor, expect 0.2 to 0.5 ms, plus a per-vertex decode of up to d matrix products unless vertices are cached. Memory is about 6 MiB. WGSL has no 64-bit integers, but a corridor of 1,000 halfedges refined from 40 m² roots to 0.01 m² needs about 22 index bits, so u32 fits. A 2 km domain needs no doubles.
- **Effort.** Port 9 kernels, build a halfedge mesh from the outline, add a second draw path beside `quadtree_gpu.rs`, and hide the seam where it overlaps the quadtree at the sea. Probably 2,000 to 3,000 lines, like the `lod` branch.

The simpler fixes, with M5 costs under the 7 ns per vertex assumption above:

| Fix | Fixes | Cost | Effort |
|---|---|---|---|
| `tile_cells` 16 | Streams twice as far, bank notches halved | 85k vertices, ≈ 0.6 ms | One field, then measure |
| River-weighted LOD distance | Streams resolved along the corridor | ≈ 5k to 10k more triangles | Moderate |
| Extend the wet surface into dry cells | Buried streams, bank notches | ≈ 0 | Small, in `heightfield.wesl` |
| Strip mesh along `RiverSpline` | Streams at any distance | ≈ 6k triangles for `beach`'s 1.5 km river | Moderate. Overlaps the quadtree and misses floods |
| Waterfall curtain mesh | Waterfall faces | ≈ 1k triangles per fall | Moderate |
| Masking dry tiles | Triangle waste | Saves ≤ 0.1 ms | Small with the task shader of v1v.26 |
| Mesh shaders (v1v.26) | Draw overhead | Unmeasured | Filed. Changes how tiles are drawn, not which |

The river-weighted distance scales the CDLOD distance by a weight that drops near a river, read from a distance field of the splines or the coverage. CDLOD stays crack-free only if the scaled distance changes no faster than true distance, so the weight has to blend over a band that widens with distance, about 0.75 d for a weight of 1/4. The bound is derived here, not taken from Strugar (2009).

## Recommendation

Close the spike without adopting CBT. Its strengths are curved and arbitrary base meshes at planet scale. Here the outline is already cut per pixel, the triangles cost a fraction of a millisecond, and the failures come from vertices sampling dry texels, from waterfalls a heightfield can't fold, and from solver cells wider than the stream. CBT fixes none of these, loses the morph, and is the most complex mesh path on the list. Revisit it if a scene needs water on a curved or non-planar base mesh at scale, or for an adaptive solver on bisection meshes, which Dupuy (2020) names as future work.

## Proposed beads

- **Measure the water mesh's vertex cost against `tile_cells`.** Record `pool` and `beach` with xctrace at `tile_cells` 8, 16 and 32, from the bank and from 40 m up. This replaces the 7 ns estimate and decides whether a global density increase is cheap enough.
- **Extend the wet surface into dry cells for the mesh height.** Add a dry mode where a dry texel takes the mean surface of wet neighbours within two cells, else the bed. Edge triangles stay level with the water and coverage still cuts the outline. Check `pool` and the Lumahaʻi in `beach`.
- **Weight the quadtree's LOD distance by distance to rivers.** Split and morph on distance times a weight that drops near `RiverSpline` centrelines, blended over a band that keeps the crack-free bound. Use a strip mesh along the spline only if this can't hold 2 m streams to 200 m. Depends on the measurement bead.
- **Draw a waterfall curtain mesh from the lip faces.** Have the waterfall pass emit lip segments with discharge and speed, and sweep a ballistic sheet to the pool below, rebuilt like `WaterBreakerCurl`, in the water material with foam. Related to bevy_fft-eyk.
- **Support water far above the mesh plane.** Bound tiles by the heights in the LOD data instead of the global `height_bounds`, and store heights relative to a datum per slice or per heightfield so f16 keeps centimetres on mountain rivers.
- **Spike a 1D channel solver for streams under three cells wide.** Run Saint-Venant along `RiverSpline` where the 2D grid is too coarse, joined to it where the stream widens, writing depth into the LOD data along the corridor. A simulation fix, independent of v1v.18 and v1v.16.
- **Cull tiles with no water in the task shader.** Once v1v.26 exists, test a coarse wet mask per slice and skip dry tiles. Keep it only if it measures. Depends on v1v.26.

## Review

An independent review (2026-11-19) confirmed the constants and the CBT figures. The buried-stream mechanism is in the code: `render_eta` returns `dry_level` for dry cells, and `eta_at` blends it bilinearly. It has not been seen in a capture, though, and the distance thresholds are derived, not observed. `Level` was a deliberate choice, so the wet extension becomes a third mode. The ±24 m tile cull removes a mountain stream outright, and that matters more than the mesh density. The per-vertex cost model and the triangle table are unmeasured. Follow-up work is in bead epic bevy_fft-zog.2, starting with a screenshot repro.
