# Nested shallow water grids

A fine shallow water grid can run inside a coarse one, so the water near the camera or a player resolves ripples and wading while the coarse grid carries the bay and the river. [`ShallowWaterNest`](../src/shallow_water/nest.rs) couples two `ShallowWaterSolver`s both ways at a refinement of 3 in space and in time, the odd ratio Debreu and Blayo (2008) recommend for staggered grids, so every third child face lies on a parent face. The kernels are in [`nest.wesl`](../src/shallow_water/nest.wesl).

## A frame

1. Both solvers zero the face volumes they accumulate.
2. The nest copies the parent's depth and face velocities.
3. The parent takes one step.
4. The child takes three steps of a third of the parent's step. After each, its dynamic band, a strip one parent cell wide along its edge, takes the parent's surface over the child's own bed and the parent's face velocities, bilinear in space and blended between the copy and the stepped parent in time. Parent cells that are dry do not count in the surface. The faces on the band's inner edge keep the child's value, so the fluxes through the interface are the child's own.
5. The feedback runs on the parent. Cells over the child's interior take the mean depth of their nine child cells, and the faces between them the mean of their three child faces. The parent cells beside that region swap the volume their shared face passed this step for the volume the child's three faces on it passed over its three steps (Berger and Colella 1989).

The child's interior changes only by the fluxes through its edge, which are the interface faces, and the parent outside it changes by its own fluxes with the interface ones swapped for the child's. So the pair conserves the parent's water to float precision. Without the reflux, the test's parent loses about 1e-6 of its volume in 90 frames.

## Use

Build both solvers, then `ShallowWaterNest::new` with a `ShallowWaterNestDescriptor`: the parent cell under the child's first cell, the band width in parent cells, and the parent's cell width. The child's cell counts must be multiples of 3 and it must lie inside the parent. Write the child's settings with a third of the parent's `dt` and `cell_size`, and give it walls on every edge, since the band carries the parent's state in. `encode_frame` encodes the whole frame into a compute pass. `set_offset` moves the child in the parent without touching either grid's contents.

`ShallowWaterSolver::encode_step_with_face_volumes` is a step that also adds each face's depth flux times dt to the `.z` lane of `mac_u_temps` and `mac_w_temps`, and `encode_clear_face_volumes` zeroes them. The nest uses both, and other couplings can too.

The overshoot limiter and evaporation remove water outside the fluxes, so turn them off where conservation matters. Over a bed that differs between the grids, the child's surface in the band follows the parent's surface, not its depth, so still water stays still. The parent's bed over the child should be the mean of the child's for the feedback to keep the surface flat.

[`tests/shallow_water_nest_gpu.rs`](../tests/shallow_water_nest_gpu.rs) runs a hump of water through a child over the parent's middle third. It checks that the parent's volume holds to 1e-7, that the child's mean surface matches the parent over its interior, and that the parent outside the child stays within 5% rms of a parent run alone. Still water over a sloped bed stays within 0.1 mm.

## Moving over the world

[`ShallowWaterShift`](../src/shallow_water/shift.rs) moves one solver's grid by whole cells, so a nest can follow the camera or a player over ground a world bed texture describes. `encode` copies the grid's depth and face velocities aside, then writes them back moved, in its own compute pass, so no thread reads a cell another thread has overwritten. `ShallowWaterShiftParams` gives the shift, the grid's corner and cell width after it, and how the cells that come in are filled:

- Their bed is the mean of `subsamples`² bilinear samples of the `ShallowWaterWorldBed`. Give a child 1, its parent 3 and the parent's own parent 9. The parent's samples then fall on its child cells' centres, so a parent cell's bed is exactly the mean of the child beds over it, and the feedback's depth mean keeps the parent's surface and its volume.
- A nest's child takes its water from the parent's surface over its own bed, wet cells only. A grid without a parent fills to `fill_level`.
- Face velocities that come in start at zero.

A shift at least the grid's size refills every cell, which is how grids start from the world. To move a child one parent cell, shift it by 3 child cells with its new `parent_offset`, and call the nest's `set_offset` with the same offset. Shift the parent before the child, so the child's new cells read the moved parent. The feedback also takes the child's mean bed for the parent cells over it.

[`tests/shallow_water_nest_moving_gpu.rs`](../tests/shallow_water_nest_moving_gpu.rs) fills both grids from a bumpy world bed with the shift kernel and moves the child 20 parent cells. Still water stays within 1e-5 m, and the child's bed matches the world under its final position. With a wave running, the parent keeps its volume to 1e-7 while the child moves.

## With the plugin

`ShallowWaterController::nested` adds a nested grid to the plugin's placed grid. `ShallowWaterNestedGrid::new` makes its images for a cell count that is a multiple of 3. Set `follow` to a world point each frame, such as the camera's. `nest_offset` rounds it to the parent cell the grid starts at, kept inside the parent, and the plugin shifts the grid whenever that changes, by 3 child cells per parent cell, or refills it after a reset or a jump past its own size. `world_bed` gives the bed of the ground it moves over. Without it the grid interpolates the parent's `bed_source`. The parent's bed is point-sampled, so the feedback replaces it with the child's mean over the child, and the parent's volume shifts a little as the grid moves. A world bed that the parent also samples three times per cell avoids that.

The grid steps with a third of the parent's time step and cell width, with walls on its own edges, no PML and no ocean forcing. The brush, sources and the border profile act on the parent only. Floating bodies press on both: while the parent has a `pressure_head`, the nested grid's `WaterHeightfield` carries its own, `ShallowWaterNestedGrid::pressure_head`, and the body pass writes each body's footprint into both. A voxel spreads its displaced volume over the cells its square overlaps, so cells a third of a meter wide under 0.25 m voxels get an even footprint. `ShallowWaterMover`s stir both grids as well. It gets a `WaterHeightfield` one `order` above the parent's, so it blends over it in the water LOD data and fades into it over `blend_cells`. Heightfields now blend in `order`.

The `beach` example nests a 288-cell grid of 2/3 m cells, 192 m across, around the camera inside its 2 m grid, with its bed from the 1 m data. `BEACH_NEST` sets its cells per edge, and 0 turns it off. [`tests/shallow_water_nested_plugin_gpu.rs`](../tests/shallow_water_nested_plugin_gpu.rs) walks a nested grid across a placed pool and checks its offset, its heightfield, its bed and its still water.

## Not yet

In the beach at 2560 × 1440 on an M5, with 2 m cells, the 288-cell nest, 96 m across, adds about 1.4 ms per frame to `shallow_water_step` (2.8–3.3 ms without it, 4.1–4.7 ms with it) and about 1 ms to `water_heightfield_pass`, about 3.5 ms of GPU time in all over two alternated pairs of xctrace recordings. A 192-cell nest, 64 m across, saves about 1 ms of that in the step; the frame totals were within the noise. The beach keeps 288 for the larger reach. The `swe-coupling` branch's camera-following rings were the prototype for this module.
