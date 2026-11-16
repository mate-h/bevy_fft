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

## Not yet

Moving the child with the camera, filling the cells that come in from world bathymetry, and drawing several nested grids into the water LOD are the next steps. The `swe-coupling` branch's camera-following rings were the prototype for this module.
