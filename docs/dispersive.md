# Dispersive hybrid waves in bevy_fft

The [dispersive module](../src/dispersive/mod.rs) runs the Jeschke and Wojtan hybrid on the GPU: each step splits the free surface into a bulk shallow-water field and an Airy surface-wave field, advances them separately, then merges again. Use it when you want boat wakes, ripples, and flooding-style bulk motion in one heightfield, without a full 3D fluid solve.

The paper is *Generalizing Shallow Water Simulations with Dispersive Surface Waves* (Jeschke and Wojtan, 2023). Compute lives in [`assets/dispersive/dispersive.wgsl`](../assets/dispersive/dispersive.wgsl). A NumPy reference that tracks the same Algorithms 1 through 4 sits in [`python/dispersive_ref`](../python/dispersive_ref).

Register `FftPlugin` before `DispersivePlugin`. Do not enable `EwavePlugin` in the same app. Both splice after FFT resolve.

## Long waves stay in the bulk

The diffusion split (paper §4.1) sends long waves into the bulk SWE and short waves into Airy. The cutoff scale is about λ ≈ 2πh. At the usual rest depth of 4 m that is roughly 25 m.

A wide mound (radius on the order of half the tile) has kh ≪ 1. Almost all of its energy stays in the bar. SWE is non-dispersive, so those waves travel together, wrap the periodic domain, and re-focus. That looks like “low-frequency waves that never disperse.” It is expected hybrid behavior, not a broken Airy step.

A compact bump with kh near 1 puts a real share into h̃ and shows Airy rings. The NumPy `ripple` scenario uses that IC. The realtime example starts from a flat basin. Brush or Space seeds the field.

## Stability note on surface momentum

Carrying a raw decompose residual for q̃ (or splitting q by the height fraction h̄/h) is not Airy-consistent on fine grids. Over long runs that residual pumps surface amplitude until the field blows up.

The working rule in both NumPy and GPU is:

1. Diffuse η only for the height split.
2. Put all momentum in the bar.
3. Zero q̃ after decompose.
4. Let Algorithm 2 regenerate surface flux from h̃ each step.

Interactive wakes still write tangential flux into the surface field with `brush_tilde_wake` after the split. Height splashes use `brush_flow_impulse` on `state` only.

## What each substep does

One hybrid frame follows paper Algorithm 1.

Decomposition runs depth-dependent FTCS diffusion on the free surface η, then sets h̄ = H − bed and h̃ = h − h̄. Bar momentum is the full cell q. Surface q starts at zero (plus any wake brush).

Bulk flow advances h̄ and staggered face velocities with the CMF10 MacCormack path (same spirit as the standalone shallow_water module).

Airy runs paper Algorithm 2 in Fourier space: multi-depth evaluation of ω(k, h̄) with the β correction for finite-volume merge dispersion, half-cell phase shift for staggered q, and a soft spectral damp on high kn. The FFT of h̃ uses the Alg. 2 time average of the previous and current pre-Airy heights (stored in `tilde.a`). Airy updates q̃ only. Spatial h̃ is band-limited with the same damp mask.

Transport grows and advects surface fields through the bulk velocity (Algorithms 3 and 4). Growth and semi-Lagrangian footprints for q̃ use the midpoint face velocity. Height uses the end-of-step face velocity. Previous cell-centered ū is cached in `bar_vel_prev`.

Merge rebuilds conserved height from the divergence of bar plus surface flux, plus the paper §4.5 check flux q̌ from a half-step SL of transported h̃ onto faces.

After merge, the same soft high-k roll-off used in Airy is applied to the free surface η and to q. Airy damps h̃ alone, but a compact IC can leave a static high-frequency spike in total h at the origin. Post-merge damp clears that leftover. Physical dispersive bands for the ripple IC sit below the soft cutoff (`kn ≈ 0.28`).

The surface mesh displaces by free-surface elevation η = h − rest depth in meters (flat bed at −4 m, rest h = 4 m).

## Placing the patch in the ocean

Set `DispersiveController::placement` to a `DispersivePlacement` to put the patch in the world. With `WaterPlugin` in the app, which `OceanPlugin` adds on its own, a compute pass writes the patch into the shared water LOD data every frame, after the FFT ocean and before height queries. The cell width is `tile_world / n`. The flow texture receives velocity as flux divided by depth, blended by the same mode as height.

`DispersiveLodBlend::Add` adds the free-surface elevation on top of the ocean, so swell keeps running through the patch and a wake rides on it. Linear waves superpose, so nothing is counted twice. `DispersiveLodBlend::Replace` takes over the height and the water depth lane instead, which suits flooding and shorelines. Both fade out over `blend_cells` along the patch edge. The domain is still periodic, so waves that reach the edge wrap around under the fade band.

## Hulls on the patch

With `enable_bodies`, floating bodies press the patch down with a pressure head, the depth of their hull below the water's open level in each cell, and in deep mode the Airy step drives the flux with the surface plus that head. A moving hull then trails the wake a moving pressure patch makes in linear theory (Havelock). The test `a_towed_hull_makes_the_wake_hull_theory_gives` tows the boat example's hull at 15 knots and reads 0.39 m crest to trough 10 m abeam, where the steady linear field of its own head gives 0.39 to 0.45 m over the 58 m of patch behind it. `the_wake_has_the_height_hull_theory_gives` drives the boat itself and reads 0.40 m. Measured small-craft wakes are of this size or more: a review by the Virginia Institute of Marine Science (2017) lists 0.41 m 100 m from the sailing line of an 8 m planing cruiser at 10 knots, from Zabawa and Ostrom (1980) and Sorensen (1973).

The head's memory, which eases it toward its new value each frame (`FloatingBodySettings::pressure_relaxation`), moves with the hull: each cell reads last frame's head where the part of the hull over it was then. Where no hull is, the head is zero.

Gilligan's obstruction map makes a hull a wall to the waves: the water under it rests at its open level and no flux crosses it. That suits a hull at rest, which reflects and shadows passing waves. A hull of draft d moving at speed U through the water meets mostly its own waves, whose transverse waves have wavenumber k = g / U², and in deep water e^{-k d} of a wave's flux runs below depth d. So a cell is a wall while that share is under half and open once it is more. For the boat's 0.3 to 0.6 m draft that happens at 2 to 3 m/s. Obstructing a fast hull cut its wake to 40% of the pressure's.

Bodies read the water under their own hull without the patch's height and flow there, only the water below the patch, as the obstruction made them do. The voxel forces arrive frames late and have no planing lift, so riding their own wave field made the boat sit in its own trough and porpoise, and its drag curve is calibrated without it. A hull therefore doesn't feel other waves in the patch under itself either, such as another boat's wake.

## NumPy reference

```bash
cd python
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m dispersive_ref video --scenario ripple --frames 2400 --stride 10 \
  --out ../target/dispersive_ref/height_ripple_10x.mp4
.venv/bin/python -m dispersive_ref validate-long --scenario baseline --n 256
```

Named scenarios live in `python/dispersive_ref/scenarios.py`. `baseline` is the wide mound used for long-horizon amplitude gates. `ripple` is the compact bump for dispersive rings. `validate-long` checks that max|η| and volume stay bounded over thousands of frames.

## Try the realtime example

```bash
cargo run --example dispersive --features free_camera
cargo run --example wake --features free_camera
```

Runs unpaused from a flat basin. LMB splash or drag for a wake. Space adds a center splash. Pause and Step for single hybrid frames. The `wake` example places a patch in the FFT ocean and circles it with a scripted disturbance.

FFT bin layout matches the rest of the crate: DC at `(0,0)`, then positive bins, then wrapped negatives. `k_xy` in the dispersive WGSL must stay on that convention. A centered `i − N/2` grid only matches if the spectrum was fftshifted first.
