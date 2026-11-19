# Rapids, jumps and falls: a solver spike

Rivers here run on the [staggered CMF10 solver](shallow_water.md) in `src/shallow_water/simulator.wesl`. This note asks whether a Kurganov and Petrova (2007) central-upwind step, as Brodtkorb et al. (2010) run it on the GPU, would do better on mountain streams, rapids, hydraulic jumps, wet and dry banks and waterfall lips. Nothing here ran on the GPU. Both schemes were ported to 1D NumPy and run on standard tests.

## The prototypes

`solvers.py` holds both. `Cmf` ports one `encode_step_inner` line by line on a strip one cell high, with the shipped constants and without the PML. A switch swaps MacCormack for the momentum-conserving staggered advection of Stelling and Duinmeijer (2003), SD03 below. `Kp` is KP07 with generalized minmod (θ = 1.3), the positivity fix, desingularized velocity, SSP-RK2 at CFL 0.25 and ghost cells. An option adds the partially flooded cell reconstruction of Bollermann, Chen, Kurganov and Noelle (2013), BCKN13. `t1_lake.py` to `t6_dt.py` print the tables. All of them, with outputs, are in `.dev/research_notes/river_solver_prototypes/` (gitignored), with the reviewer's checks `r1.py` and `r2.py`.

## Lake at rest

The water stands at 5 m for 60 s over two beds with 1 m cells. The rough bed has cell-to-cell steps up to 4 m, a dry island and a one-cell pinnacle 2.5 m above the water. The stream bed has 50% banks, a 1 m riffle step and a one-cell boulder 2.5 m above the water.

| Scheme | Rough bed: max \|q\|, m²/s | Rough: volume | Stream bed: max \|q\| | Stream: volume |
|---|---|---|---|---|
| CMF10, dt 1/60 | 2.6 | +47% | 11 | +240% |
| CMF10, overshoot limiter off | 0 | 0 | 0 | 0 |
| KP07 | 1.7 | 0 | 3e-14 | 0 |
| KP07 + BCKN13 | 3e-14 | 0 | 3e-14 | 0 |

Without the overshoot limiter, CMF10 is exact. With it, a dry cell more than 2 Δx above a wet neighbour, with a lower wet cell on its other side, gains an eighth of that cell's depth every step, because `overshootReduce` blends toward the mean of two depths, not toward a lower surface. The pinnacle becomes a fountain. On the nest's 2/3 m cells, 2 Δx is 1.3 m, a boulder in a stream.

KP07 is exact when every cell is fully wet. Where a cell is partly flooded over a bed that drops metres within it, the positivity fix tilts the surface, as Brodtkorb et al. warn. Thirteen shoreline cells settle into a state that keeps moving water at up to 4.5 m/s, and the wet cells beside them sit up to 1.7 cm off level. BCKN13 removes it.

## Fronts and bores

Dam breaks from 1 m of water at x = 50 m, after 6 s:

| Scheme | Dry bed, front x (exact 87.6) | Wet bed h_r = 0.1, shock x (exact 68.6) | Middle depth (exact 0.396) | Middle velocity (exact 2.32) |
|---|---|---|---|---|
| CMF10, dx 1 / 0.5 / 0.25 | 65.0 / 64.0 / 63.8 | 64.0 / 64.5 / 64.3 | 0.49 | 1.88 |
| CMF10 + 2nd-order depth | 64.0 / 64.0 / 63.5 | 64.0 / 64.0 / 64.0 | | |
| CMF10 + SD03 | 81.0 / 81.0 / 80.8 | 68.0 / 68.5 / 68.8 | 0.394 | 2.32 |
| KP07 | 84.0 / 84.5 / 84.3 | 69.0 / 68.5 / 68.8 | 0.396 | 2.32 |

This is the clearest failure. CMF10 advects velocity in non-conservative form, so a bore carries the right discharge split wrongly between depth and speed: 24% too deep, 19% too slow, running at 2.3 m/s instead of 3.1. It converges to that wrong answer as the grid refines. A dry front travels 40% of Ritter's distance, and 37% on a 2 m dam break at any dt (`t6_dt.py`). Flash floods, a log dam giving way and swash on banks all run slow. SD03, changing only the advection, puts the bore within a cell, and KP07 does too with half the L1 error.

## Standing jump over a bump

This is the transcritical bump of Goutal and Maurel (1997): q = 0.18 m²/s, 0.33 m downstream, jump at x = 11.67 m from the conjugate depths.

| Scheme | dx 0.25: jump x | dx 0.1: jump x | Upstream depth (exact 0.414) | Discharge error away from jump |
|---|---|---|---|---|
| CMF10, Courant 0.23 / 0.6 | 11.75 / 11.50 | 11.70 / 11.50 | 0.401–0.410 | 0.5–4.4% |
| CMF10 + SD03 | 11.75 / 11.50 | 11.60 / 11.50 | 0.405–0.412 | 0.5–3.5% |
| KP07 | 11.75 | 11.70 | 0.413–0.414 | 0.1–0.2% |

Every scheme puts the steady jump within two cells, so non-conservative advection misplaces moving bores, not standing jumps. CMF10's steady discharge varies a few percent along the channel at Courant 0.6, and its upstream depth is up to 3% low.

## Steep channels

A 200 m channel carries q = 0.5 m²/s with a 5% inflow wobble, Manning n = 0.04, and drains at normal depth. Every scheme holds normal depth within 1–2% at slopes of 2 to 30% and conserves mass to round-off. None makes or loses water through its clamps. Roll waves should grow for Froude numbers above about 1.5:

| Slope (Fr) | dx | CMF10 | CMF10 2nd-order depth | CMF10 + SD03 | KP07 |
|---|---|---|---|---|---|
| 10% (1.9) | 1 m | flat | flat | flat | flat |
| 10% (1.9) | 0.5 m | flat | 0.18–0.21 | flat | flat |
| 30% (3.1) | 1 m | flat | 0.05–0.30 | flat | 0.136–0.142 |
| 30% (3.1) | 0.5 m | 0.09–0.23 | 0.06–0.42 | 0.13–0.15 | 0.05–0.28 |

Entries are the depth range at 180 m over the last 60 s, against normal depths of 0.191 and 0.137 m. No scheme goes unstable. Roll waves depend on resolution and numerical diffusion more than on the scheme. `second_order_depth` grows them most readily, and first-order SD03 damps them.

## The 3 m drop

60 m at slope 0.002, a 3 m step in one face, 60 m more, n = 0.03, filled from dry. The jump should sit where friction has slowed the jet to the tailwater's sequent depth, which gradually varied flow from a toe depth between the lossless one and Rand's (1955) gives.

| Scheme | q 0.25, dx 1: jump x (GVF 61.7–62.3) | q 1.0, dx 0.5: jump x (GVF 66.0–67.9) | Depth 30 m upstream (M2: 0.31 / 0.67) | Outflow / inflow |
|---|---|---|---|---|
| CMF10, dt 1/60 | 63.0 | 68.0 | 0.305 / 0.659 | 0.997 / 1.000 |
| CMF10, dt 0.08 | 63.0 | 62.0, jet clamped | 0.305 / 0.662 | 0.998 / 1.000 |
| CMF10 + SD03 | 63.0 | 67.0 | 0.305 / 0.659 | 0.997 / 1.000 |
| KP07 | 62.0 | 64.0 | 0.315 / 0.693 | 0.998 / 1.000 |
| KP07 + BCKN13 | 62.0 | 62.5 | 0.315 / 0.693 | mass +3.4% / +0.03% |

The plain schemes are stable and conserve water, and none resolves the jet: brink depth is 0.04–0.06 m against Rouse's 0.715 h_c = 0.13 m. The fall itself stays a VFX job, as in Crimson Desert. At dt 0.08 the clamp 0.5 Δx/Δt is 3.1 m/s on 0.5 m cells, under the 7.7 m/s of a 3 m fall, so the jet piles up at the foot. My BCKN13 port lacks the draining time step, so its clamps make water at the lip.

## Time step

On a dry 2 m dam break and the 30% channel at 1 m, CMF10 stays finite at every dt up to 0.12 s (Courant 0.74), held by its clamps, with the front at 116.5 m against 144.3 m whatever the dt. KP07 ran to CFL 1.0 in 1D, though positivity is guaranteed only to 0.5, and 0.25 in 2D. At the beach's steps a 10 m/s jet is Courant 0.08 on both the 2 m parent at 1/60 s and the 2/3 m nest at 1/180 s, so KP07 needs no substeps, but it evaluates fluxes twice per step.

## KP07 on this GPU

| | CMF10 step now | KP07, one fused kernel per RK stage |
|---|---|---|
| Dispatches per step | 25 (27 with face volumes) | 2, plus a max-speed reduction if dt adapts |
| Texel loads per cell | about 130; v1v.35 counted 130–155 in `limitOutflow` and `integrateHeight` | about 5 Rgba32 (w, hu, hv, B with a 16² tile and 2-cell halo, twice, plus Uⁿ) |
| Storage textures | 8, at Metal's limit | 3–4: U, U*, B, optional flux |
| Arithmetic | light | heavy: reconstruction, 4 Riemann fluxes and a source per cell per stage |
| Measured | 2.4 ms at 1024² on the M5 (v1v.35) | Brodtkorb: 2.6 ms at 512² on a GTX 285 |

KP07 trades texture reads for arithmetic, which suits a step v1v.35 found texture-cache bound. From 10 ns per cell-step on 2009 hardware against CMF10's 2.3 ns on the M5, I'd guess parity to twice the cost, unmeasured.

It is not a drop-in: cell-centred conserved variables replace the face velocities everything else reads.

- **Nesting.** `nest.wesl` prolongs face velocities and refluxes face volumes. KP needs a variant on w, hu and hv that refluxes all three fluxes, as Sætra, Brodtkorb and Lie do. Refinement 3 still works.
- **Shift, settled caches, movers, seeding** all write face velocities and need hu, hv versions.
- **Borders** map to ghost cells, with the inflow discharge imposed on the mass flux. The PML has no counterpart, but rivers don't use it, and the ocean band only touches depth.
- **Bodies.** The pressure head becomes the source −g h ∂(B + P), reconstructing w + P, which keeps a body at rest balanced. About 30 lines.
- **LOD blend and waterfalls.** `heightfield.wesl` reads only `bed_water` and the cell `velocity`, so it works if KP writes hu/h there. `waterfalls.wesl` would read the mass flux.
- **Dispersion.** The Airy hybrid (v1v.16, v1v.44) is aligned to staggered faces.

So KP07 is a parallel solver about the size of the current module.

## Recommendation

Keep CMF10 and make three targeted fixes. Don't replace it, and don't add KP07 yet.

1. Fix the overshoot limiter so it only ever lowers a cell. This is a bug, and the fix is one line.
2. Add SD03 advection as a solver option for the stream grid and nest. It fixes bore speed and dry fronts while keeping the staggered layout, nest, shift, Airy faces and bodies, and replaces eight MacCormack dispatches with one kernel per axis. Being first-order upwind, it may damp eddies behind boulders faster than MacCormack, which needs a 2D check.
3. Turn on `second_order_depth` where roll waves are wanted, and keep Δt ≤ 1/30 s or scale the clamp on falls.

KP07 wins on accuracy here but has its own steep-bed wet-dry flaw, which needs BCKN13 and a draining time step, and it means a second module. Revisit it only if SD03 fails in 2D.

## Proposed beads

- **Stop the overshoot limiter from wetting dry pinnacles** (bug, P2). In the CPU port `overshootReduce` grew a lake 47–240% in 60 s by wetting dry pinnacles 2 Δx above the water. Apply the blend only where it lowers the depth, and add a GPU test of still water around a one-cell pinnacle that holds its volume to 1e-6.
- **Add Stelling and Duinmeijer momentum advection** (task, P2). A `momentum_advection` setting swaps the MacCormack kernels for momentum-conserving upwind face advection. GPU tests: the Stoker bore within a cell of 68.6 m and the dry front within 10% of Ritter's. Profile it against MacCormack at 1024².
- **Check 2D rapids with the new advection** (task, P3, depends on the previous bead). Eddy decay behind a boulder against MacCormack, oblique standing waves in a 10% chute, and roll waves with second-order depth at 2/3 m cells. Set the stream grid and nest defaults from the results.
- **Raise the velocity clamp over waterfall faces** (task, P4). At dt 0.08 the 0.5 Δx/Δt clamp is below free-fall speed. Scale it on faces that pass the waterfall test or document a dt bound, then check the toe depth.
- **Revisit a KP07 stream solver with BCKN13 wet-dry fronts** (spike, P4, deferred, depends on the 2D rapids bead). Only if SD03 fails in 2D: a 2D CPU prototype with the draining time step, then a fused-kernel cost on the M5.

v1v.16's Airy layer stays simpler if the river stays staggered. v1v.27's CBT meshing reads only the LOD data and doesn't depend on this.

## Not covered

Everything is 1D, so cross waves, oblique jumps, bends and eddies are untested. The CMF10 port follows the source but hasn't been checked against GPU output, so confirm the overshoot fountain on the GPU first. KP07's GPU cost is an estimate.

## Review

An independent review (2026-11-19) re-ran the ports and read the shader, and changed these conclusions.

- The overshoot fountain is real and the port is faithful. It is CMF10 §2.2 verbatim, which the paper admits does not conserve volume. The threshold is exactly 2 Δx. Every GPU test that checks conservation turns the limiter off (`tests/shallow_water_solver_gpu.rs`, the nest tests), and `nest.rs` advises turning it off, but it defaults on (`solver.rs`, `mod.rs`), so no test can catch it. Turning it off on river grids and nests needs no code. It exists for shoaling waves on beaches.
- The dry-front error is a property of the non-conservative formulation, not of the clamp, `limitOutflow` or the wet threshold. Removing MacCormack advection gives the same front. Ritter's frictionless front is never the target in a river, though: with Manning n 0.03 the front is 25% short, and 13% at n 0.05. The wet-bed bore, about 20% slow and 24% deep with friction, is the defect that justifies SD03.
- The velocity clamp doesn't bite at the shipped dt of 1/60. On 2 m cells it is 60 m/s, and on 0.25 m cells 7.5 m/s. It bites on 0.1 m cells with substeps. The crate also lacks CMF10 §2.4.3's ΔH_cap pressure cap on lip faces.
- The suggestion to use `second_order_depth` for roll waves rests on single 1D runs and slightly worsens the fronts. The figure of about 130 texel loads per cell is disputed, and the KP07 cost is a guess.

Follow-up work is in bead epic bevy_fft-zog.1.
