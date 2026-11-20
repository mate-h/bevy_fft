# Grids for narrow streams: a decision

A mountain stream is 0.3 to 2 m wide and runs at 5 to 30% slope over cobbles and boulders. The shallow water solver needs about three cells across a channel to carry it, so a stream needs cells of 0.1 to 0.67 m. The beach's grid has 2 m cells and its nest 2/3 m. This note compares a fine 2D grid against a 1D channel along the `RiverSpline`, and picks one. Nothing here is implemented beyond what the beads below describe.

## What a fine grid costs

A whole solver step with momentum-conserving advection measured 1.68 ms on a 1024² grid on the M5, about 1.6 ns per cell per step (`tests/shallow_water_dam_break_gpu.rs`). The step count per frame comes from two limits:
- **Stability:** the time step must stay under 0.5 Δx / (|u| + √(g h)).
- **Velocity clamp:** water falling a 3 m drop reaches 7.7 m/s, so 0.5 Δx/Δt must exceed that.

For water at 3 m/s and 0.2 m deep, at 60 frames a second:

| Cells | Steps per frame | 60 m stream, 7 m corridor | 51 m square around the camera |
|---|---|---|---|
| 0.2 m | 2 | 12k cells, 0.04 ms | 66k cells, 0.21 ms |
| 0.1 m | 3 | 42k cells, 0.20 ms | 262k cells, 1.26 ms |
| 0.074 m | 4 | 76k cells, 0.49 ms | 477k cells, 3.1 ms |

The 16 ms estimate in the review came from the nest's time refinement, not from the cells. Each nest level takes three steps of a third of its parent's step. The third level below the beach's 2 m grid therefore runs 27 steps a frame, where its own stability limit needs four. A 512² grid at that level costs 11 ms a frame instead of 1.6 ms.

## The 1D channel

Saint-Venant along the spline costs microseconds and handles a channel of any width. But it has no cross-stream direction, so it can't hold any of the targets that make a stream look real:
- recirculation behind boulders, or eddy lines
- flow splitting around rocks
- the oblique standing waves where water leaves a step
- foam gathering on convergence lines

It also needs a 1D–2D junction where the stream widens into the 2D grid, with its own conservation work. It suits only the far field, a stream seen from hundreds of meters, where a flow map along the spline is cheaper again.

## Decision

Use a 2D grid of 0.1 to 0.2 m cells, stepped as often as its own stability limit asks, not locked to a multiple of its parent's step. Use momentum-conserving advection on it: it is cheaper than MacCormack and moves bores at the right speed. Far streams stay on the coarse grid or get a flow map; the 1D channel is dropped.

There are two ways to place such a grid:
1. **A stream grid along the spline:** a placed grid over the stream's corridor. Its cost follows the stream's length, not the camera. The plugin has one `ShallowWaterController`, so this needs either several placed grids or a nest that follows the spline instead of the camera.
2. **A camera-following fine level:** a third nest level around the player. The nest has to let a child take as many steps as its stability limit needs rather than three. The nest's time interpolation and reflux already sum over the child's steps, so they generalize to N steps of Δt/N.

Option 2 reuses the shift and nest code and serves wading anywhere, so it goes first. Option 1 follows for streams the player watches from a distance.
