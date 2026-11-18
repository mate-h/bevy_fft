# Driveable boat demo: plan

A new `boat` example: one small motor boat you steer with the keyboard across the FFT ocean. The boat floats on the existing GPU voxel forces, Rapier integrates it, and a dispersive wake patch follows it so its hull leaves a Kelvin wake. A chase camera follows it.

## What already exists

- **Floating bodies** (`src/bodies`): CPU voxelizer (`voxelize`), GPU buoyancy, friction, pressure drag, suction and splashes per voxel, read back a few frames late. `FloatingBody::compensated` corrects for that delay. The `floating` example hands them to Rapier as `ExternalForce`.
- **Wake patch** (`src/dispersive`): deep-mode Airy patch that follows a point by whole cells (`DispersiveController::follow`), stored as a deviation from the FFT surface. `enable_bodies` adds pressure-head and obstruction maps, and `sync_dispersive_footprint` keeps a `BodyFootprintTarget` over the patch so floating bodies press their hull into it. The GPU test `a_towed_body_trails_a_wake_in_deep_mode` shows a kinematic barge towed at 4 m/s leaves a wake.
- **`wake` example**: patch plus FFT ocean, but the wake comes from a scripted brush, not a hull.
- **Splashes** into `WaterParticlesPlugin` from waterline patches moving into the water.

The gap: nothing yet combines a free, Rapier-driven floating body with the wake patch, adds propulsion and steering, or follows it with a camera.

## Design

### Hull and boat

- A procedural displacement hull built in code: about 7 m on the waterline, 2.5 m beam, 0.6 m draft at rest, V-bottomed with a fine bow and a transom stern. The mass is whatever the voxel volume floats at that draft. A block coefficient of about 0.4–0.45 gives about 3.5–4 t, not the 2.5 t first guessed. It's voxelized at 0.2 m for the forces.
- The hull builder lives in `examples/common/boat_hull.rs`, so the GPU test can include it with `#[path]` the way the examples include `screenshot.rs`. Move it into the library only if a second user appears.
- The render mesh is the hull plus a simple deck, cabin and rail. A Rapier convex hull of the hull mesh handles collisions.
- Centre of mass set low and slightly aft, with the inertia from Rapier's collider mass properties, overridden by a mass that gives the target draft.

### Propulsion and steering (CPU, in the example; promote later if reused)

A `BoatDrive` component with the helm state and coefficients. Each frame, before Rapier's step:

- **Thrust** along the hull's forward axis at the propeller point, about 0.5 m below the waterline at the transom. Scaled by throttle, from -0.4 to 1. Faded to zero as the propeller leaves the water.
  - `OceanQueries` is a GPU readback 1–3 frames late, which is fine for a fade. But its height leaves out the pressure-head dent, so a point under the transom reads about a draft low.
  - So sample a point just astern of the hull, or use the body's `submerged_volume` share as a proxy. Decide in bead 4.
  - Thrust's full value is sized from the measured drag curve (bead 3).
- **Rudder** as a lifting surface behind the propeller. Lift is `0.5 ρ A C_L(α) v²`, applied at the rudder point, so it yaws and slightly heels the boat.
  - The angle of attack comes from the local flow at the rudder, from the hull's linear and angular velocity, plus a share of the propeller wash. So the rudder also acts as a skeg against yaw, and the boat steers at low speed.
  - `C_L` is linear up to stall at about 25°.
- **Skeg**: a fixed lifting term at the keel aft, computed the same way. The voxel pressure drag on the bow gives a destabilising yaw moment, and this balances it so the boat runs straight hands-off.
- Helm input is smoothed: the throttle ramps at about 0.5/s and the rudder slews at about 40°/s.

### Wake

- A deep-mode patch with `enable_bodies`, following the boat with a margin of 4 cells, as in the `wake` example.
  - Free floating bodies already write their footprint into the patch through `BodyFootprintTarget` (`body_footprints` in `forces.rs`, gated by `displaces_water`).
  - The head memory is keyed per world cell, so follow shifts and the footprint cooperate.
- **Cell size**: try 0.5 m cells (256 cells, 128 m) against 1 m. The dent is only 2–3 cells wide at 1 m on a 2.5 m beam, and the transverse wavelength at 3 m/s is 5.8 m. The code comment against finer cells concerns the hybrid decomposition, which deep mode skips.
- **Time step**: the patch steps a fixed `controller.dt` (1/60 s) per rendered frame, while Rapier and the boat run on real time. At 120 Hz the wake would run twice as fast as the boat, and the Kelvin wavelength and angle would be wrong. Set `controller.dt` from the clamped frame delta each frame, or make the library do so, in bead 4. `pressure_relaxation` is per frame too.
- **Speed**: the waterline is 7 m, so hull speed is about 3.3 m/s (Froude 0.4). The force model has no wave-making resistance, so nothing caps speed naturally. Target 2–4 m/s, and cap thrust by the drag curve from bead 3. Planing is out of scope.
- Bow and quarter spray from the existing splash path, tuned up for the boat.
- **Wake foam**: `hull_foam` in the patch's LOD write is driven only by the brush pointer. Drive `pointer` from the boat's stern with `brush_strength = 0`, in bead 7. This looks workable, but it's untested.

### Camera and controls

- A chase camera behind and above the boat: a spring on position, looking at a point ahead of the bow, ignoring the boat's roll and pitch. The wheel zooms, within bounds.
- **The camera moves the water the boat feels.** The water LOD slices centre on the viewer, and the finest is 8 m wide. Each voxel takes the finest slice that contains it, so zooming or flying the free camera away changes the forces.
  - Bound the chase distance so the hull stays in the fine slices.
  - The `F` free camera flies freely, and the helm is off while it's active. Bead 2 measures how much the boat's draft and heave change with the camera far away, and says so in the example's docs. A LOD centre pinned to the boat is the library follow-up if it matters.
- Controls: W/S or Up/Down for throttle, A/D or Left/Right for the rudder, Space to centre the helm, R to reset the boat upright where it is. The gamepad and the orbit camera are follow-ups.
- An egui HUD: speed in knots and m/s, throttle, rudder angle, heading, and a collapsible tuning panel (thrust, rudder area, sea state).

### Scene

- FFT ocean with the same three cascades, a moderate default sea, and a sea-state slider.
- Atmosphere, sun, shadows, bloom and exposure copied from `wake`.

## Work breakdown (beads)

1. **Boat hull builder.**
   - A procedural hull in `examples/common/boat_hull.rs`: render mesh, collider, voxelization, and the mass set from the voxel volume at design draft.
   - Unit test: it floats at its design draft, from the voxel volume against mass.
2. **Boat example skeleton.**
   - Ocean, a Rapier floating boat at rest, the chase camera, the free camera toggle, a screenshot hook.
   - The boat must sit level and stable in calm and moderate seas.
   - Measure how much the draft and heave change with the camera close vs far.
3. **Drag curve.**
   - Tow the hull kinematically, or push it at constant speeds in the example, at 1–5 m/s, and read back the drag force.
   - Size the full thrust from it and pick the top speed.
   - Record the numbers in the bead.
4. **Wake patch following the boat.**
   - A deep-mode patch with body maps and follow, with `controller.dt` set from the frame time.
   - Compare 0.5 m and 1 m cells.
   - Check that a free, Rapier-driven boat leaves a visible Kelvin wake, and that the head lag (about 20 frames at relaxation 0.05) and the delayed forces don't make it porpoise.
   - Decide the propeller depth source.
   - Headless screenshot from above.
5. **Propulsion, rudder and skeg.**
   - `BoatDrive`: helm input and smoothing, thrust with the out-of-water fade, rudder and skeg lift from the local flow.
   - Unit tests for the force math: zero rudder gives no yaw, the turn has the right sign, and the skeg moment opposes a yaw rate.
6. **HUD and tuning panel.** Speed, helm, heading, sea state, the physics knobs.
7. **Spray, wake foam and polish.**
   - Bow spray tuning, and stern foam through the patch's `hull_foam` pointer.
   - Defaults that look good.
   - README section, CHANGELOG entry, showcase screenshot.
8. **Headless GPU behaviour test.**
   - Run on `TimeUpdateStrategy::ManualDuration`, serialized with the other GPU tests, `#[ignore]` like them.
   - Full throttle for about 30 s of simulated time, then assert a steady speed in the expected band, and heel and pitch under limits.
   - Hands-off with the rudder centred: the heading drifts less than a bound.
   - Hard-over rudder: it turns through 180° within a bound.
9. **Performance check.**
   - An xctrace profile of the example at 1440p.
   - The budget is the `wake` example's frame plus at most about 1 ms for the body pass and footprint.
   - Record the numbers.

Dependencies:
- 1 → 2 → 3 → 4 → 5 → 6 → 7.
- 8 comes after 5.
- 9 comes after 5.

## Risks

- **Force delay at speed.** The forces are read back a few frames late.
  - `compensated` corrects heave and scales drag, but leaves the buoyancy torque stale, so roll and pitch may oscillate.
  - Thrust, rudder and skeg are applied on the CPU without delay.
  - Don't raise `pressure_relaxation` to fight this: its docs record that a higher value made a body bob without end.
- **Pressure head lag.** At relaxation 0.05 the head takes about 20 frames, so at 3 m/s the dent trails the hull by about 1 m. The towed test only covered a 4 m wide barge at 4 m/s. Check it in bead 4, and if the dent visibly lags, move the head toward the hull's motion (a library change).
- **Frame-rate dependence.** The fixed patch `dt` (see Wake) is a correctness bug, not just a look.
- **Patch following.** Whole-cell shifts at 3–4 m/s are a few cells a second. Check visually for a pop under the hull.
- **Deep vs hybrid mode.** Open sea deep mode is right for 100 m water. `choose_mode` keeps it.
- **Wave forces on a small hull in a large sea** may roll it over. Start with a moderate sea, and keep the metacentric height positive with a low centre of mass.

## Out of scope / follow-ups

- Planing hull physics.
- Propeller wash foam and a turbulent stern trail.
- Wakes beyond the patch (bevy_fft-v1v.55).
- Multiple boats and AI boats.
- Boat sound.
- Gamepad, orbit camera, buoys as slalom marks.
- A LOD centre pinned to the boat rather than the camera.
- The stale `displaces_water` doc comment in `forces.rs`, which mentions only `WaterHeightfield`. Fix it in passing in bead 4.
