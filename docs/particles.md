# Water particles

[`particles`](../src/particles/mod.rs) throws spray, splashes and droplets off the GPU water. Particles spawn from the water data on the GPU, fly under gravity and drag, land on the same surface the water draws, and leave foam where they land, all without a readback to the CPU. `WaterParticlesPlugin` adds it, and brings in [`WaterPlugin`](../src/water/mod.rs) if it is missing.

## The pool

The particles live in one storage buffer in world space, `WaterParticleSettings::capacity` slots of 32 bytes: position and size, velocity and the seconds left to live. A slot with no life left is free. New particles take slots from an atomic cursor that runs around the pool, so once it is full the oldest are overwritten. Four counters sit beside the pool: the cursor, particles spawned, particles that died, of age, by landing, by leaving the LOD data's reach or by being overwritten, and particles alive after the frame. Spawned minus died is always alive.

Each frame one compute pass, `water_particles_pass`, runs in `WaterLodGraphSystems::Simulate` before the foam sim:

1. It clears the per-frame counts.
2. The crest emitter adds spray from breaking water.
3. Spawns from the CPU land in the pool.
4. Particles other GPU passes queued since the last frame land too.
5. Every live particle moves under gravity and quadratic air drag against the wind, `a = g - drag |v - wind| (v - wind)`. The step is exact for a constant acceleration, so a particle without drag follows its ballistic arc.
6. With `collide` on, a particle that falls through the water surface dies there and leaves foam, and one past the LOD data's reach dies too.
7. Every particle still alive appends its slot to `WaterParticles::draw`, the list the draw reads.

The simulated step is the frame's `Time` delta, the one the foam sim takes. `enabled: false` holds every particle in place.

## Where particles come from

- **The CPU.** Push `WaterParticle`s to `WaterParticleSpawns` at any time. The render world takes everything in it at the frame's extraction, so each spawn reaches the GPU once. `WaterParticleEmitter` on an entity sprays from its position at a rate, a handy source for tests and examples.
- **Breaking crests.** `WaterCrestSpray` turns breaking water into spray. The ocean, shallow water grids and dispersive patches all write their breaking intensity into the LOD data's foam lane, so one emitter covers them all. It scans the finest `slices` slices and leaves a texel that a finer slice also covers to that slice, so a breaking patch emits once however the slices nest. A texel emits `rate · area · dt · breaking` particles on average, rounded at random so the mean is exact, at full rate where the surface rises as fast as `speed` and half where it stands still. They start on the displaced surface with the water's velocity plus an upward launch. `rate` is zero by default, which turns the emitter off.
- **Floating bodies.** `FloatingBodySettings::splash` turns hulls into splashes. In the force pass, a surface patch at a body's waterline that moves into the water faster than `min_speed` emits `rate · wetted area · (speed - min_speed) · dt` particles on average, launched up and out along the hull at about the entry speed. A dropped crate and a moving bow both splash this way. They reach the pool through the queue below, a frame later.
- **Waterfalls.** `ShallowWaterController::waterfalls` throws spray where a river pours over a drop in its bed, in proportion to the water that crosses the lip. See [shallow water](shallow_water.md#waterfalls).
- **Other GPU passes.** `WaterParticleQueue` is a render-world storage buffer: an `atomic<u32>` count, three `u32` of padding, then `WATER_PARTICLE_QUEUE_CAPACITY` particles laid out like `WaterParticle`. A pass appends with `atomicAdd` on the count and writes the slot it got while that is below the capacity. The particle pass spawns them on its next run and empties the queue.

## Landing and foam

The surface test reads the finest LOD slice that holds the particle, undoes the horizontal displacement in two fixed-point steps and compares the height with the particle's, on top of `surface_y`. A landing particle adds active foam to every slice that holds its point, its footprint's share of a texel times `foam_per_drop`, in fixed point to `WaterFoamDeposit`. That is an atomic per-texel buffer the foam sim adds to its active foam that same frame and then clears. Other passes can deposit foam there too, in `WaterFoamDepositSystems`.

## Drawing

The live particles are drawn in the transparent pass with premultiplied alpha, by one indirect draw of six vertices per particle in the draw list. The vertex shader turns each six into a camera-facing quad from the pool. The draw replaces the material's mesh draw, which would run six vertices per slot. If Bevy shares the material's properties when the swap is made, the mesh draw stays, with a one-time warning, and the slots past the list collapse to nothing. A drop:

- scatters each directional light with a Henyey-Greenstein phase function of asymmetry `anisotropy`, so spray glows between the camera and the sun, and takes the sky's light, `ambient`, through the camera's exposure;
- smears along its velocity across the view over `stretch_time`, spreading the same opacity over the longer quad;
- keeps at least `min_pixels` on screen, with its opacity cut by the area it lacks, so distant spray thins into a haze instead of flickering, and at most `max_pixels`, to bound the fill;
- fades into the geometry behind it over `soft_distance` when the camera has a `DepthPrepass`, so it never cuts a hard line into the water or the shore.

## Budget

The crest spray thins away from the viewer as `1 / (1 + (d / half_distance)²)`, measured from the centre of the finest slice, and its drops grow by the inverse square root, so distant spray covers the same share of the surface with fewer drops. `cap` bounds the spray per frame. `WaterParticleSettings::with_quality` sets the pool size, the spray's cap, slices and reach, and the smallest drop on screen for `WaterParticleQuality::Low`, `Medium` or `High`.

On an M5 at 2560 × 1440, the beach with spray on spent 0.14 ms in `water_particles_pass` and 0.78 ms in the transparent pass's vertex stage on the default 65 536 slots, back when the draw ran over every slot. It now runs over the live particles only.

## Debugging

- `WaterCrestSpray::debug_potential` replaces the spray with one still marker per emitting texel, as wide as the texel times the square root of its share of the full rate.
- `WaterParticleSettings::read_counters` reads the counters back into `WaterParticleStats` every frame, a few frames late. It is off by default, since the readback costs a copy and a map per frame.

## Examples and tests

- `pool` has a fountain on its near bank and a crate that splashes into a river.
- `floating` splashes as the bodies land on the ocean.
- `beach` and `ocean` have a Spray panel for the crest spray, its debug view and the counters. `BEVY_FFT_SPRAY=0` starts them with spray off.

[`tests/water_particles_gpu.rs`](../tests/water_particles_gpu.rs) checks ballistic arcs, the counts when the ring overruns, a particle landing on still water and foaming its texel only, crest spray in proportion to a breaking patch's area across two slices and none on still water, the distance falloff, and the counters' readback flag. [`tests/floating_body_splash_gpu.rs`](../tests/floating_body_splash_gpu.rs) checks that a body entering the water splashes and one at rest does not. Run them with `cargo test --test <name> -- --ignored`.

## Not yet

Bubbles under the water, surface-riding particles, sorting and ribbons along breaking fronts are open beads. Particles don't fade with distance fog yet.
