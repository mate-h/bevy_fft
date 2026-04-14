# Shallow water in bevy_fft

The [shallow_water module](../src/shallow_water/mod.rs) runs a **virtual-pipe** shallow water model on the GPU: bed height and water depth share a 2D field, fluxes live on staggered edges, and each step does accelerate → mass-limited scale → move, with optional particles and brush-style interaction. A **root render-graph compute** pass runs before the camera so the simulation textures stay in sync when the PBR surface samples them.

The **example** (`examples/shallow_water/main.rs`) adds atmosphere-style lighting, an environment map light, and egui controls. It does not use the FFT pipeline; `ShallowWaterPlugin` is standalone.

## Bevy integration

Simulation WGSL is in [`assets/shallow_water/simulator.wgsl`](../assets/shallow_water/simulator.wgsl). State is a [`ShallowWaterController`](../src/shallow_water/mod.rs) resource extracted to the render world, with GPU buffers and bind groups prepared each frame. The example uses bevy_egui on `EguiPrimaryContextPass` so panels run after egui’s context is ready.

**Storage.** Read-write storage on `Rg32Float` is not available on some GPUs (notably Metal). This crate uses `Rgba32Float` for the bed or water and velocity textures and only uses the **R** and **G** channels. Flow textures stay `R32Float`.

**Rendering.** [`ShallowWaterSurfaceMaterial`](../src/shallow_water/mod.rs) is an `ExtendedMaterial<StandardMaterial, ShallowWaterSurfaceExtension>` in the same style as the ocean surface: `StandardMaterial` carries the dark teal base and low roughness for strong specular and environment reflections. The extension displaces Y from filtered bed + water height, recomputes smooth world normals from bilinear height samples, and only nudges albedo toward sand in a thin shallow band.

## Try it

```bash
cargo run --example shallow_water --features free_camera
```

The example enables `FLOAT32_FILTERABLE` where supported so linear sampling on the simulation textures is valid.

Based on [webgpu-shallow-water](https://github.com/mate-h/webgpu-shallow-water).
