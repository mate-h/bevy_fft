# bevy_fft

This crate is a small GPU FFT library for [Bevy](https://bevyengine.org). It runs a 256×256 complex 2D transform with ping-pong workspace buffers, and leaves a spectrum slot between forward and inverse passes for your own compute. The fft example applies a radial band-pass on spectrum buffer C in the middle of the pipeline; the ocean example fills C each frame and displaces a mesh using `FftTextures::spatial_output`.

<img src="./assets/showcase.jpg" alt="FFT ocean simulation demo showcase" width="600"/>

<img src="./assets/showcase2.jpg" alt="FFT round-trip band pass filter demo showcase" width="600"/>

## What it includes

The stock pipeline is built around a 256×256 complex transform with workspace tiles labeled A through D. After the graph finishes, resolved images `spatial_output` and `power_spectrum` are available for sampling. The Rust API exposes `FftPlugin`, `FftSource`, `FftSchedule`, `FftInputTexture`, `FftInputDomain`, and `FftPatternTarget`. Run `cargo doc --open` for generated API documentation, or open [`src/fft/mod.rs`](src/fft/mod.rs) as the source of truth.

FFT compute runs on the root [`RenderGraph`](https://docs.rs/bevy_render/latest/bevy_render/render_graph/graph/struct.RenderGraph.html) so it executes once per frame before camera work (the graph edges `ResolveOutputs` → `CameraDriverLabel`). Between `ComputeFFT` and `ComputeIFFT` the root graph visits `SpectrumPass`, which is a no-op until something is wired in. Register your custom node on that same root graph, call `splice_spectrum_pass` from plugin `finish`, and reuse `FftBindGroupLayouts::common` to match FFT bindings.

There is also an [ocean](src/ocean/mod.rs) entry point. `OceanPlugin` splices ocean spectrum compute into the FFT graph and registers `OceanSurfaceMaterial`, which displaces a mesh using `FftTextures::spatial_output`. It is a building block, not a complete water renderer.

Ambitious extras such as a full ocean sim or FFT bloom are sketched in [`ROADMAP.md`](ROADMAP.md).

## Try it

Generate optional test patterns if you like.

```bash
pip install numpy matplotlib pillow
python assets/generate_test_patterns.py
```

Then run the demo. The `file_watcher` feature hot-reloads WGSL while you iterate.

```bash
cargo run --example fft --features file_watcher
cargo run --example ocean --features free_camera
```

The `fft` example drives `FftSchedule::ForwardThenInverse`. Data starts in spatial A, moves to spectrum C for a radial band-pass tuned with the sliders at the top-left, then returns through IFFT to B. The `ocean` example uses `FftSchedule::Inverse` with `OceanPlugin`: a compute pass fills spectrum C each frame, then the same IFFT and resolve path writes `spatial_output` for the ocean material.

## A few types worth knowing

Pick `FftSchedule` to control how much runs each frame. `Forward` stops after the transform into C. `Inverse` assumes C is already filled and writes B. `ForwardThenInverse` runs both passes so spectrum buffer C can be edited on the GPU between them.

`FftInputDomain` steers where `FftInputTexture` lands on the CPU each update, either spatial A in `Spatial` mode or spectrum C in `Spectrum` mode. `FftPatternTarget` tells procedural shaders whether to write A or C, in line with the uniform in [`bindings.wgsl`](src/fft/bindings.wgsl). Most apps import from `bevy_fft::fft::prelude` and add `bevy_fft::ocean` only when using the mesh shader.

Textures use Rgba32Float pairs for real and imaginary storage. Kernels currently launch 256 threads per row. The ping-pong layout is meant to grow to bigger grids later. Broader wishes such as 1D or 3D FFTs and packed formats stay in [`ROADMAP.md`](ROADMAP.md).
