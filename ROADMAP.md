# Roadmap

Longer-term and research directions for **bevy_fft**. The main [README](README.md) describes what works today.

---

## FFT core

- Support dual **Rgba32Float** textures, packed **Rgba32Uint** variants, or real and imaginary samples in RG channels for single-channel cases.
- Tune workgroup sizing and access patterns for larger grids, such as **1024×1024** and beyond.
- Normalization, boundary handling, and memory barriers to limit drift and edge artifacts.
- **1D** audio and **3D** volume FFT paths on top of the same building blocks.

---

## Ocean surface simulation

Target: real-time height fields from statistical wave models and IFFT.

- Initial spectrum from **Phillips**, **JONSWAP**, and related parameter sets as data-driven inputs.
- Time evolution through the dispersion relation and phase updates each frame in frequency domain.
- Horizontal displacement consistent with the spectrum, often called choppiness.
- **IFFT** into a spatial height map; optionally normals and derived lighting inputs.
- Wind direction and speed driving the spectrum.
- DC and low-frequency handling to avoid visible bias.
- Components for artist-facing amplitude, scale, and ocean extent.
- Vertex displacement from the height field. The **`ocean`** module provides a starting WGSL hook; lighting, foam, SSR, HDR, and LOD stay separate work.
- Foam at crests, multi-octave summation to reduce tiling, distance-based LOD and detail scaling.

Conceptually, many pipelines generate **H(k,t)** into spectrum buffer **C**, then run **`FftSchedule::Inverse`** and optional spectrum-only passes before sampling the result on the mesh.

---

## Physically based bloom via FFT convolution

Target: HDR bloom using spectral multiplication instead of large spatial kernels.

**Idea**

- Find or define a bright anchor for the kernel, such as the peak luminance region.
- Measure **center** and **scatter** energy so the kernel stays bounded and stable.
- **Forward FFT** of the HDR contribution and of the scatter kernel.
- **Complex multiply** in frequency domain without bilinear filtering on spectra.
- **Inverse FFT** back to a bloom layer.
- Downsampled FFTs for performance on large resolutions.
- Precomputed kernels for common lights such as the sun or moon where helpful.
- Cache or reuse transforms when inputs change slowly.
- Tint from energy distribution; temporal stability to avoid flicker; composition with other post steps.

This is **not** implemented in the repo yet; the current crate focuses on the shared FFT machinery and the band-pass example.
