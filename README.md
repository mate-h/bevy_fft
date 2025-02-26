# FFT Compute Shader in Bevy

This project implements GPU based Fast Fourier Transform (FFT) using compute shaders in Bevy engine.

## Project Goals

## Core FFT Component

- Use appropriate texture formats for representing complex numbers
  - multi-channel use cases, assuming a maximum of 4 channels in the texture
    - a pair of Rgba32Float textures
    - or a single Rgba32Uint texture with real and imaginary components packed at half the precision
  - single channel use case
    - put the real and imaginary parts into a Rgba32Float texture as the red and green channels
- Use appropriate workgroup sizes to maximize GPU parallelism and memory access patterns, typically 256 threads
- Use ping-pong buffers for FFT computation to handle large textures (1024x1024+) that exceed workgroup shared memory limits
- Implement proper normalization and boundary conditions to prevent floating-point errors and edge artifacts
- Memory barriers to prevent race conditions in compute shaders
- Multi dimensional FFT, 1D (audio), 2D (image), 3D (volume)

### Realistic Ocean Surface Simulation

Create a real-time ocean surface simulation using the Fast Fourier Transform (FFT) to generate realistic wave patterns based on physical models.
We compute the Inverse FFT of the frequency domain spectrum to get the time domain wave height.
In the frequency domain spectrum, each complex value represents a wave component where the magnitude corresponds to the wave amplitude and the phase angle determines the wave offset. When transformed back to the spatial domain through IFFT, these components combine to produce the height field of the ocean surface.

### Functional Requirements
- Generate initial wave spectrum based on statistical wave models
  - JONSWAP spectrum default constants
  - Phillips spectrum default constants
- Implement time-dependent phase evolution using dispersion relation
- Calculate spectrum with horizontal displacement (choppiness)
- Perform horizontal and vertical FFT passes using butterfly operations
- Generate normal maps from displacement data for realistic lighting
- Support wind direction and speed parameters to affect wave patterns
- Implement proper dispersion relation formula
- Handle special cases like DC term removal to prevent artifacts
- Apply statistical wave constants as user facing components
- Scale displacement based on ocean size and desired wave height
- Apply displacement mapping to mesh vertices in a vertex shader
- Implement realistic water lighting with fresnel effect and subsurface scattering
- Support environment reflections and refractions
- Add foam generation at wave peaks
- Reduce tiling artifacts by compute multiple octaves of ocean waves at different scales
- Use HDR rendering with proper exposure control, also see Bloom section
- Utilize compute shaders for GPU-accelerated FFT
- Support dynamic tessellation based on camera distance
- Scale detail and displacement based on viewing distance

### Physically Based Bloom Convolution

Develop a high-quality bloom post-processing effect using FFT-based convolution with physically accurate light scattering kernels. This can be obtained from a camera by taking reference photographs or can be generated using a compute shader, or an external software.

# Bloom Algorithm

The Bloom algorithm first identifies the brightest point in the scene, which becomes the center of the kernel. It then separately measures the center energy (concentrated at the kernel center) and scatter dispersion energy (dispersed outward from the center). The algorithm also determines the maximum scatter dispersion values to properly clamp the kernel and prevent artifacts.

The user-facing component provides separate control over center energy and scatter dispersion energy. This allows for more realistic bloom that mimics how real camera lenses respond to bright light sources.

## Functional Requirements

- forward FFT of the Scene, transform the HDR scene into frequency domain
- forward FFT of the Kernel, transform the light scattering kernel into frequency domain
- point-wise multiply the two frequency-domain representations, no texture filtering is needed
- inverse FFT, transform back to spatial domain to get the bloom result
- downsampling for large kernels, reduces the resolution of the multiplication, improving performance.
- pre-compute and store optimized kernel values for common light sources (e.g., sun, moon) to reduce runtime computation.
- cache frequently used FFT results to avoid redundant calculations.
- reduction-based approach to find the maximum luminance pixel in the scene
- energy measurement separately surveys the center zone and edge zone to accurately measure energy distribution
- packs kernel data into structured buffers for optimal GPU access
- dynamic resizing of the kernel based on the current screen resolution while preserving its optical properties
- bloom effect remains stable between frames to prevent flickering
- integrate with other post-processing effects in the rendering pipeline
- calculate tint based on the energy distribution to ensure physically plausible coloration