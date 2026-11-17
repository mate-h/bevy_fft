# Wakes that outlive the patch: a research spike

A boat's wake lives in a [dispersive patch](dispersive.md), 256 m across on 256² cells in the `wake` and `floating` examples. The patch follows the boat and trims its edge with Tessendorf's (d/L)^α taper, so a wake fades within about half the patch, some 128 m, behind the boat. This note works out what Water Surface Wavelets (Jeschke et al. 2018) would cost to carry the wake further. It also compares wave packets (Jeschke et al. 2017) for the same job. Nothing here is implemented.

## What the paper does

Water Surface Wavelets store an amplitude A(x, θ, k) per cell, direction and wavenumber, and advect it at group speed. The paper uses 16 directions and usually one wavenumber sample, on a grid of up to 4096² cells about 1 m apart. Two diffusion terms keep fronts from breaking apart. The drawn height sums 120 directions. Each direction looks up a precomputed 1D profile, 4096 samples integrated over 400 wavenumbers, with a fixed random phase per direction.

On a GTX 1070, the 4 km sea takes 8.54 ms for the amplitudes and 4.16 ms for the height. That is 4096² × 16 ≈ 268 M samples, about 32 ps per sample. Four wavenumber samples cut a 70 fps case to 20 fps.

## Cost here

A wake-only domain doesn't need the paper's size. It needs to cover where the wake should still be visible, such as 1 km around the boat. That domain recentres by whole cells, as `ShallowWaterShift` and `DispersiveController::follow` already do.

| Domain | Cells | Directions × k | Samples | f16 memory | Amplitude step (paper rate) |
|---|---|---|---|---|---|
| 1 km at 4 m | 256² | 16 × 1 | 1.0 M | 2 MB | ≈ 0.03 ms |
| 1 km at 2 m | 512² | 16 × 1 | 4.2 M | 8 MB | ≈ 0.13 ms |
| 2 km at 2 m | 1024² | 16 × 2 | 34 M | 67 MB | ≈ 1.1 ms |

The amplitude step is cheap. Reconstruction is the cost. The water LOD has 10 slices of 256², but only the slices the wake domain covers at a useful texel size need it. The 120 direction taps per texel then mean about 4 slices × 65 536 texels × 120 ≈ 31 M profile reads a frame. That's in the range of `ocean_lod_write`, which reads several cascades per texel and costs about 4.9 ms at the beach on the M5 with a cache-bound texture unit. Halving the directions to 60 and touching only the texels that are inside the domain and whose amplitude is non-zero keeps it near 1 ms. So the estimate is 0.2 ms to 2 ms a frame on the M5, depending on reach and direction count. Reconstruction would write the LOD data in `WaterLodGraphSystems::Blend` beside the patch, adding to the height and the horizontal shift.

## Why it doesn't fit the wake well

- **Phase.** The amplitude grid drops phase, and the paper notes that Kelvin interference needs phase the coarse grid doesn't keep. A wake carried this way turns from a Kelvin pattern into an incoherent band of chop with the right directions and energy. That suits the far wake seen from a distance. It doesn't suit a wake seen from the boat a few hundred metres back, which is where the patch hands over.
- **Hand-over.** The patch would need to turn the outgoing field near its trim band into A(x, θ, k). That means a windowed directional transform of the band, or measuring the energy flux the trim removes per direction each frame. Both are new work with no reference in the paper, which injects wakes from an analytic source.
- **Depth.** The paper's dispersion has no tanh(kh), so it has no shoaling or refraction. Adding depth to the group speed is easy, but refraction needs a θ-advection term from the depth gradient that the paper doesn't derive. Wakes reaching the beach are exactly the case that needs it.
- **Ambient swell.** The FFT ocean already supplies the incoherent detail the paper gets from wind sources and its initial spectrum, so most of its other benefits are already covered here.

## Wave packets

Wave packets (Jeschke et al. 2017) keep phase. Each packet carries a Gaussian envelope at group speed around crests moving at phase speed, with tanh(kh) dispersion and refraction along depth. They reproduce Kelvin wakes from a moving source. At about 2 × 10⁻⁴ ms per packet on a CPU, 85 000 packets fit in a 60 fps frame. A GPU version with one thread per packet, splatted into the LOD data, would be cheaper per packet. A wake needs packets only along the trim band, emitted from the boat's path rather than measured from the patch: one per crest per direction, with the source matched to the patch's speed and hull width.

Their cost is the splat. A packet covers a few wavelengths, so 50 000 packets touching about 100 texels each is about 5 M texel writes a frame. That is atomic or sorted-tile work, comparable to the particle pool's foam deposit.

## Recommendation

- Don't build Water Surface Wavelets for wakes. The amplitude grid is cheap, but it loses the Kelvin pattern at the hand-over and has no depth refraction, and both matter for the wakes this crate draws.
- If long wakes are wanted, prototype GPU wave packets seeded from the boat's path beyond the patch. The first measurement should be splat cost and hand-over seams at 1 km of reach. That is filed as a follow-up bead.
- Water Surface Wavelets remain a candidate for an open-sea scene with reflections off many islands and no shore. No example needs that yet.
