# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `bodies::voxelize` turns a closed mesh into voxels with the hull's volume per voxel, and the outward normal and area of the surface voxels, for buoyancy and drag on floating bodies.
- `WaterBreakingFronts` traces the breaking fronts of a `WaterHeightfield` into lines along the crest after Thuerey et al. 2007, with ids kept while a front is tracked. The `beach` panel draws them.
- `tools/bake_topobathy.py` bakes terrain tiles and a sea-floor depth texture for any bay or river area from Esri TopoBathy, with river channels carved along splines.
- Shared water LOD data in [`water`](src/water/mod.rs): camera-centred texture array slices that the FFT ocean, shallow water grids and dispersive patches all write into, drawn by one `WaterSurfaceMaterial`.
- Ocean cascades with finite-depth dispersion, TMA spectra, swell and full 2D horizontal displacement. Several FFT sources can run in one app.
- GPU height, slope and velocity queries on the water surface with async readback through `OceanQueries`.
- `OceanClipmap` and `WaterQuadtree` water meshes. The quadtree selects tiles on the GPU and draws them indirectly.
- Shallow water grids with a world placement, ocean forcing along their edges from the FFT long waves, and a solver core that other crates can step.
- Foam carried between frames on the LOD slices, whitecaps from the cascade Jacobian, breaking foam on steep shallow water fronts, and per-pixel foam advected with Neyret's flow layers.
- Water optics over opaque scenes: depth-aware transmission, screen-space refraction and reflection, Jerlov extinction, and caustics on the bed.
- An underwater view with Snell's window, total internal reflection, multiple and single scattering in the water body, and light shafts.
- The `beach` example on Kauai LiDAR terrain drawn by bevy_terrain, and the `wake` example with a dispersive patch on the FFT ocean.

### Changed

- The water surface lights itself with a reduced copy of Bevy's PBR model. Directional lights, their shadows, the view environment map and ambient light still apply, while point, spot and area lights and reflection probes no longer light the water.
- `WaterOpticsUniform::trace_steps` sets the march steps of the screen-space traces.

## [0.2.0] - 2026-10-14

### Added

- Dispersive surface waves: the [`dispersive`](src/dispersive/mod.rs) module combines bulk shallow water with Airy surface waves, with docs in [docs/dispersive.md](docs/dispersive.md) and the `dispersive` example.

### Changed

- Upgrade to Bevy 0.20 and Rust 1.97.1.
- Migrate internal shaders to WESL for Bevy 0.20 and naga_oil.

## [0.1.0] - 2026-07-01

Initial crates.io release with GPU FFT, ocean, shallow water, and eWave on Bevy 0.19.
