//! Common import path for in-repo examples and domain plugins: FFT entities, render-graph ordering
//! ([`crate::fft::disable_spectrum_passthrough`], [`crate::fft::FftNode`], [`crate::fft::run_fft_forward`],
//! [`crate::fft::run_fft_resolve_spectrum`]), [`crate::ocean`], [`ewave`](crate::ewave), and
//! [`dispersive`](crate::dispersive) surface types, shallow-water surface types, and symbols the `fft`
//! example uses ([`crate::fft::FftInputTexture`], [`crate::fft::prepare_fft_bind_groups`]).
//!
//! Twiddle helpers, [`crate::fft::run_forward_fft`], [`crate::fft::run_inverse_fft`], and other
//! internals remain on [`crate::fft`] and [`crate::fft::resources`].

pub use crate::dispersive::{
    DispersiveController, DispersiveGridImages, DispersiveLodBlend, DispersiveMaterialUniform,
    DispersivePlacement, DispersivePlugin, DispersiveSimRoot, DispersiveSurfaceExtension,
    DispersiveSurfaceMaterial, DispersiveSurfaceTag,
};
pub use crate::ewave::{
    EwaveController, EwaveGridImages, EwaveMaterialUniform, EwavePlugin, EwaveSimRoot,
    EwaveSurfaceExtension, EwaveSurfaceMaterial, EwaveSurfaceTag,
};
pub use crate::fft::resources::{FftBindGroupLayouts, FftBindGroups, prepare_fft_bind_groups};
pub use crate::fft::{
    FftInputTexture, FftNode, FftPlugin, FftSchedule, FftSettings, FftSkipStockPipeline, FftSource,
    FftSystemSet, FftTextures, disable_spectrum_passthrough, run_fft_forward,
    run_fft_resolve_outputs, run_fft_resolve_spectrum,
};
pub use crate::ocean::{
    OCEAN_MAX_CASCADES, OCEAN_QUERY_MAX_POINTS, OceanCascade, OceanClipmap, OceanDynamicUniform,
    OceanFoamMask, OceanFoamPhase, OceanFoamUniform, OceanH0Image, OceanH0Uniform, OceanPlugin,
    OceanQueries, OceanQueryResults, OceanSample, OceanSimSettings, OceanSurfaceSettings,
};
pub use crate::shallow_water::{
    ShallowWaterBorder, ShallowWaterController, ShallowWaterMaterialUniform, ShallowWaterPlacement,
    ShallowWaterPlugin, ShallowWaterSurfaceExtension, ShallowWaterSurfaceMaterial,
    ShallowWaterSurfaceTag, round_particle_count,
};
pub use crate::water::{
    WATER_LOD_MAX_SLICES, WaterLodData, WaterLodGraphSystems, WaterLodScratch, WaterLodSettings,
    WaterLodSlices, WaterLodSystems, WaterPlugin, WaterSurfaceExtension, WaterSurfaceMaterial,
    WaterSurfaceTag, WaterSurfaceUniform, WaterViewer,
};
