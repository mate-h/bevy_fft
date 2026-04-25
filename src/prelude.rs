//! Common import path for in-repo examples and domain plugins: FFT entities, splice helpers,
//! [`crate::ocean`] and [`ewave`](crate::ewave) surface types, shallow-water surface types, and symbols
//! the `fft` example uses ([`crate::fft::FftInputTexture`], [`crate::fft::prepare_fft_bind_groups`]).
//!
//! Twiddle helpers, [`crate::fft::FftSpectrumPassthroughNode`], [`crate::fft::run_forward_fft`],
//! [`crate::fft::run_inverse_fft`], and other internals remain on [`crate::fft`] and [`crate::fft::resources`].

pub use crate::ewave::{
    EwaveController, EwaveGridImages, EwaveMaterialUniform, EwavePlugin, EwaveSimRoot,
    EwaveSurfaceExtension, EwaveSurfaceMaterial, EwaveSurfaceTag,
};
pub use crate::fft::resources::{FftBindGroupLayouts, FftBindGroups, prepare_fft_bind_groups};
pub use crate::fft::{
    FftInputTexture, FftNode, FftPlugin, FftSchedule, FftSettings, FftSkipStockPipeline, FftSource,
    FftSystemSet, FftTextures, splice_after_resolve_outputs, splice_spectrum_pass,
};
pub use crate::ocean::{
    OceanDynamicUniform, OceanFoamMask, OceanFoamPhase, OceanFoamUniform, OceanH0Image,
    OceanH0Uniform, OceanMaterialUniform, OceanPlugin, OceanSimSettings, OceanSurfaceExtension,
    OceanSurfaceMaterial, OceanSurfaceTag,
};
pub use crate::shallow_water::{
    ShallowWaterBorder, ShallowWaterController, ShallowWaterMaterialUniform, ShallowWaterPlugin,
    ShallowWaterSurfaceExtension, ShallowWaterSurfaceMaterial, ShallowWaterSurfaceTag,
    round_particle_count,
};
