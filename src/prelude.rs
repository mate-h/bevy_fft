//! One import path for FFT utilities, render-graph extension helpers, and commonly paired [`crate::ocean`] types.
//!
//! The list includes everything needed for day-to-day use and for custom spectrum or pattern passes
//! (`FftBindGroupLayouts`, [`crate::fft::splice_spectrum_pass`], and related items).
//! For WGSL or types that are not re-exported here, use [`crate::fft`] and [`crate::fft::resources`] directly.

pub use crate::fft::resources::{FftBindGroupLayouts, FftBindGroups};
pub use crate::fft::{
    FftInputDomain, FftInputTexture, FftInvalidSize, FftNode, FftPatternTarget, FftPlugin,
    FftSchedule, FftSettings, FftSource, FftSpectrumPassthroughNode, FftSystemSet, FftTextures,
    fft_orders_for_size, fill_forward_fft_twiddles, forward_fft_twiddle_table,
    prepare_fft_bind_groups, prepare_fft_textures, splice_spectrum_pass,
};
pub use crate::ocean::{
    OceanDynamicUniform, OceanH0Image, OceanH0Uniform, OceanMaterialUniform, OceanPlugin,
    OceanSimSettings, OceanSurfaceExtension, OceanSurfaceMaterial,
};
