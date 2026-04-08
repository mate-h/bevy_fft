//! Re-exports the types most applications need when using this crate.
//!
//! Use [`FftSource::square_inverse_only`] or [`FftSource::square_forward_then_inverse`] with the
//! same power-of-two edge length as your simulation textures, or the built-in `grid_256_*` helpers.

pub use super::resources::{prepare_fft_bind_groups, prepare_fft_textures};
pub use super::{
    FftInputDomain, FftInputTexture, FftNode, FftPatternTarget, FftPlugin, FftSchedule, FftSource,
    FftSpectrumPassthroughNode, FftTextures, fft_orders_for_size, fill_forward_fft_twiddles,
    forward_fft_twiddle_table, splice_spectrum_pass,
};
