//! Typical setup: add [`fft::FftPlugin`], spawn [`fft::FftSource`], read
//! [`fft::FftTextures::spatial_output`] and [`fft::FftTextures::power_spectrum`] after GPU resolve.
//! Use [`fft::prelude`] for a small import set; 256×256 round-trip helpers live on [`fft::FftSource`].

pub mod complex;
pub mod fft;
