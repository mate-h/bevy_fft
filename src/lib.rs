//! GPU-backed FFT **library** for Bevy.
//!
//! Add [`fft::FftPlugin`], spawn [`fft::FftSource`], and select [`fft::FftSchedule`] as needed.
//! After the resolve step, [`fft::FftTextures::spatial_output`] holds the spatial result and
//! [`fft::FftTextures::power_spectrum`] summarizes the spectrum for display.
//!
//! Common types live in [`fft::prelude`]. For GPU edits to the spectrum, insert a compute pass
//! on buffer **C** between the forward FFT and the inverse FFT, then connect it with
//! [`fft::splice_spectrum_pass`].
//!
//! The [`ocean`] module registers [`ocean::OceanSurfaceMaterial`] ([`bevy::pbr::ExtendedMaterial`] over
//! [`bevy::pbr::StandardMaterial`] plus [`ocean::OceanSurfaceExtension`]) and displaces a mesh using
//! [`fft::FftTextures::spatial_output`]. Broader ocean and bloom plans live in **`ROADMAP.md`**.

pub mod complex;
pub mod fft;
pub mod ocean;
