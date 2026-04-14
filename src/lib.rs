//! GPU-backed FFT **library** for Bevy.
//!
//! Add [`fft::FftPlugin`], spawn [`fft::FftSource`], and select [`fft::FftSchedule`] as needed.
//! After the resolve step, [`fft::FftTextures::spatial_output`] holds the spatial result and
//! [`fft::FftTextures::power_spectrum`] summarizes the spectrum for display.
//!
//! Import the common API from [`prelude`], which includes FFT types, extension helpers such as
//! [`fft::FftBindGroupLayouts`], and the usual [`ocean`] surface types.
//!
//! **Main world vs render world.** [`fft::FftSource`] is the component you spawn and edit in the
//! main app. Each frame it is extracted into [`fft::FftSettings`], [`fft::FftRoots`], and related
//! render-world components. Queries on the render sub-app (for example `With<FftSettings>`) use
//! the extracted types, not [`fft::FftSource`] directly.
//!
//! For GPU edits to the spectrum, insert a compute pass on buffer **C** between the forward FFT
//! and the inverse FFT, then connect it with [`fft::splice_spectrum_pass`].
//!
//! The [`ocean`] module registers [`ocean::OceanSurfaceMaterial`] ([`bevy::pbr::ExtendedMaterial`] over
//! [`bevy::pbr::StandardMaterial`] plus [`ocean::OceanSurfaceExtension`]) and displaces a mesh using
//! [`fft::FftTextures::spatial_output`]. Broader ocean and bloom plans live in **`ROADMAP.md`**.

pub mod complex;
pub mod fft;
pub mod ocean;
pub mod prelude;
pub mod shallow_water;
