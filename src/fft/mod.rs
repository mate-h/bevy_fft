use bevy::{
    app::{App, Plugin, Update},
    asset::{Handle, load_internal_asset},
    core_pipeline::core_2d::graph::{Core2d, Node2d},
    ecs::{
        component::Component, query::QueryItem, schedule::IntoScheduleConfigs,
        system::lifetimeless::Read,
    },
    math::UVec2,
    prelude::Image,
    reflect::Reflect,
    render::{
        Render, RenderApp, RenderSystems,
        extract_component::{ExtractComponent, ExtractComponentPlugin, UniformComponentPlugin},
        render_graph::RenderGraphExt,
        render_resource::*,
    },
    shader::Shader,
};

mod node;
pub mod prelude;
pub mod resources;

pub use node::{FftNode, FftSpectrumPassthroughNode, splice_spectrum_pass};
pub use resources::{prepare_fft_bind_groups, FftTextures, prepare_fft_textures};

use node::{FftComputeNode, FftResolveOutputsNode};
use resources::{
    FftBindGroupLayouts, FftPipelines, FftRootsBuffer, copy_input_textures_to_fft_buffers,
    prepare_fft_resolve_bind_groups, prepare_fft_roots_buffer,
};

use crate::complex::c32;

/// Fills `roots` with the twiddle factors used by the forward FFT. For each stage with
/// `base = 2^order`, the value at index `base + k` is `exp(-i·2π·k / base)`. The WGSL `get_root`
/// routines in `fft.wgsl` and `ifft.wgsl` read the table using the same layout.
pub fn fill_forward_fft_twiddles(roots: &mut [c32; 8192]) {
    roots.fill(c32::new(0.0, 0.0));
    for order in 0..13u32 {
        let base = 1u32 << order;
        let count = (base >> 1).max(1);
        for k in 0..count {
            let theta = -2.0 * std::f32::consts::PI * (k as f32) / (base as f32);
            roots[(base + k) as usize] = c32::new(theta.cos(), theta.sin());
        }
    }
}

/// Builds a twiddle table that covers forward stages up through `orders ≤ 12`.
pub fn forward_fft_twiddle_table() -> [c32; 8192] {
    let mut roots = [c32::new(0.0, 0.0); 8192];
    fill_forward_fft_twiddles(&mut roots);
    roots
}

#[cfg(test)]
mod layout_tests {
    use super::{FftSettings, forward_fft_twiddle_table};
    use bevy::render::render_resource::ShaderType;
    use std::f32::consts::PI;

    #[test]
    fn fft_settings_uniform_size_matches_wgsl() {
        // If this fails, update `bindings.wgsl` so `FftSettings` matches the Rust uniform layout.
        let n = FftSettings::min_size().get() as usize;
        assert_eq!(n, 48, "update bindings.wgsl FftSettings if this changes");
    }

    /// Cheap regression check for the twiddle indexing logic.
    #[test]
    fn twiddle_table_matches_formula() {
        let roots = forward_fft_twiddle_table();
        for order in 1u32..=12 {
            let base = 1u32 << order;
            let count = (base >> 1).max(1);
            for k in [0u32, 1, count / 2, count.saturating_sub(1)] {
                if k >= count {
                    continue;
                }
                let idx = (base + k) as usize;
                let theta = -2.0 * PI * (k as f32) / (base as f32);
                assert!(
                    (roots[idx].re - theta.cos()).abs() < 1e-5
                        && (roots[idx].im - theta.sin()).abs() < 1e-5,
                    "roots[{idx}] order={order} k={k}",
                );
            }
        }
    }
}

pub(crate) mod shaders {
    use bevy::asset::{Handle, uuid_handle};
    use bevy::shader::Shader;

    pub const C32: Handle<Shader> = uuid_handle!("f9123e70-23a6-4dc3-a9fb-4a02ea636cfb");
    pub const BUFFER: Handle<Shader> = uuid_handle!("33f1ccb3-7d87-48d3-8984-51892e6652d0");
    pub const BINDINGS: Handle<Shader> = uuid_handle!("1900debb-855d-489b-a973-2559249c3945");
    pub const PLOT: Handle<Shader> = uuid_handle!("a021a614-a32b-4b4b-9604-00005bce1436");
    pub const RESOLVE_OUTPUTS: Handle<Shader> =
        uuid_handle!("c4d5e6f0-1111-4222-a333-444455556666");
}

/// Chooses how much of the 2D FFT pipeline runs on each frame.
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq, Reflect)]
#[repr(u32)]
pub enum FftSchedule {
    /// Runs the forward transform so spatial buffer **A** becomes spectrum **C**. Skips the inverse pass.
    #[default]
    Forward = 0,
    /// Runs the inverse transform so spectrum **C** becomes spatial **B**. Skips the forward pass.
    Inverse = 1,
    /// Runs forward then inverse so data flows **A** → **C** → **B**.
    /// Pick this when spectrum buffer **C** is edited on the GPU between the two passes.
    ForwardThenInverse = 2,
}

impl FftSchedule {
    #[inline]
    pub const fn to_bits(self) -> u32 {
        self as u32
    }

    #[inline]
    pub fn try_from_bits(bits: u32) -> Option<Self> {
        match bits {
            0 => Some(Self::Forward),
            1 => Some(Self::Inverse),
            2 => Some(Self::ForwardThenInverse),
            _ => None,
        }
    }
}

/// Describes what your [`FftInputTexture`] images represent so CPU uploads go to the right buffer.
///
/// Spatial images land in **A**. Ready-made spectra land in **C**.
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq, Reflect)]
#[repr(u32)]
pub enum FftInputDomain {
    #[default]
    Spatial = 0,
    Spectrum = 1,
}

/// Tells procedural or pattern shaders whether they are filling spatial samples or a spectrum.
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq, Reflect)]
#[repr(u32)]
pub enum FftPatternTarget {
    /// Writes spatial-domain samples into buffer **A**. Fits a forward FFT or a full round trip.
    #[default]
    SpatialA = 0,
    /// Writes spectral data into **C**, which pairs well with a workflow that only runs an inverse FFT.
    SpectrumC = 1,
}

#[derive(Component, Clone, Reflect)]
pub struct FftSource {
    /// Grid width and height for this FFT entity.
    pub size: UVec2,
    /// Base-two logarithm of the transform size. The stock WGSL expects eight stages for 256×256 inputs.
    pub orders: u32,
    /// Extra border pixels reserved for future windowing or padding work.
    pub padding: UVec2,
    /// Twiddle factors shared with the GPU through [`FftRoots`].
    pub roots: [c32; 8192],
    /// Forward, inverse, or both. See [`FftSchedule`].
    pub schedule: FftSchedule,
    /// Whether CPU uploads are spatial images or spectra. Spectrum mode targets buffer **C**.
    pub input_domain: FftInputDomain,
    /// Where generated patterns should write, either **A** or **C**, independent of the schedule.
    pub pattern_target: FftPatternTarget,
    /// Scales the resolved spatial preview. The underlying FFT buffers stay untouched.
    pub spatial_display_gain: f32,
}

impl Default for FftSource {
    fn default() -> Self {
        Self {
            size: UVec2::new(256, 256),
            orders: 8,
            padding: UVec2::ZERO,
            roots: forward_fft_twiddle_table(),
            schedule: FftSchedule::Forward,
            input_domain: FftInputDomain::Spatial,
            pattern_target: FftPatternTarget::SpatialA,
            spatial_display_gain: 1.0,
        }
    }
}

impl FftSource {
    /// Builds the usual 256×256 setup that runs a forward FFT and inverse FFT each frame.
    ///
    /// Data moves **A** → **C** → **B**. Keep `orders` at eight to match the bundled radix-2 stages.
    pub fn grid_256_forward_then_inverse() -> Self {
        Self {
            size: UVec2::splat(256),
            orders: 8,
            padding: UVec2::ZERO,
            roots: forward_fft_twiddle_table(),
            schedule: FftSchedule::ForwardThenInverse,
            input_domain: FftInputDomain::Spatial,
            pattern_target: FftPatternTarget::SpatialA,
            spatial_display_gain: 1.0,
        }
    }

}

/// Hooks user images into the FFT path. The real handle is required, and a missing imaginary image is treated as zero.
#[derive(Component, Clone, Reflect)]
pub struct FftInputTexture {
    pub real: Handle<Image>,
    pub imag: Option<Handle<Image>>,
}

impl ExtractComponent for FftInputTexture {
    type QueryData = Read<FftInputTexture>;
    type QueryFilter = ();
    type Out = FftInputTexture;

    fn extract_component(item: QueryItem<'_, '_, Self::QueryData>) -> Option<Self::Out> {
        Some(item.clone())
    }
}

#[derive(Component, Clone, Copy, Reflect, ShaderType)]
#[repr(C)]
pub struct FftSettings {
    pub size: UVec2,
    pub orders: u32,
    pub padding: UVec2,
    /// [`FftSchedule`] encoded the way the WGSL uniform expects.
    pub schedule: u32,
    /// [`FftPatternTarget`] encoded the way the WGSL uniform expects.
    pub pattern_target: u32,
    pub window_type: u32,
    pub window_strength: f32,
    pub radial_falloff: f32,
    pub normalization: f32,
}

impl ExtractComponent for FftSettings {
    type QueryData = Read<FftSource>;
    type QueryFilter = ();
    type Out = FftSettings;

    fn extract_component(item: QueryItem<'_, '_, Self::QueryData>) -> Option<Self::Out> {
        Some(FftSettings {
            size: item.size,
            orders: item.orders,
            padding: item.padding,
            schedule: item.schedule.to_bits(),
            pattern_target: item.pattern_target as u32,
            window_type: 0,
            window_strength: 0.0,
            radial_falloff: 0.0,
            normalization: item.spatial_display_gain,
        })
    }
}

#[derive(Component, Clone, Copy, Reflect, ShaderType)]
#[repr(C)]
pub struct FftRoots {
    pub roots: [c32; 8192],
}

impl ExtractComponent for FftRoots {
    type QueryData = Read<FftSource>;
    type QueryFilter = ();
    type Out = FftRoots;

    fn extract_component(item: QueryItem<'_, '_, Self::QueryData>) -> Option<Self::Out> {
        Some(FftRoots { roots: item.roots })
    }
}

pub struct FftPlugin;

impl Plugin for FftPlugin {
    fn build(&self, app: &mut App) {
        load_internal_asset!(app, shaders::BUFFER, "buffer.wgsl", Shader::from_wgsl);
        load_internal_asset!(app, shaders::BINDINGS, "bindings.wgsl", Shader::from_wgsl);
        load_internal_asset!(app, shaders::C32, "../complex/c32.wgsl", Shader::from_wgsl);
        load_internal_asset!(app, shaders::PLOT, "plot.wgsl", Shader::from_wgsl);
        load_internal_asset!(
            app,
            shaders::RESOLVE_OUTPUTS,
            "resolve_outputs.wgsl",
            Shader::from_wgsl
        );
        // Forward and inverse passes still load from `assets/` so they are easy to tweak.

        app.register_type::<FftSource>()
            .register_type::<FftSchedule>()
            .register_type::<FftInputDomain>()
            .register_type::<FftPatternTarget>()
            .register_type::<FftRoots>()
            .register_type::<FftInputTexture>()
            .add_systems(
                Update,
                (
                    resources::prepare_fft_textures,
                    copy_input_textures_to_fft_buffers.after(resources::prepare_fft_textures),
                ),
            )
            .add_plugins((
                ExtractComponentPlugin::<FftSettings>::default(),
                UniformComponentPlugin::<FftSettings>::default(),
                ExtractComponentPlugin::<FftRoots>::default(),
                ExtractComponentPlugin::<FftTextures>::default(),
                ExtractComponentPlugin::<FftInputTexture>::default(),
            ));
    }

    fn finish(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .init_resource::<FftBindGroupLayouts>()
            .init_resource::<FftPipelines>()
            .init_resource::<FftRootsBuffer>()
            .add_systems(
                Render,
                (
                    prepare_fft_roots_buffer
                        .in_set(RenderSystems::Prepare)
                        .before(RenderSystems::PrepareBindGroups),
                    prepare_fft_bind_groups.in_set(RenderSystems::PrepareBindGroups),
                    prepare_fft_resolve_bind_groups
                        .in_set(RenderSystems::PrepareBindGroups)
                        .after(prepare_fft_bind_groups),
                ),
            )
            .add_render_graph_node::<FftComputeNode>(Core2d, FftNode::ComputeFFT)
            .add_render_graph_node::<FftSpectrumPassthroughNode>(Core2d, FftNode::SpectrumPass)
            .add_render_graph_node::<FftComputeNode>(Core2d, FftNode::ComputeIFFT)
            .add_render_graph_node::<FftResolveOutputsNode>(Core2d, FftNode::ResolveOutputs)
            .add_render_graph_edge(Core2d, FftNode::ComputeFFT, FftNode::SpectrumPass)
            .add_render_graph_edge(Core2d, FftNode::SpectrumPass, FftNode::ComputeIFFT)
            // Ordering ensures `spatial_output` and `power_spectrum` exist before the main 2D pass samples them.
            .add_render_graph_edge(Core2d, FftNode::ComputeIFFT, FftNode::ResolveOutputs)
            .add_render_graph_edge(Core2d, FftNode::ResolveOutputs, Node2d::MainOpaquePass);
    }
}
