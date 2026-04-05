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
pub mod resources;
pub use node::FftNode;
pub use resources::FftTextures;

use node::{FftComputeNode, FftResolveOutputsNode};
use resources::{
    FftBindGroupLayouts, FftPipelines, FftRootsBuffer, copy_input_textures_to_fft_buffers,
    prepare_fft_bind_groups, prepare_fft_resolve_bind_groups, prepare_fft_roots_buffer,
    prepare_fft_textures,
};

use crate::complex::c32;

/// Twiddle factors `exp(-i·2π·k / base)` at `roots[base + k]` for `base = 2^order`,
/// matching `get_root` in `assets/fft.wgsl` / `assets/ifft.wgsl`.
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

/// Pre-filled table for `orders ≤ 12` (4096-point) forward FFT stages.
pub fn forward_fft_twiddle_table() -> [c32; 8192] {
    let mut roots = [c32::new(0.0, 0.0); 8192];
    fill_forward_fft_twiddles(&mut roots);
    roots
}

#[cfg(test)]
mod layout_tests {
    use super::{forward_fft_twiddle_table, FftSettings};
    use bevy::render::render_resource::ShaderType;
    use std::f32::consts::PI;

    #[test]
    fn fft_settings_uniform_size_matches_wgsl() {
        // WGSL uniform layout for `bindings.wgsl` must match `ShaderType` / encase.
        let n = FftSettings::min_size().get() as usize;
        assert_eq!(n, 48, "update bindings.wgsl FftSettings if this changes");
    }

    /// Spots mistakes in `fill_forward_fft_twiddles` indexing without duplicating the full DIT.
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

// Public-facing component
#[derive(Component, Clone, Reflect)]
pub struct FftSource {
    /// Size of the FFT texture
    pub size: UVec2,
    /// Number of FFT steps (log2 of size)
    pub orders: u32,
    /// Padding around the source texture
    pub padding: UVec2,
    /// Roots of the FFT
    pub roots: [c32; 8192],
    /// Inverse flag (`IFFT` reads from C). Ignored for dispatch when [`roundtrip`](Self::roundtrip) is true.
    pub inverse: bool,
    /// If true, each frame runs forward FFT then inverse FFT (buffers A, then C, then B), with the
    /// pattern written to spatial buffer A, like a spatial to spectrum to spatial check.
    /// If false and `inverse` is true, the pattern may load a made up spectrum into C; that need not
    /// match the FFT of whatever you draw in space, so inverse FFT will not recover that spatial image.
    pub roundtrip: bool,
}

impl Default for FftSource {
    fn default() -> Self {
        Self {
            size: UVec2::new(256, 256),
            orders: 8,
            padding: UVec2::ZERO,
            roots: forward_fft_twiddle_table(),
            inverse: false,
            roundtrip: false,
        }
    }
}

/// Provide external textures to the FFT/IFFT pipelines. The real component is
/// required; imag may be omitted to default to zero.
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

// Internal render-world component
#[derive(Component, Clone, Copy, Reflect, ShaderType)]
#[repr(C)]
pub struct FftSettings {
    pub size: UVec2,
    pub orders: u32,
    pub padding: UVec2,
    pub inverse: u32,
    pub roundtrip: u32,
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
            inverse: item.inverse as u32,
            roundtrip: item.roundtrip as u32,
            window_type: 0,
            window_strength: 0.0,
            radial_falloff: 0.0,
            normalization: 1.0,
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
        // Load shaders
        load_internal_asset!(app, shaders::BUFFER, "buffer.wgsl", Shader::from_wgsl);
        load_internal_asset!(app, shaders::BINDINGS, "bindings.wgsl", Shader::from_wgsl);
        load_internal_asset!(app, shaders::C32, "../complex/c32.wgsl", Shader::from_wgsl);
        load_internal_asset!(app, shaders::PLOT, "plot.wgsl", Shader::from_wgsl);
        load_internal_asset!(app, shaders::RESOLVE_OUTPUTS, "resolve_outputs.wgsl", Shader::from_wgsl);
        // TODO: Add FFT and IFFT shaders as internal assets

        app.register_type::<FftSource>()
            .register_type::<FftRoots>()
            .register_type::<FftInputTexture>()
            .add_systems(
                Update,
                (
                    prepare_fft_textures,
                    copy_input_textures_to_fft_buffers.after(prepare_fft_textures),
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
            .add_render_graph_node::<FftComputeNode>(Core2d, FftNode::ComputeIFFT)
            .add_render_graph_node::<FftResolveOutputsNode>(Core2d, FftNode::ResolveOutputs)
            .add_render_graph_edge(Core2d, FftNode::ComputeFFT, FftNode::ComputeIFFT)
            // Resolve runs after IFFT so `spatial_output` / `power_spectrum` are valid for the 2D pass.
            .add_render_graph_edge(Core2d, FftNode::ComputeIFFT, FftNode::ResolveOutputs)
            .add_render_graph_edge(Core2d, FftNode::ResolveOutputs, Node2d::MainOpaquePass);
    }
}
