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

use node::FftComputeNode;
use resources::{
    FftBindGroupLayouts, FftPipelines, FftRootsBuffer, copy_input_textures_to_fft_buffers,
    prepare_fft_bind_groups, prepare_fft_roots_buffer, prepare_fft_textures,
};

use crate::complex::c32;

#[cfg(test)]
mod layout_tests {
    use super::FftSettings;
    use bevy::render::render_resource::ShaderType;

    #[test]
    fn fft_settings_uniform_size_matches_wgsl() {
        // WGSL uniform layout for `bindings.wgsl` must match `ShaderType` / encase.
        let n = FftSettings::min_size().get() as usize;
        assert_eq!(n, 48, "update bindings.wgsl FftSettings if this changes");
    }
}

mod shaders {
    use bevy::asset::{Handle, uuid_handle};
    use bevy::shader::Shader;

    pub const C32: Handle<Shader> = uuid_handle!("f9123e70-23a6-4dc3-a9fb-4a02ea636cfb");
    pub const BUFFER: Handle<Shader> = uuid_handle!("33f1ccb3-7d87-48d3-8984-51892e6652d0");
    pub const BINDINGS: Handle<Shader> = uuid_handle!("1900debb-855d-489b-a973-2559249c3945");
    pub const PLOT: Handle<Shader> = uuid_handle!("a021a614-a32b-4b4b-9604-00005bce1436");
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
    /// Inverse flag
    pub inverse: bool,
}

impl Default for FftSource {
    fn default() -> Self {
        Self {
            size: UVec2::new(256, 256),
            orders: 8,
            padding: UVec2::ZERO,
            roots: [c32::new(0.0, 0.0); 8192],
            inverse: false,
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
                ),
            )
            .add_render_graph_node::<FftComputeNode>(Core2d, FftNode::ComputeFFT)
            .add_render_graph_node::<FftComputeNode>(Core2d, FftNode::ComputeIFFT)
            .add_render_graph_edge(Core2d, FftNode::ComputeFFT, FftNode::ComputeIFFT)
            // Without this edge, the 2D pass can run before FFT/IFFT compute finishes, so sprites
            // sample still-empty storage textures (appears blank).
            .add_render_graph_edge(Core2d, FftNode::ComputeIFFT, Node2d::MainOpaquePass);
    }
}
