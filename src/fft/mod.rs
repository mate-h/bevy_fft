use bevy::core_pipeline::core_3d::graph::Core3d;
use bevy_app::{App, Plugin};
use bevy_asset::load_internal_asset;
use bevy_ecs::schedule::IntoSystemConfigs;
use bevy_ecs::{component::Component, resource::Resource};
use bevy_math::UVec2;
use bevy_reflect::Reflect;
use bevy_render::{
    extract_component::{ExtractComponent, ExtractComponentPlugin, UniformComponentPlugin},
    render_graph::RenderGraphApp,
    render_resource::*,
    texture::CachedTexture,
    Render, RenderApp, RenderSet,
};

mod node;
pub mod resources;

use node::{FftComputeNode, FftNode};
use resources::{
    prepare_fft_bind_groups, prepare_fft_roots_buffer, prepare_fft_textures, FftBindGroupLayouts,
    FftPipelines, FftRootsBuffer,
};

use crate::complex::c32;

mod shaders {
    use bevy_asset::{weak_handle, Handle};
    use bevy_render::render_resource::Shader;

    pub const FFT: Handle<Shader> = weak_handle!("7f7dd4fd-50bd-40fb-9dd2-d69dc6d5dd40");
    pub const C32: Handle<Shader> = weak_handle!("f9123e70-23a6-4dc3-a9fb-4a02ea636cfb");
    pub const TEXEL: Handle<Shader> = weak_handle!("33f1ccb3-7d87-48d3-8984-51892e6652d0");
    pub const BINDINGS: Handle<Shader> = weak_handle!("1900debb-855d-489b-a973-2559249c3945");
}

/// Component that configures FFT computation parameters
#[derive(Component, Clone, Copy, Reflect, ShaderType)]
#[repr(C)]
pub struct FftSettings {
    /// Size of the FFT texture
    pub size: UVec2,
    /// Number of FFT steps (log2 of size)
    pub orders: u32,
    /// Padding around the source texture
    pub padding: UVec2,
}

#[derive(Component, Clone, Copy, Reflect, ShaderType, ExtractComponent)]
#[repr(C)]
pub struct FftRoots {
    pub roots: [c32; 8192],
}

impl Default for FftSettings {
    fn default() -> Self {
        Self {
            size: UVec2::new(256, 256),
            orders: 8,
            padding: UVec2::ZERO,
        }
    }
}

impl ExtractComponent for FftSettings {
    type QueryData = &'static FftSettings;
    type QueryFilter = ();
    type Out = FftSettings;

    fn extract_component(
        item: bevy_ecs::query::QueryItem<'_, Self::QueryData>,
    ) -> Option<Self::Out> {
        Some(*item)
    }
}

pub struct FftPlugin;

impl Plugin for FftPlugin {
    fn build(&self, app: &mut App) {
        // Load shaders
        load_internal_asset!(app, shaders::FFT, "fft.wgsl", Shader::from_wgsl);
        load_internal_asset!(app, shaders::TEXEL, "texel.wgsl", Shader::from_wgsl);
        load_internal_asset!(app, shaders::BINDINGS, "bindings.wgsl", Shader::from_wgsl);
        load_internal_asset!(app, shaders::C32, "../complex/c32.wgsl", Shader::from_wgsl);
        // load_internal_asset!(app, shaders::IFFT, "ifft.wgsl", Shader::from_wgsl);

        app.register_type::<FftSettings>()
            .register_type::<FftRoots>()
            .add_plugins((
                ExtractComponentPlugin::<FftSettings>::default(),
                UniformComponentPlugin::<FftSettings>::default(),
                ExtractComponentPlugin::<FftRoots>::default(),
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
                    prepare_fft_textures.in_set(RenderSet::PrepareResources),
                    prepare_fft_bind_groups.in_set(RenderSet::PrepareBindGroups),
                    prepare_fft_roots_buffer
                        .in_set(RenderSet::PrepareResources)
                        .before(RenderSet::PrepareBindGroups),
                ),
            )
            .add_render_graph_node::<FftComputeNode>(Core3d, FftNode::ComputeFFT);
    }
}

#[derive(Component)]
pub struct FftTextures {
    pub input: CachedTexture,
    pub output: CachedTexture,
}

#[derive(Component)]
pub struct FftBindGroups {
    pub compute: BindGroup,
}

#[derive(Resource)]
pub struct FftBuffer {
    pub buffer: StorageBuffer<FftSettings>,
}
