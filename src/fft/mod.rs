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
use resources::{prepare_fft_bind_groups, prepare_fft_textures, FftBindGroupLayouts, FftPipelines};

mod shaders {
    use bevy_asset::{weak_handle, Handle};
    use bevy_render::render_resource::Shader;

    pub const FFT: Handle<Shader> = weak_handle!("7f7dd4fd-50bd-40fb-9dd2-d69dc6d5dd40");
    pub const C32: Handle<Shader> = weak_handle!("f9123e70-23a6-4dc3-a9fb-4a02ea636cfb");
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

#[derive(Component, Clone, Copy, Reflect, ShaderType)]
#[repr(C)]
pub struct FftRoots {
    pub roots: [f32; 8192],
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
        load_internal_asset!(app, shaders::C32, "../complex/c32.wgsl", Shader::from_wgsl);
        // load_internal_asset!(app, shaders::IFFT, "ifft.wgsl", Shader::from_wgsl);

        app.register_type::<FftSettings>().add_plugins((
            ExtractComponentPlugin::<FftSettings>::default(),
            UniformComponentPlugin::<FftSettings>::default(),
        ));
    }

    fn finish(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .init_resource::<FftBindGroupLayouts>()
            .init_resource::<FftPipelines>()
            .add_systems(
                Render,
                (
                    prepare_fft_textures.in_set(RenderSet::PrepareResources),
                    prepare_fft_bind_groups.in_set(RenderSet::PrepareBindGroups),
                ),
            )
            .add_render_graph_node::<FftComputeNode>(Core3d, FftNode::ComputeFFT);
    }
}

// Add necessary type definitions for bind groups and pipeline specialization
#[derive(Component)]
pub struct FftTextures {
    pub input: CachedTexture,
    pub output: CachedTexture,
    pub temp: CachedTexture,
}

#[derive(Component)]
pub struct FftBindGroups {
    pub compute: BindGroup,
}

#[derive(Resource)]
pub struct FftBuffer {
    pub buffer: StorageBuffer<FftSettings>,
}
