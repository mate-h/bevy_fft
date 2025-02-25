use bevy::core_pipeline::core_2d::graph::Core2d;
use bevy::core_pipeline::core_3d::graph::Core3d;
use bevy_app::{App, Plugin, Startup};
use bevy_asset::{load_internal_asset, Handle};
use bevy_ecs::query::QueryItem;
use bevy_ecs::schedule::IntoSystemConfigs;
use bevy_ecs::system::lifetimeless::Read;
use bevy_ecs::system::{Commands, Query};
use bevy_ecs::world::{FromWorld, World};
use bevy_ecs::{component::Component, resource::Resource};
use bevy_image::Image;
use bevy_math::UVec2;
use bevy_reflect::Reflect;
use bevy_render::extract_resource::{ExtractResource, ExtractResourcePlugin};
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

// Public-facing component
#[derive(Component, Clone, Reflect)]
pub struct FftSource {
    /// The complex image to transform
    pub image: Handle<Image>,
    /// Size of the FFT texture
    pub size: UVec2,
    /// Number of FFT steps (log2 of size)
    pub orders: u32,
    /// Padding around the source texture
    pub padding: UVec2,
    /// Roots of the FFT
    pub roots: [c32; 8192],
}

impl Default for FftSource {
    fn default() -> Self {
        Self {
            image: Handle::default(),
            size: UVec2::new(256, 256),
            orders: 8,
            padding: UVec2::ZERO,
            roots: [c32::new(0.0, 0.0); 8192],
        }
    }
}

// Internal render-world component
#[derive(Component, Clone, Copy, Reflect, ShaderType)]
#[repr(C)]
pub struct FftSettings {
    pub size: UVec2,
    pub orders: u32,
    pub padding: UVec2,
}

impl ExtractComponent for FftSettings {
    type QueryData = Read<FftSource>;
    type QueryFilter = ();
    type Out = FftSettings;

    fn extract_component(item: QueryItem<'_, Self::QueryData>) -> Option<Self::Out> {
        Some(FftSettings {
            size: item.size,
            orders: item.orders,
            padding: item.padding,
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

    fn extract_component(item: QueryItem<'_, Self::QueryData>) -> Option<Self::Out> {
        Some(FftRoots { roots: item.roots })
    }
}

#[derive(Component, Clone)]
pub struct FftSourceImage(pub Handle<Image>);

impl ExtractComponent for FftSourceImage {
    type QueryData = Read<FftSource>;
    type QueryFilter = ();
    type Out = FftSourceImage;

    fn extract_component(item: QueryItem<'_, Self::QueryData>) -> Option<Self::Out> {
        Some(FftSourceImage(item.image.clone()))
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

        app.register_type::<FftSource>()
            .register_type::<FftRoots>()
            .add_plugins((
                ExtractComponentPlugin::<FftSettings>::default(),
                UniformComponentPlugin::<FftSettings>::default(),
                ExtractComponentPlugin::<FftRoots>::default(),
                ExtractComponentPlugin::<FftSourceImage>::default(),
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
            .add_render_graph_node::<FftComputeNode>(Core2d, FftNode::ComputeFFT);
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
