//! Opaque-only depth for the water viewer, captured before transmissive water draws.

use bevy::{
    camera::{CameraUpdateSystems, Hdr},
    core_pipeline::{core_3d::main_opaque_pass_3d, prepass::DepthPrepass},
    pbr::{ScreenSpaceTransmission, main_transmissive_pass_3d},
    prelude::*,
    render::{
        RenderApp,
        extract_resource::ExtractResource,
        render_asset::RenderAssets,
        render_resource::{Extent3d, TextureDimension, TextureFormat, TextureUsages},
        renderer::{RenderContext, ViewQuery},
        texture::GpuImage,
        view::{ExtractedView, ViewDepthStencilTexture},
    },
};

use super::WaterViewer;

/// Enables terrain-aware transmission on the water viewer. Use with `Msaa::Off`.
/// Water materials opt in separately through [`super::WaterOpticsUniform::enabled`].
/// The shared water data and depth snapshot belong to this viewer.
#[derive(Component, Default)]
#[require(WaterViewer, Hdr, DepthPrepass, ScreenSpaceTransmission)]
#[require(Msaa = Msaa::Off)]
pub struct WaterTransmission;

#[derive(Resource, Clone, ExtractResource)]
#[extract_app(RenderApp)]
pub(super) struct WaterSceneDepth {
    pub image: Handle<Image>,
    camera: Option<Entity>,
}

pub(super) fn create_scene_depth(mut commands: Commands, mut images: ResMut<Assets<Image>>) {
    let mut image = Image::new_uninit(
        Extent3d::default(),
        TextureDimension::D2,
        TextureFormat::Depth32Float,
        default(),
    );
    image.texture_descriptor.usage = TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST;
    commands.insert_resource(WaterSceneDepth {
        image: images.add(image),
        camera: None,
    });
}

pub(super) fn resize_scene_depth(
    cameras: Query<(Entity, &Camera, &Msaa), With<WaterTransmission>>,
    mut snapshot: ResMut<WaterSceneDepth>,
    mut images: ResMut<Assets<Image>>,
) {
    snapshot.camera = None;
    let Some((entity, camera, msaa)) = cameras.iter().find(|(_, camera, _)| camera.is_active)
    else {
        return;
    };
    if *msaa != Msaa::Off {
        return;
    }
    let Some(size) = camera.physical_target_size() else {
        return;
    };
    let Some(image) = images.get(&snapshot.image) else {
        return;
    };
    let size = Extent3d {
        width: size.x.max(1),
        height: size.y.max(1),
        depth_or_array_layers: 1,
    };
    if image.texture_descriptor.size != size {
        images
            .get_mut(&snapshot.image)
            .unwrap()
            .texture_descriptor
            .size = size;
    }
    snapshot.camera = Some(entity);
}

pub(super) fn copy_scene_depth(
    view: ViewQuery<(&ExtractedView, &ViewDepthStencilTexture)>,
    snapshot: Res<WaterSceneDepth>,
    images: Res<RenderAssets<GpuImage>>,
    mut ctx: RenderContext,
) {
    let (view, depth) = view.into_inner();
    let Some(camera) = snapshot.camera else {
        return;
    };
    if view.retained_view_entity.main_entity != camera.into() {
        return;
    }
    let Some(target) = images.get(&snapshot.image) else {
        return;
    };
    let source = depth.texture();
    if source.size() != target.texture.size()
        || source.sample_count() != 1
        || !source.usage().contains(TextureUsages::COPY_SRC)
    {
        return;
    }
    ctx.command_encoder().copy_texture_to_texture(
        source.as_image_copy(),
        target.texture.as_image_copy(),
        source.size(),
    );
}

pub(super) fn configure(app: &mut App) {
    app.add_plugins(bevy::render::extract_resource::ExtractResourcePlugin::<
        WaterSceneDepth,
    >::default())
        .add_systems(PreStartup, create_scene_depth)
        .add_systems(PostUpdate, resize_scene_depth.after(CameraUpdateSystems));
    if let Some(render) = app.get_sub_app_mut(RenderApp) {
        render.add_systems(
            bevy::core_pipeline::schedule::Core3d,
            copy_scene_depth
                .after(main_opaque_pass_3d)
                .before(main_transmissive_pass_3d)
                .in_set(bevy::core_pipeline::schedule::Core3dSystems::MainPass),
        );
    }
}
