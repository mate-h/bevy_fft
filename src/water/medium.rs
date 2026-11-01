//! The water body as a participating medium for the water viewer, after Monzon et al. (2024).
//!
//! When [`super::WaterUnderwater`] is on the viewer, a fullscreen pass runs after the main pass and
//! before tonemapping. Every pixel whose near-plane point lies under water, by the waterline mask,
//! fades with the extinction of [`WaterMedium`] along its ray and gains the multiple-scattered
//! light of Monzon et al. Eq. 9. Scene points under water are dimmed as the downwelling light that
//! reaches their depth, the `exp(-K_d y)` factor of Eq. 10. The water surface draws its own
//! underside, so pixels on the surface are left alone. The pass binds the view's mesh bind groups,
//! as Bevy's deferred lighting does, so it lights the medium with the same sun and sky as the
//! surface.

use bevy::{
    camera::Camera3dDepthTextureUsage,
    core_pipeline::{
        FullscreenShader,
        schedule::{Core3d, Core3dSystems},
        tonemapping::tonemapping,
    },
    pbr::{MeshPipeline, MeshPipelineKey, MeshPipelineSystems, MeshViewBindGroup, ViewKeyCache},
    prelude::*,
    render::{
        Render, RenderApp, RenderStartup, RenderSystems,
        camera::ExtractedCamera,
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        render_asset::RenderAssets,
        render_resource::{binding_types::*, *},
        renderer::{RenderContext, RenderDevice, RenderQueue, ViewQuery},
        texture::GpuImage,
        view::{ExtractedView, ViewDepthStencilTexture, ViewTarget},
    },
    shader::ShaderDefVal,
};

use super::{
    WaterLodData, WaterLodGpu, WaterLodSlices, WaterMediumUniform, WaterUnderwater,
    underwater::WaterWaterlineFrame,
};

pub const WATER_MEDIUM_SHADER: &str = "embedded://bevy_fft/water/medium.wesl";

/// The water the viewer looks through from under the surface. [`super::WaterPlugin`] copies it
/// into every water surface material, so the underside and the medium pass agree.
#[derive(Resource, Clone, Copy, Debug, Default, Reflect, ExtractResource)]
#[extract_app(RenderApp)]
pub struct WaterMedium(pub WaterMediumUniform);

/// Lets the medium pass sample the viewer's depth buffer.
pub(super) fn bind_viewer_depth(mut cameras: Query<&mut Camera3d, With<WaterUnderwater>>) {
    for mut camera in &mut cameras {
        let usages = TextureUsages::from(camera.depth_texture_usages);
        if !usages.contains(TextureUsages::TEXTURE_BINDING) {
            camera.depth_texture_usages =
                Camera3dDepthTextureUsage::from(usages | TextureUsages::TEXTURE_BINDING);
        }
    }
}

#[derive(Resource)]
pub struct WaterMediumLayout {
    mesh_pipeline: MeshPipeline,
    empty: BindGroupLayoutDescriptor,
    layout: BindGroupLayoutDescriptor,
    shader: Handle<Shader>,
    fullscreen: FullscreenShader,
}

pub(super) fn init_water_medium_layout(
    mut commands: Commands,
    mesh_pipeline: Res<MeshPipeline>,
    fullscreen: Res<FullscreenShader>,
    asset_server: Res<AssetServer>,
) {
    let unfiltered = TextureSampleType::Float { filterable: false };
    let layout = BindGroupLayoutDescriptor::new(
        "water_medium_layout",
        &BindGroupLayoutEntries::with_indices(
            ShaderStages::FRAGMENT,
            (
                (0, texture_2d(unfiltered)),
                (1, texture_depth_2d()),
                (2, texture_2d(unfiltered)),
                (3, uniform_buffer::<WaterLodSlices>(false)),
                (
                    4,
                    texture_2d_array(TextureSampleType::Float { filterable: true }),
                ),
                (
                    5,
                    texture_2d_array(TextureSampleType::Float { filterable: true }),
                ),
                (6, sampler(SamplerBindingType::Filtering)),
                (7, uniform_buffer::<Vec4>(false)),
                (112, uniform_buffer::<WaterMediumUniform>(false)),
            ),
        ),
    );
    commands.insert_resource(WaterMediumLayout {
        mesh_pipeline: mesh_pipeline.clone(),
        empty: BindGroupLayoutDescriptor::new("water_medium_empty_layout", &[]),
        layout,
        shader: asset_server.load(WATER_MEDIUM_SHADER),
        fullscreen: fullscreen.clone(),
    });
}

impl SpecializedRenderPipeline for WaterMediumLayout {
    type Key = MeshPipelineKey;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        // The view bindings the shader declares must match the layout this key selects, so the
        // defs follow Bevy's deferred lighting pass.
        let mut shader_defs: Vec<ShaderDefVal> = vec![ShaderDefVal::UInt(
            "MATERIAL_BIND_GROUP".into(),
            bevy::pbr::MATERIAL_BIND_GROUP_INDEX as u32,
        )];
        let flags = [
            (
                MeshPipelineKey::SCREEN_SPACE_AMBIENT_OCCLUSION,
                "SCREEN_SPACE_AMBIENT_OCCLUSION",
            ),
            (MeshPipelineKey::ENVIRONMENT_MAP, "ENVIRONMENT_MAP"),
            (MeshPipelineKey::IRRADIANCE_VOLUME, "IRRADIANCE_VOLUME"),
            (MeshPipelineKey::NORMAL_PREPASS, "NORMAL_PREPASS"),
            (MeshPipelineKey::DEPTH_PREPASS, "DEPTH_PREPASS"),
            (
                MeshPipelineKey::MOTION_VECTOR_PREPASS,
                "MOTION_VECTOR_PREPASS",
            ),
            (MeshPipelineKey::DEFERRED_PREPASS, "DEFERRED_PREPASS"),
            (
                MeshPipelineKey::SCREEN_SPACE_REFLECTIONS,
                "SCREEN_SPACE_REFLECTIONS",
            ),
            (MeshPipelineKey::CONTACT_SHADOWS, "CONTACT_SHADOWS"),
            (MeshPipelineKey::DISTANCE_FOG, "DISTANCE_FOG"),
            (MeshPipelineKey::ATMOSPHERE, "ATMOSPHERE"),
            (MeshPipelineKey::OIT_ENABLED, "OIT_ENABLED"),
            (MeshPipelineKey::TONEMAP_IN_SHADER, "TONEMAP_IN_SHADER"),
        ];
        for (flag, def) in flags {
            if key.contains(flag) {
                shader_defs.push(def.into());
            }
        }
        if key.msaa_samples() > 1 {
            shader_defs.push("MULTISAMPLED".into());
        }
        if cfg!(not(target_arch = "wasm32")) {
            shader_defs.push("IRRADIANCE_VOLUMES_ARE_USABLE".into());
        }
        if self.mesh_pipeline.binding_arrays_are_usable {
            shader_defs.push("MULTIPLE_LIGHT_PROBES_IN_ARRAY".into());
            shader_defs.push("MULTIPLE_LIGHTMAPS_IN_ARRAY".into());
        }
        if self.mesh_pipeline.clustered_decals_are_usable {
            shader_defs.push("CLUSTERED_DECALS_ARE_USABLE".into());
        }
        let view = self.mesh_pipeline.get_view_layout(key.into());
        RenderPipelineDescriptor {
            label: Some("water_medium_pipeline".into()),
            layout: vec![
                view.main_layout,
                view.binding_array_layout,
                self.empty.clone(),
                self.layout.clone(),
            ],
            vertex: self.fullscreen.to_vertex_state(),
            fragment: Some(FragmentState {
                shader: self.shader.clone(),
                shader_defs,
                entry_point: Some("water_medium".into()),
                targets: vec![Some(ColorTargetState {
                    format: key.target_format(),
                    blend: None,
                    write_mask: ColorWrites::ALL,
                })],
                ..default()
            }),
            ..default()
        }
    }
}

#[derive(Component)]
pub struct WaterMediumPipeline(CachedRenderPipelineId);

#[allow(clippy::type_complexity)]
pub(super) fn prepare_water_medium_pipelines(
    mut commands: Commands,
    pipeline_cache: Res<PipelineCache>,
    view_key_cache: Res<ViewKeyCache>,
    mut pipelines: ResMut<SpecializedRenderPipelines<WaterMediumLayout>>,
    layout: Res<WaterMediumLayout>,
    views: Query<(Entity, &ExtractedView), (With<ExtractedCamera>, With<WaterUnderwater>)>,
) {
    for (entity, view) in &views {
        let Some(key) = view_key_cache.get(&view.retained_view_entity) else {
            continue;
        };
        let id = pipelines.specialize(&pipeline_cache, &layout, *key);
        commands.entity(entity).insert(WaterMediumPipeline(id));
    }
}

#[derive(Resource, Default)]
pub struct WaterMediumGpu {
    medium: UniformBuffer<WaterMediumUniform>,
    plane: UniformBuffer<Vec4>,
}

pub(super) fn prepare_water_medium(
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    medium: Option<Res<WaterMedium>>,
    frame: Res<WaterWaterlineFrame>,
    mut gpu: ResMut<WaterMediumGpu>,
) {
    gpu.medium.set(medium.map_or_else(default, |m| m.0));
    gpu.medium.write_buffer(&device, &queue);
    gpu.plane.set(Vec4::new(frame.plane(), 0.0, 0.0, 0.0));
    gpu.plane.write_buffer(&device, &queue);
}

#[allow(clippy::too_many_arguments)]
pub(super) fn water_medium(
    view: ViewQuery<(
        &ExtractedView,
        &MeshViewBindGroup,
        &ViewTarget,
        &ViewDepthStencilTexture,
        &WaterMediumPipeline,
    )>,
    pipeline_cache: Res<PipelineCache>,
    layout: Res<WaterMediumLayout>,
    frame: Res<WaterWaterlineFrame>,
    gpu: Res<WaterMediumGpu>,
    lod: Option<Res<WaterLodData>>,
    lod_gpu: Res<WaterLodGpu>,
    images: Res<RenderAssets<GpuImage>>,
    mut ctx: RenderContext,
) {
    let (view, mesh_view, target, depth, pipeline) = view.into_inner();
    if frame.camera() != Some(view.retained_view_entity.main_entity) {
        return;
    }
    let Some(render_pipeline) = pipeline_cache.get_render_pipeline(pipeline.0) else {
        return;
    };
    let Some(lod) = lod else {
        return;
    };
    let (Some(mask), Some(displacement), Some(foam)) = (
        images.get(frame.mask()),
        images.get(&lod.displacement),
        images.get(&lod.foam),
    ) else {
        return;
    };
    let (Some(slices), Some(medium), Some(plane)) = (
        lod_gpu.slices.binding(),
        gpu.medium.binding(),
        gpu.plane.binding(),
    ) else {
        return;
    };
    let Some(depth_view) = depth.attachment.depth_stencil_views().depth_only_view() else {
        return;
    };
    let depth_texture = depth.texture();
    if depth_texture.sample_count() != 1
        || !depth_texture
            .usage()
            .contains(TextureUsages::TEXTURE_BINDING)
        || mask.texture.size() != depth_texture.size()
    {
        return;
    }
    let post_process = target.post_process_write();
    let device = ctx.render_device().clone();
    let empty = device.create_bind_group(
        "water_medium_empty",
        &pipeline_cache.get_bind_group_layout(&layout.empty),
        &[],
    );
    let group = device.create_bind_group(
        "water_medium_bind_group",
        &pipeline_cache.get_bind_group_layout(&layout.layout),
        &BindGroupEntries::with_indices((
            (0, post_process.source),
            (1, depth_view),
            (2, &mask.texture_view),
            (3, slices),
            (4, &displacement.texture_view),
            (5, &foam.texture_view),
            (6, &displacement.sampler),
            (7, plane),
            (112, medium),
        )),
    );
    let mut pass = ctx.begin_tracked_render_pass(RenderPassDescriptor {
        label: Some("water_medium"),
        color_attachments: &[Some(RenderPassColorAttachment {
            view: post_process.destination,
            depth_slice: None,
            resolve_target: None,
            ops: Operations::default(),
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
    pass.set_render_pipeline(render_pipeline);
    pass.set_bind_group(0, &mesh_view.main, &mesh_view.main_offsets);
    pass.set_bind_group(1, &mesh_view.binding_array, &[]);
    pass.set_bind_group(2, &empty, &[]);
    pass.set_bind_group(3, &group, &[]);
    pass.draw(0..3, 0..1);
}

pub(super) fn configure(app: &mut App) {
    bevy::asset::embedded_asset!(app, "medium.wesl");
    app.register_type::<WaterMedium>()
        .init_resource::<WaterMedium>()
        .add_plugins(ExtractResourcePlugin::<WaterMedium>::default())
        .add_systems(PostUpdate, bind_viewer_depth);
    if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
        render_app
            .init_resource::<WaterMediumGpu>()
            .init_resource::<SpecializedRenderPipelines<WaterMediumLayout>>()
            .add_systems(
                RenderStartup,
                init_water_medium_layout.after(MeshPipelineSystems),
            )
            .add_systems(
                Render,
                (
                    prepare_water_medium_pipelines.in_set(RenderSystems::Prepare),
                    prepare_water_medium.in_set(RenderSystems::PrepareResources),
                ),
            )
            .add_systems(
                Core3d,
                water_medium
                    .before(tonemapping)
                    .in_set(Core3dSystems::PostProcess),
            );
    }
}
