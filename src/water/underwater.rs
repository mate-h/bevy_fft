//! Waterline mask that splits the water viewer's image into parts above and below the water.
//!
//! The surface material draws the water from either side, but pixels that see no surface still
//! need to know which medium the camera looks through. When the camera sits near the surface,
//! the near plane itself can cut the waves, so part of the image starts under water and part
//! above it. [`WaterUnderwater`] on the viewer camera turns on a compute pass that measures, for
//! every pixel, how far its point on the near plane lies above the displaced surface. The pass
//! also measures the eye itself, and [`WaterWaterline`] carries that result back to the main world
//! for gameplay such as audio or a swimming state.

use bevy::{
    asset::RenderAssetUsages,
    camera::CameraUpdateSystems,
    prelude::*,
    render::{
        RenderApp,
        extract_resource::ExtractResource,
        gpu_readback::{Readback, ReadbackComplete},
        render_asset::RenderAssets,
        render_resource::{binding_types::*, *},
        renderer::{RenderContext, RenderDevice, RenderQueue},
        storage::{GpuShaderBuffer, ShaderBuffer},
        texture::GpuImage,
    },
};

use super::{WaterLodData, WaterLodGpu, WaterLodSlices, WaterQuadtreeBuffer, WaterSurfaceTag};

pub const WATER_WATERLINE_SHADER: &str = "embedded://bevy_fft/water/waterline.wesl";

/// Format of [`WaterWaterline::mask`].
pub const WATER_WATERLINE_FORMAT: TextureFormat = TextureFormat::R32Float;

/// Enables the waterline mask on the water viewer. Only one active camera is measured.
#[derive(Component, Clone, Copy, Debug, Default, Reflect)]
#[require(super::WaterViewer)]
pub struct WaterUnderwater;

/// Where the water viewer meets the water surface.
///
/// `mask` matches the viewer's render target in size. Each texel holds the height in meters of
/// that pixel's point on the near plane above the displaced water surface, so negative texels look
/// out from under water and the zero crossing is the waterline. It is rewritten every frame before
/// the cameras draw.
#[derive(Resource, Clone)]
pub struct WaterWaterline {
    pub mask: Handle<Image>,
    /// Height of the eye above the water surface below it, in meters. Negative when the camera is
    /// under water. Arrives one to three frames late through a readback, and is `None` before the
    /// first one.
    pub eye_height: Option<f32>,
    /// Height of the water surface below the eye, in meters, from the same readback.
    pub surface_height: Option<f32>,
    eye: Handle<ShaderBuffer>,
}

impl WaterWaterline {
    /// Whether the last readback found the eye below the displaced water surface.
    pub fn camera_below_surface(&self) -> bool {
        self.eye_height.is_some_and(|h| h < 0.0)
    }
}

/// Layout matches `WaterlineUniform` in `waterline.wesl`.
#[derive(Clone, Copy, Default, ShaderType)]
pub struct WaterlineUniform {
    pub world_from_clip: Mat4,
    /// Eye position, then the height of the water mesh plane.
    pub eye: Vec4,
    pub size: UVec2,
    pub _pad: UVec2,
}

/// Render-world copy of what the waterline pass needs this frame.
#[derive(Resource, Clone, Default, ExtractResource)]
#[extract_app(RenderApp)]
pub struct WaterWaterlineFrame {
    enabled: bool,
    uniform: WaterlineUniform,
    mask: Handle<Image>,
    eye: Handle<ShaderBuffer>,
}

/// Marks the entity that carries the eye height [`Readback`].
#[derive(Component)]
pub(super) struct WaterlineReadback;

fn mask_image(size: Extent3d) -> Image {
    let mut image = Image::new_uninit(
        size,
        TextureDimension::D2,
        WATER_WATERLINE_FORMAT,
        RenderAssetUsages::RENDER_WORLD,
    );
    image.texture_descriptor.usage =
        TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_SRC;
    image
}

pub(super) fn create_waterline(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    mut buffers: ResMut<Assets<ShaderBuffer>>,
) {
    let eye = buffers.add(ShaderBuffer::with_size(16, RenderAssetUsages::RENDER_WORLD));
    commands
        .spawn((WaterlineReadback, Name::new("Waterline readback")))
        .observe(
            |event: On<ReadbackComplete>, mut waterline: ResMut<WaterWaterline>| {
                let lane = |i: usize| {
                    event
                        .data
                        .get(i * 4..i * 4 + 4)
                        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                };
                waterline.eye_height = lane(0);
                waterline.surface_height = lane(1);
            },
        );
    commands.insert_resource(WaterWaterline {
        mask: images.add(mask_image(Extent3d::default())),
        eye_height: None,
        surface_height: None,
        eye,
    });
}

/// Sizes the mask to the viewer and gathers this frame's camera and water plane.
#[allow(clippy::type_complexity)]
pub(super) fn update_waterline(
    mut commands: Commands,
    cameras: Query<(&Camera, &GlobalTransform), With<WaterUnderwater>>,
    surfaces: Query<(&GlobalTransform, Has<WaterQuadtreeBuffer>), With<WaterSurfaceTag>>,
    readback: Query<(Entity, Has<Readback>), With<WaterlineReadback>>,
    mut waterline: ResMut<WaterWaterline>,
    mut frame: ResMut<WaterWaterlineFrame>,
    mut images: ResMut<Assets<Image>>,
) {
    frame.enabled = false;
    let camera = cameras.iter().find(|(camera, _)| camera.is_active);
    let size = camera.and_then(|(camera, _)| camera.physical_target_size());
    let Ok((readback_entity, has_readback)) = readback.single() else {
        return;
    };
    let (Some((camera, transform)), Some(size)) = (camera, size) else {
        if has_readback {
            commands.entity(readback_entity).remove::<Readback>();
        }
        waterline.eye_height = None;
        waterline.surface_height = None;
        return;
    };
    let extent = Extent3d {
        width: size.x.max(1),
        height: size.y.max(1),
        depth_or_array_layers: 1,
    };
    if images
        .get(&waterline.mask)
        .is_some_and(|image| image.texture_descriptor.size != extent)
    {
        // A new image rather than a resize, since the mask carries no data on the CPU.
        waterline.mask = images.add(mask_image(extent));
    }
    // Quadtree tiles always lie on the plane y = 0. Other meshes displace from their own height.
    let plane = surfaces.iter().next().map_or(
        0.0,
        |(t, quadtree)| if quadtree { 0.0 } else { t.translation().y },
    );
    let eye = transform.translation();
    frame.enabled = true;
    frame.uniform = WaterlineUniform {
        world_from_clip: transform.to_matrix() * camera.clip_from_view().inverse(),
        eye: eye.extend(plane),
        size: UVec2::new(extent.width, extent.height),
        _pad: UVec2::ZERO,
    };
    frame.mask = waterline.mask.clone();
    frame.eye = waterline.eye.clone();
    if !has_readback {
        commands
            .entity(readback_entity)
            .insert(Readback::buffer(waterline.eye.clone()));
    }
}

#[derive(Resource)]
pub struct WaterWaterlinePipeline {
    layout: BindGroupLayoutDescriptor,
    pipeline: CachedComputePipelineId,
}

impl FromWorld for WaterWaterlinePipeline {
    fn from_world(world: &mut World) -> Self {
        let pipeline_cache = world.resource::<PipelineCache>();
        let asset_server = world.resource::<AssetServer>();
        let entries = BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                uniform_buffer::<WaterlineUniform>(false),
                uniform_buffer::<WaterLodSlices>(false),
                texture_2d_array(TextureSampleType::Float { filterable: true }),
                texture_2d_array(TextureSampleType::Float { filterable: true }),
                sampler(SamplerBindingType::Filtering),
                texture_storage_2d(WATER_WATERLINE_FORMAT, StorageTextureAccess::WriteOnly),
                storage_buffer_sized(false, None),
            ),
        );
        let layout = BindGroupLayoutDescriptor::new("water_waterline_layout", &entries);
        let pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("water_waterline".into()),
            layout: vec![layout.clone()],
            immediate_size: 0,
            shader: asset_server.load(WATER_WATERLINE_SHADER),
            shader_defs: vec![],
            entry_point: Some("waterline_mask".into()),
            zero_initialize_workgroup_memory: false,
            constants: vec![],
        });
        Self { layout, pipeline }
    }
}

#[derive(Resource, Default)]
pub struct WaterWaterlineGpu {
    uniform: UniformBuffer<WaterlineUniform>,
    group: Option<BindGroup>,
    dispatch: UVec2,
}

#[allow(clippy::too_many_arguments)]
pub(super) fn prepare_waterline(
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    pipeline_cache: Res<PipelineCache>,
    pipeline: Res<WaterWaterlinePipeline>,
    frame: Res<WaterWaterlineFrame>,
    lod: Option<Res<WaterLodData>>,
    lod_gpu: Res<WaterLodGpu>,
    images: Res<RenderAssets<GpuImage>>,
    buffers: Res<RenderAssets<GpuShaderBuffer>>,
    mut gpu: ResMut<WaterWaterlineGpu>,
) {
    gpu.group = None;
    let Some(lod) = lod else {
        return;
    };
    if !frame.enabled {
        return;
    }
    let (Some(mask), Some(eye), Some(displacement), Some(foam), Some(slices)) = (
        images.get(&frame.mask),
        buffers.get(&frame.eye),
        images.get(&lod.displacement),
        images.get(&lod.foam),
        lod_gpu.slices.binding(),
    ) else {
        return;
    };
    gpu.uniform.set(frame.uniform);
    gpu.uniform.write_buffer(&device, &queue);
    let Some(uniform) = gpu.uniform.binding() else {
        return;
    };
    let layout = pipeline_cache.get_bind_group_layout(&pipeline.layout);
    gpu.group = Some(device.create_bind_group(
        "water_waterline_bind_group",
        &layout,
        &BindGroupEntries::sequential((
            uniform,
            slices,
            &displacement.texture_view,
            &foam.texture_view,
            &displacement.sampler,
            &mask.texture_view,
            eye.buffer.as_entire_binding(),
        )),
    ));
    gpu.dispatch = frame.uniform.size.map(|n| n.div_ceil(8));
}

pub(super) fn run_waterline(
    mut ctx: RenderContext,
    pipeline: Res<WaterWaterlinePipeline>,
    cache: Res<PipelineCache>,
    gpu: Res<WaterWaterlineGpu>,
) {
    let Some(group) = &gpu.group else {
        return;
    };
    let Some(compute) = cache.get_compute_pipeline(pipeline.pipeline) else {
        return;
    };
    let mut pass = ctx
        .command_encoder()
        .begin_compute_pass(&ComputePassDescriptor {
            label: Some("water_waterline_pass"),
            timestamp_writes: None,
        });
    pass.set_pipeline(compute);
    pass.set_bind_group(0, group, &[]);
    pass.dispatch_workgroups(gpu.dispatch.x, gpu.dispatch.y, 1);
}

pub(super) fn configure(app: &mut App) {
    bevy::asset::embedded_asset!(app, "waterline.wesl");
    app.register_type::<WaterUnderwater>()
        .init_resource::<WaterWaterlineFrame>()
        .add_plugins(bevy::render::extract_resource::ExtractResourcePlugin::<
            WaterWaterlineFrame,
        >::default())
        .add_systems(PreStartup, create_waterline)
        .add_systems(
            PostUpdate,
            update_waterline
                .after(CameraUpdateSystems)
                .after(bevy::transform::TransformSystems::Propagate),
        );
    if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
        render_app.init_resource::<WaterWaterlineGpu>().add_systems(
            bevy::render::Render,
            prepare_waterline.in_set(bevy::render::RenderSystems::PrepareBindGroups),
        );
        render_app.add_systems(
            bevy::render::renderer::RenderGraph,
            run_waterline.in_set(super::WaterLodGraphSystems::Read),
        );
    }
}

pub(super) fn finish(app: &mut App) {
    if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
        render_app.init_resource::<WaterWaterlinePipeline>();
    }
}
