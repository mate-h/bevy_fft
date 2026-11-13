//! Class-based coastal materials and persistent sand wetness.

use bevy::{
    image::{
        ImageAddressMode, ImageFilterMode, ImageLoaderSettings, ImageSampler,
        ImageSamplerDescriptor,
    },
    pbr::Material,
    prelude::*,
    render::{
        GpuResourceAppExt, Render, RenderApp, RenderSystems,
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        render_asset::RenderAssets,
        render_resource::{binding_types::*, *},
        renderer::{RenderContext, RenderDevice, RenderGraph, RenderGraphSystems, RenderQueue},
        storage::{GpuShaderBuffer, ShaderBuffer},
        texture::GpuImage,
    },
    shader::ShaderRef,
};
use bevy_fft::shallow_water::{ShallowWaterController, run_shallow_water_sim};

const ROOT: &str = "beach_materials";

#[derive(Clone, Copy, ShaderType)]
pub struct GroundSettings {
    /// Repeats in meters from Poly Haven's scanned dimensions, in texture-layer order.
    pub repeat_a: Vec4,
    pub repeat_b: Vec4,
    pub domain: Vec4,
}

#[derive(Asset, AsBindGroup, TypePath, Clone)]
pub struct KauaiGroundMaterial {
    #[texture(0, dimension = "2d_array")]
    #[sampler(1)]
    pub albedo: Handle<Image>,
    #[texture(2, dimension = "2d_array")]
    pub normal_roughness: Handle<Image>,
    #[texture(3)]
    pub classes: Handle<Image>,
    #[storage(4, read_only)]
    pub wetness: Handle<ShaderBuffer>,
    #[uniform(5)]
    pub settings: GroundSettings,
}

impl Material for KauaiGroundMaterial {
    fn fragment_shader() -> ShaderRef {
        "examples/kauai_ground.wesl".into()
    }
}

pub fn load_ground(
    commands: &mut Commands,
    server: &AssetServer,
    buffers: &mut Assets<ShaderBuffer>,
) -> KauaiGroundMaterial {
    let wetness = buffers.add(ShaderBuffer::new(
        vec![0.0_f32; (super::GRID * super::GRID) as usize],
        default(),
    ));
    let baseline = server
        .load_builder()
        .with_settings(|settings: &mut ImageLoaderSettings| {
            settings.is_srgb = false;
            settings.expand_grayscale = false;
        })
        .load(format!("{ROOT}/sand_capillary.png"));
    commands.insert_resource(CoastWetnessFrame {
        wetness: wetness.clone(),
        baseline,
        params: WetnessParams {
            grid: UVec4::new(super::GRID, super::GRID, 0, 0),
            timing: Vec4::new(0.0, 120.0, 0.0, 0.0),
        },
    });
    let albedo = server
        .load_builder()
        .with_settings(|settings: &mut ImageLoaderSettings| {
            settings.sampler = ImageSampler::Descriptor(ImageSamplerDescriptor {
                address_mode_u: ImageAddressMode::Repeat,
                address_mode_v: ImageAddressMode::Repeat,
                mag_filter: ImageFilterMode::Linear,
                min_filter: ImageFilterMode::Linear,
                mipmap_filter: ImageFilterMode::Linear,
                anisotropy_clamp: 16,
                ..default()
            });
        })
        .load(format!("{ROOT}/albedo.dds"));
    let normal_roughness = server
        .load_builder()
        .with_settings(|settings: &mut ImageLoaderSettings| {
            settings.is_srgb = false;
        })
        .load(format!("{ROOT}/normal_roughness.dds"));
    let classes = server
        .load_builder()
        .with_settings(|settings: &mut ImageLoaderSettings| {
            settings.is_srgb = false;
            settings.expand_grayscale = false;
        })
        .load(format!("{ROOT}/surface_classes.png"));
    KauaiGroundMaterial {
        albedo,
        normal_roughness,
        classes,
        wetness,
        settings: GroundSettings {
            repeat_a: Vec4::new(1.5, 1.8, 2.51, 2.0),
            repeat_b: Vec4::new(3.0, 1.3, 2.0, 3.0),
            domain: Vec4::new(
                super::TERRAIN_SIDE,
                super::GRID as f32,
                match std::env::var("BEVY_FFT_GROUND_VIEW").as_deref() {
                    Ok("classes") => 1.0,
                    Ok("wetness") => 2.0,
                    _ => 0.0,
                },
                0.0,
            ),
        },
    }
}

#[derive(Clone, Copy, Default, ShaderType)]
struct WetnessParams {
    grid: UVec4,
    timing: Vec4,
}

#[derive(Resource, Clone, ExtractResource)]
#[extract_app(RenderApp)]
struct CoastWetnessFrame {
    wetness: Handle<ShaderBuffer>,
    baseline: Handle<Image>,
    params: WetnessParams,
}

pub struct CoastGroundPlugin;
impl Plugin for CoastGroundPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(ExtractResourcePlugin::<CoastWetnessFrame>::default())
            .add_systems(PostUpdate, update_wetness_time);
        if let Some(render) = app.get_sub_app_mut(RenderApp) {
            render
                .init_gpu_resource::<WetnessPipeline>()
                .init_resource::<WetnessGroup>()
                .add_systems(
                    Render,
                    prepare_wetness.in_set(RenderSystems::PrepareBindGroups),
                )
                .add_systems(
                    RenderGraph,
                    run_wetness
                        .after(run_shallow_water_sim)
                        .before(bevy::core_pipeline::schedule::camera_driver)
                        .in_set(RenderGraphSystems::Render),
                );
        }
    }
}

fn update_wetness_time(
    time: Res<Time>,
    controller: Option<Res<ShallowWaterController>>,
    frame: Option<ResMut<CoastWetnessFrame>>,
) {
    let (Some(controller), Some(mut frame)) = (controller, frame) else {
        return;
    };
    frame.params.timing.x = if controller.paused {
        0.0
    } else {
        time.delta_secs().min(0.1)
    };
}

#[derive(Resource)]
struct WetnessPipeline {
    layout: BindGroupLayoutDescriptor,
    pipeline: CachedComputePipelineId,
}
impl FromWorld for WetnessPipeline {
    fn from_world(world: &mut World) -> Self {
        let cache = world.resource::<PipelineCache>();
        let layout = BindGroupLayoutDescriptor::new(
            "beach_wetness",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (
                    uniform_buffer::<WetnessParams>(false),
                    texture_2d(TextureSampleType::Float { filterable: false }),
                    texture_2d(TextureSampleType::Float { filterable: false }),
                    storage_buffer_sized(false, None),
                ),
            ),
        );
        let pipeline = cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("beach_wetness".into()),
            layout: vec![layout.clone()],
            shader: world
                .resource::<AssetServer>()
                .load("examples/beach_wetness.wesl"),
            entry_point: Some("update_wetness".into()),
            ..default()
        });
        Self { layout, pipeline }
    }
}

#[derive(Resource, Default)]
struct WetnessGroup {
    group: Option<BindGroup>,
    uniform: UniformBuffer<WetnessParams>,
    dispatch: UVec2,
}

#[allow(clippy::too_many_arguments)]
fn prepare_wetness(
    frame: Option<Res<CoastWetnessFrame>>,
    controller: Option<Res<ShallowWaterController>>,
    pipeline: Res<WetnessPipeline>,
    cache: Res<PipelineCache>,
    images: Res<RenderAssets<GpuImage>>,
    buffers: Res<RenderAssets<GpuShaderBuffer>>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    mut state: ResMut<WetnessGroup>,
) {
    state.group = None;
    let (Some(frame), Some(controller)) = (frame, controller) else {
        return;
    };
    let (Some(bed), Some(baseline), Some(wetness)) = (
        images.get(&controller.bed_water),
        images.get(&frame.baseline),
        buffers.get(&frame.wetness),
    ) else {
        return;
    };
    state.uniform.set(frame.params);
    state.uniform.write_buffer(&device, &queue);
    state.group = Some(device.create_bind_group(
        "beach_wetness",
        &cache.get_bind_group_layout(&pipeline.layout),
        &BindGroupEntries::sequential((
            state.uniform.binding().unwrap(),
            &bed.texture_view,
            &baseline.texture_view,
            wetness.buffer.as_entire_binding(),
        )),
    ));
    state.dispatch = UVec2::new(
        frame.params.grid.x.div_ceil(8),
        frame.params.grid.y.div_ceil(8),
    );
}

fn run_wetness(
    mut ctx: RenderContext,
    pipeline: Res<WetnessPipeline>,
    cache: Res<PipelineCache>,
    state: Res<WetnessGroup>,
) {
    let (Some(group), Some(compute)) =
        (&state.group, cache.get_compute_pipeline(pipeline.pipeline))
    else {
        return;
    };
    let mut pass = ctx
        .command_encoder()
        .begin_compute_pass(&ComputePassDescriptor {
            label: Some("beach_wetness"),
            timestamp_writes: None,
        });
    pass.set_pipeline(compute);
    pass.set_bind_group(0, group, &[]);
    pass.dispatch_workgroups(state.dispatch.x, state.dispatch.y, 1);
}
