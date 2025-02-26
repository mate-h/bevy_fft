use std::num::NonZero;

use bevy::log;
use bevy_asset::{Assets, RenderAssetUsages};
use bevy_ecs::{
    component::Component,
    entity::Entity,
    query::{With, Without},
    resource::Resource,
    system::{Commands, Query, Res, ResMut},
    world::{FromWorld, World},
};
use bevy_image::Image;
use bevy_render::{
    extract_component::ComponentUniforms,
    render_asset::RenderAssets,
    render_resource::{
        binding_types::{storage_buffer_sized, texture_2d, texture_storage_2d, uniform_buffer},
        *,
    },
    renderer::{RenderDevice, RenderQueue},
    texture::GpuImage,
};
use bevy_utils::once;

use crate::complex::c32;

use super::{shaders, FftRoots, FftSettings, FftSource, FftSourceImage, FftTextures};

#[derive(Resource)]
pub(crate) struct FftBindGroupLayouts {
    pub compute: BindGroupLayout,
}

impl FromWorld for FftBindGroupLayouts {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();

        let compute = render_device.create_bind_group_layout(
            "fft_compute_bind_group_layout",
            &BindGroupLayoutEntries::with_indices(
                ShaderStages::COMPUTE,
                (
                    (0, uniform_buffer::<FftSettings>(false)),
                    (
                        1,
                        storage_buffer_sized(
                            false,
                            Some(
                                NonZero::<u64>::new(std::mem::size_of::<FftRoots>() as u64)
                                    .unwrap(),
                            ),
                        ),
                    ),
                    (
                        2,
                        texture_2d(TextureSampleType::Float { filterable: false }),
                    ),
                    (
                        3,
                        texture_storage_2d(
                            TextureFormat::Rgba32Uint,
                            StorageTextureAccess::ReadWrite,
                        ),
                    ),
                    (
                        4,
                        texture_storage_2d(
                            TextureFormat::Rgba32Float,
                            StorageTextureAccess::ReadWrite,
                        ),
                    ),
                    (
                        5,
                        texture_storage_2d(
                            TextureFormat::Rgba32Float,
                            StorageTextureAccess::ReadWrite,
                        ),
                    ),
                ),
            ),
        );

        Self { compute }
    }
}

#[derive(Resource)]
pub(crate) struct FftPipelines {
    pub fft: CachedComputePipelineId,
    // pub ifft: CachedComputePipelineId,
}

impl FromWorld for FftPipelines {
    fn from_world(world: &mut World) -> Self {
        let pipeline_cache = world.resource::<PipelineCache>();
        let layouts = world.resource::<FftBindGroupLayouts>();
        let shader_defs = vec![
            // number of channels in the input and output textures
            ShaderDefVal::UInt("CHANNELS".into(), 4),
        ];
        let fft = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("fft_pipeline".into()),
            layout: vec![layouts.compute.clone()],
            push_constant_ranges: vec![],
            shader: shaders::FFT,
            shader_defs,
            entry_point: "fft".into(),
            zero_initialize_workgroup_memory: false,
        });

        Self { fft }
    }
}

pub(crate) fn prepare_fft_textures(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    query: Query<(Entity, &FftSource), Without<FftTextures>>,
) {
    for (entity, source) in &query {
        // if the entity has a FftTextures component, skip

        // Create a new image that can be used as a render target
        let mut image = Image::new_fill(
            Extent3d {
                width: source.size.x,
                height: source.size.y,
                depth_or_array_layers: 1,
            },
            TextureDimension::D2,
            &[0; 16],
            TextureFormat::Rgba32Uint,
            RenderAssetUsages::default(),
        );

        // Set the texture usage flags needed for FFT computation and display
        image.texture_descriptor.usage = TextureUsages::STORAGE_BINDING
            | TextureUsages::TEXTURE_BINDING
            | TextureUsages::COPY_DST
            | TextureUsages::RENDER_ATTACHMENT;

        // Add the image to the asset system and get its handle
        let image_handle = images.add(image);

        let mut re_image = Image::new_fill(
            Extent3d {
                width: source.size.x,
                height: source.size.y,
                depth_or_array_layers: 1,
            },
            TextureDimension::D2,
            &[0; 16],
            TextureFormat::Rgba32Float,
            RenderAssetUsages::default(),
        );

        re_image.texture_descriptor.usage = TextureUsages::STORAGE_BINDING
            | TextureUsages::TEXTURE_BINDING
            | TextureUsages::COPY_DST
            | TextureUsages::RENDER_ATTACHMENT;

        let im_image = re_image.clone();

        let re_image_handle = images.add(re_image);
        let im_image_handle = images.add(im_image);

        log::info!("Prepared FFT textures");

        commands.entity(entity).insert(FftTextures {
            output: image_handle,
            re: re_image_handle,
            im: im_image_handle,
        });
    }
}

#[derive(Component)]
pub struct FftBindGroups {
    pub compute: BindGroup,
}

pub(crate) fn prepare_fft_bind_groups(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    layouts: Res<FftBindGroupLayouts>,
    fft_uniforms: Res<ComponentUniforms<FftSettings>>,
    fft_roots_buffer: Res<FftRootsBuffer>,
    gpu_images: Res<RenderAssets<GpuImage>>,
    query: Query<
        (Entity, &FftTextures, &FftSourceImage),
        (With<FftSettings>, Without<FftBindGroups>),
    >,
) {
    let settings_binding = fft_uniforms
        .binding()
        .expect("Failed to prepare FFT bind groups. FftSettings uniform buffer missing");
    let roots_binding = fft_roots_buffer
        .buffer
        .binding()
        .expect("Failed to prepare FFT bind groups. FftRootsBuffer buffer missing");

    for (entity, textures, img) in &query {
        let source_image = gpu_images
            .get(&img.0)
            .expect("Failed to prepare FFT bind groups. Source image not found");

        let output_image = gpu_images
            .get(&textures.output)
            .expect("Failed to prepare FFT bind groups. Output image not found");

        let re_image = gpu_images
            .get(&textures.re)
            .expect("Failed to prepare FFT bind groups. Re image not found");

        let im_image = gpu_images
            .get(&textures.im)
            .expect("Failed to prepare FFT bind groups. Im image not found");

        let compute = render_device.create_bind_group(
            "fft_compute_bind_group",
            &layouts.compute,
            &BindGroupEntries::with_indices((
                (0, settings_binding.clone()),
                (1, roots_binding.clone()),
                (2, &source_image.texture_view),
                (3, &output_image.texture_view),
                (4, &re_image.texture_view),
                (5, &im_image.texture_view),
            )),
        );

        once!(log::info!("Prepared FFT bind groups"));

        commands.entity(entity).insert(FftBindGroups { compute });
    }
}

#[derive(Resource)]
pub struct FftRootsBuffer {
    pub buffer: StorageBuffer<FftRoots>,
}

impl FromWorld for FftRootsBuffer {
    fn from_world(world: &mut World) -> Self {
        let data = world
            .query_filtered::<&FftRoots, With<FftSettings>>()
            .iter(world)
            .next()
            .map_or_else(
                || FftRoots {
                    roots: [c32::new(0.0, 0.0); 8192],
                },
                |roots| *roots,
            );

        Self {
            buffer: StorageBuffer::from(data),
        }
    }
}

pub(crate) fn prepare_fft_roots_buffer(
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    query: Query<(&FftRoots, &FftSettings), With<FftSettings>>,
    mut fft_roots_buffer: ResMut<FftRootsBuffer>,
) {
    let Ok((roots, _)) = query.get_single() else {
        return;
    };

    once!(log::info!("Prepared FFT roots buffer"));

    fft_roots_buffer.buffer.set(*roots);
    fft_roots_buffer.buffer.write_buffer(&device, &queue);
}
