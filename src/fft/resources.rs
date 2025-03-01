use std::num::NonZero;

use bevy_asset::{AssetServer, Assets, Handle, RenderAssetUsages};
use bevy_ecs::{
    component::Component,
    entity::Entity,
    query::{With, Without},
    resource::Resource,
    system::{Commands, Query, Res, ResMut},
    world::{FromWorld, World},
};
use bevy_image::Image;
use bevy_log::info;
use bevy_render::{
    extract_component::{ComponentUniforms, ExtractComponent},
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

use super::{shaders, FftRoots, FftSettings, FftSource, FftSourceImage};

#[derive(Resource)]
pub(crate) struct FftBindGroupLayouts {
    pub common: BindGroupLayout,
}

impl FromWorld for FftBindGroupLayouts {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();

        // Common bind group layout (group 0)
        let common = render_device.create_bind_group_layout(
            "fft_common_bind_group_layout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (
                    uniform_buffer::<FftSettings>(false),
                    storage_buffer_sized(
                        false,
                        Some(NonZero::<u64>::new(std::mem::size_of::<FftRoots>() as u64).unwrap()),
                    ),
                    texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::ReadWrite),
                    texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::ReadWrite),
                    texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::ReadWrite),
                    texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::ReadWrite),
                    texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::ReadWrite),
                    texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::ReadWrite),
                    texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::ReadWrite),
                    texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::ReadWrite),
                ),
            ),
        );

        Self { common }
    }
}

#[derive(Resource)]
pub(crate) struct FftPipelines {
    pub fft_horizontal: CachedComputePipelineId,
    pub fft_vertical: CachedComputePipelineId,
    pub ifft_horizontal: CachedComputePipelineId,
    pub ifft_vertical: CachedComputePipelineId,
}

impl FromWorld for FftPipelines {
    fn from_world(world: &mut World) -> Self {
        let pipeline_cache = world.resource::<PipelineCache>();
        let layouts = world.resource::<FftBindGroupLayouts>();
        let asset_server = world.resource::<AssetServer>();
        let fft = asset_server.load("fft.wgsl");
        let ifft = asset_server.load("ifft.wgsl");

        // shader definitions
        let base_shader_defs = vec![ShaderDefVal::UInt("CHANNELS".into(), 4)];
        let mut horizontal_shader_defs = base_shader_defs.clone();
        horizontal_shader_defs.push(ShaderDefVal::Bool("HORIZONTAL".into(), true));
        let mut vertical_shader_defs = base_shader_defs.clone();
        vertical_shader_defs.push(ShaderDefVal::Bool("VERTICAL".into(), true));

        let push_constant_range = PushConstantRange {
            stages: ShaderStages::COMPUTE,
            range: 0..4,
        };

        let fft_horizontal = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("fft_horizontal_pipeline".into()),
            layout: vec![layouts.common.clone()],
            push_constant_ranges: vec![push_constant_range.clone()],
            shader: fft.clone(),
            shader_defs: horizontal_shader_defs.clone(),
            entry_point: "fft".into(),
            zero_initialize_workgroup_memory: false,
        });

        let fft_vertical = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("fft_vertical_pipeline".into()),
            layout: vec![layouts.common.clone()],
            push_constant_ranges: vec![push_constant_range.clone()],
            shader: fft.clone(),
            shader_defs: vertical_shader_defs.clone(),
            entry_point: "fft".into(),
            zero_initialize_workgroup_memory: false,
        });

        let ifft_horizontal = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("ifft_horizontal_pipeline".into()),
            layout: vec![layouts.common.clone()],
            push_constant_ranges: vec![push_constant_range.clone()],
            shader: ifft.clone(),
            shader_defs: horizontal_shader_defs,
            entry_point: "ifft".into(),
            zero_initialize_workgroup_memory: false,
        });

        let ifft_vertical = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("ifft_vertical_pipeline".into()),
            layout: vec![layouts.common.clone()],
            push_constant_ranges: vec![push_constant_range],
            shader: ifft.clone(),
            shader_defs: vertical_shader_defs,
            entry_point: "ifft".into(),
            zero_initialize_workgroup_memory: false,
        });

        Self {
            fft_horizontal,
            fft_vertical,
            ifft_horizontal,
            ifft_vertical,
        }
    }
}

#[derive(Component, ExtractComponent, Clone)]
pub struct FftTextures {
    // Ping pong buffers (read-write)
    pub buffer_a_re: Handle<Image>,
    pub buffer_a_im: Handle<Image>,
    pub buffer_b_re: Handle<Image>,
    pub buffer_b_im: Handle<Image>,
    pub buffer_c_re: Handle<Image>,
    pub buffer_c_im: Handle<Image>,
    pub buffer_d_re: Handle<Image>,
    pub buffer_d_im: Handle<Image>,
}

pub fn prepare_fft_textures(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    query: Query<(Entity, &FftSource), Without<FftTextures>>,
) {
    for (entity, source) in &query {
        let mut image = Image::new_fill(
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

        image.texture_descriptor.usage = TextureUsages::STORAGE_BINDING
            | TextureUsages::TEXTURE_BINDING
            | TextureUsages::COPY_DST
            | TextureUsages::COPY_SRC
            | TextureUsages::RENDER_ATTACHMENT;

        let buffer_a_re = images.add(image.clone());
        let buffer_a_im = images.add(image.clone());
        let buffer_b_re = images.add(image.clone());
        let buffer_b_im = images.add(image.clone());
        let buffer_c_re = images.add(image.clone());
        let buffer_c_im = images.add(image.clone());
        let buffer_d_re = images.add(image.clone());
        let buffer_d_im = images.add(image.clone());

        commands.entity(entity).insert(FftTextures {
            buffer_a_re,
            buffer_a_im,
            buffer_b_re,
            buffer_b_im,
            buffer_c_re,
            buffer_c_im,
            buffer_d_re,
            buffer_d_im,
        });
    }
}

pub fn copy_source_to_input(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    gpu_images: Res<RenderAssets<GpuImage>>,
    query: Query<(&FftTextures, &FftSourceImage)>,
) {
    for (textures, source_image) in &query {
        if let (Some(src_re), Some(src_im), Some(buf_c_re), Some(buf_c_im)) = (
            gpu_images.get(&source_image.image),
            gpu_images.get(&source_image.image_im),
            gpu_images.get(&textures.buffer_c_re),
            gpu_images.get(&textures.buffer_c_im),
        ) {
            let mut encoder = render_device.create_command_encoder(&CommandEncoderDescriptor {
                label: Some("copy_source_to_input"),
            });

            encoder.copy_texture_to_texture(
                src_re.texture.as_image_copy(),
                buf_c_re.texture.as_image_copy(),
                Extent3d {
                    width: src_re.texture.size().width,
                    height: src_re.texture.size().height,
                    depth_or_array_layers: 1,
                },
            );

            encoder.copy_texture_to_texture(
                src_im.texture.as_image_copy(),
                buf_c_im.texture.as_image_copy(),
                Extent3d {
                    width: src_im.texture.size().width,
                    height: src_im.texture.size().height,
                    depth_or_array_layers: 1,
                },
            );

            queue.submit([encoder.finish()]);
        }
    }
}

#[derive(Component)]
pub struct FftBindGroups {
    pub horizontal_io: BindGroup,
    pub vertical_io: BindGroup,
}

pub(crate) fn prepare_fft_bind_groups(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    layouts: Res<FftBindGroupLayouts>,
    fft_uniforms: Res<ComponentUniforms<FftSettings>>,
    fft_roots_buffer: Res<FftRootsBuffer>,
    gpu_images: Res<RenderAssets<GpuImage>>,
    query: Query<(Entity, &FftTextures, &FftSettings), Without<FftBindGroups>>,
) {
    let Some(settings_binding) = fft_uniforms.binding() else {
        info!("FftSettings uniform buffer missing, skipping bind group creation");
        return;
    };

    let Some(roots_binding) = fft_roots_buffer.buffer.binding() else {
        info!("FftRootsBuffer buffer missing, skipping bind group creation");
        return;
    };

    for (entity, textures, settings) in &query {
        let Some(buffer_a_re) = gpu_images.get(&textures.buffer_a_re) else {
            continue;
        };
        let Some(buffer_a_im) = gpu_images.get(&textures.buffer_a_im) else {
            continue;
        };
        let Some(buffer_b_re) = gpu_images.get(&textures.buffer_b_re) else {
            continue;
        };
        let Some(buffer_b_im) = gpu_images.get(&textures.buffer_b_im) else {
            continue;
        };
        let Some(buffer_c_re) = gpu_images.get(&textures.buffer_c_re) else {
            continue;
        };
        let Some(buffer_c_im) = gpu_images.get(&textures.buffer_c_im) else {
            continue;
        };
        let Some(buffer_d_re) = gpu_images.get(&textures.buffer_d_re) else {
            continue;
        };
        let Some(buffer_d_im) = gpu_images.get(&textures.buffer_d_im) else {
            continue;
        };

        // Horizontal I/O bind group (group 1)
        let horizontal_io = render_device.create_bind_group(
            "fft_horizontal_io_bind_group",
            &layouts.common,
            &BindGroupEntries::sequential((
                settings_binding.clone(),
                roots_binding.clone(),
                &buffer_a_re.texture_view,
                &buffer_a_im.texture_view,
                &buffer_b_re.texture_view,
                &buffer_b_im.texture_view,
                &buffer_c_re.texture_view,
                &buffer_c_im.texture_view,
                &buffer_d_re.texture_view,
                &buffer_d_im.texture_view,
            )),
        );

        // Vertical I/O bind group (group 1)
        let vertical_io = render_device.create_bind_group(
            "fft_vertical_io_bind_group",
            &layouts.common,
            &BindGroupEntries::sequential((
                settings_binding.clone(),
                roots_binding.clone(),
                &buffer_a_re.texture_view,
                &buffer_a_im.texture_view,
                &buffer_b_re.texture_view,
                &buffer_b_im.texture_view,
                &buffer_c_re.texture_view,
                &buffer_c_im.texture_view,
                &buffer_d_re.texture_view,
                &buffer_d_im.texture_view,
            )),
        );

        commands.entity(entity).insert(FftBindGroups {
            horizontal_io,
            vertical_io,
        });
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

    once!(info!("Prepared FFT roots buffer"));

    fft_roots_buffer.buffer.set(*roots);
    fft_roots_buffer.buffer.write_buffer(&device, &queue);
}
