use std::num::NonZero;

use bevy_asset::{Assets, Handle, RenderAssetUsages};
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
    pub horizontal_io: BindGroupLayout,
    pub vertical_io: BindGroupLayout,
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
                ),
            ),
        );

        // Horizontal I/O bind group layout (group 1)
        let horizontal_io = render_device.create_bind_group_layout(
            "fft_horizontal_io_bind_group_layout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (
                    texture_2d(TextureSampleType::Float { filterable: false }),
                    texture_2d(TextureSampleType::Float { filterable: false }),
                    texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::WriteOnly),
                    texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::WriteOnly),
                ),
            ),
        );

        // Vertical I/O bind group layout (group 1)
        let vertical_io = render_device.create_bind_group_layout(
            "fft_vertical_io_bind_group_layout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (
                    texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::ReadWrite),
                    texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::ReadWrite),
                    texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::WriteOnly),
                    texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::WriteOnly),
                ),
            ),
        );

        Self {
            common,
            horizontal_io,
            vertical_io,
        }
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
            layout: vec![layouts.common.clone(), layouts.horizontal_io.clone()],
            push_constant_ranges: vec![push_constant_range.clone()],
            shader: shaders::FFT,
            shader_defs: horizontal_shader_defs.clone(),
            entry_point: "fft".into(),
            zero_initialize_workgroup_memory: false,
        });

        let fft_vertical = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("fft_vertical_pipeline".into()),
            layout: vec![layouts.common.clone(), layouts.vertical_io.clone()],
            push_constant_ranges: vec![push_constant_range.clone()],
            shader: shaders::FFT,
            shader_defs: vertical_shader_defs.clone(),
            entry_point: "fft".into(),
            zero_initialize_workgroup_memory: false,
        });

        let ifft_horizontal = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("ifft_horizontal_pipeline".into()),
            layout: vec![layouts.common.clone(), layouts.horizontal_io.clone()],
            push_constant_ranges: vec![push_constant_range.clone()],
            shader: shaders::IFFT,
            shader_defs: horizontal_shader_defs,
            entry_point: "ifft".into(),
            zero_initialize_workgroup_memory: false,
        });

        let ifft_vertical = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("ifft_vertical_pipeline".into()),
            layout: vec![layouts.common.clone(), layouts.vertical_io.clone()],
            push_constant_ranges: vec![push_constant_range],
            shader: shaders::IFFT,
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
    // Input textures (for first pass)
    pub re_in: Handle<Image>,
    pub im_in: Handle<Image>,

    // Ping-pong buffers (both read/write)
    pub buffer_a_re: Handle<Image>,
    pub buffer_a_im: Handle<Image>,
    pub buffer_b_re: Handle<Image>,
    pub buffer_b_im: Handle<Image>,

    // Final output handles (for display)
    pub re_out: Handle<Image>,
    pub im_out: Handle<Image>,
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
            | TextureUsages::RENDER_ATTACHMENT;

        let re_in = images.add(image.clone());
        let im_in = images.add(image.clone());
        let re_out = images.add(image.clone());
        let im_out = images.add(image.clone());
        let buffer_a_re = images.add(image.clone());
        let buffer_a_im = images.add(image.clone());
        let buffer_b_re = images.add(image.clone());
        let buffer_b_im = images.add(image.clone());

        info!("Prepared FFT textures");

        commands.entity(entity).insert(FftTextures {
            re_in,
            im_in,
            re_out,
            im_out,
            buffer_a_re,
            buffer_a_im,
            buffer_b_re,
            buffer_b_im,
        });
    }
}

#[derive(Component)]
pub struct FftBindGroups {
    pub common: BindGroup,
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
    query: Query<
        (Entity, &FftTextures, &FftSourceImage),
        (With<FftSettings>, Without<FftBindGroups>),
    >,
) {
    let Some(settings_binding) = fft_uniforms.binding() else {
        info!("FftSettings uniform buffer missing, skipping bind group creation");
        return;
    };

    let Some(roots_binding) = fft_roots_buffer.buffer.binding() else {
        info!("FftRootsBuffer buffer missing, skipping bind group creation");
        return;
    };

    for (entity, textures, img) in &query {
        let Some(source_image) = gpu_images.get(&img.0) else {
            continue;
        };
        let Some(re_in) = gpu_images.get(&textures.re_in) else {
            continue;
        };
        let Some(im_in) = gpu_images.get(&textures.im_in) else {
            continue;
        };
        let Some(re_out) = gpu_images.get(&textures.re_out) else {
            continue;
        };
        let Some(im_out) = gpu_images.get(&textures.im_out) else {
            continue;
        };
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

        // Common bind group (group 0)
        let common = render_device.create_bind_group(
            "fft_common_bind_group",
            &layouts.common,
            &BindGroupEntries::with_indices((
                (0, settings_binding.clone()),
                (1, roots_binding.clone()),
                (2, &buffer_a_re.texture_view),
                (3, &buffer_a_im.texture_view),
                (4, &buffer_b_re.texture_view),
                (5, &buffer_b_im.texture_view),
            )),
        );

        // Horizontal I/O bind group (group 1)
        let horizontal_io = render_device.create_bind_group(
            "fft_horizontal_io_bind_group",
            &layouts.horizontal_io,
            &BindGroupEntries::with_indices((
                (0, &source_image.texture_view),
                (1, &im_in.texture_view),
                (2, &re_out.texture_view), // dummy output, not actually used
                (3, &im_out.texture_view),
            )),
        );

        // Vertical I/O bind group (group 1)
        let vertical_io = render_device.create_bind_group(
            "fft_vertical_io_bind_group",
            &layouts.vertical_io,
            &BindGroupEntries::with_indices((
                // use the ping-pong buffers as input to the vertical pass
                (0, &buffer_a_re.texture_view),
                (1, &buffer_a_im.texture_view),
                (2, &re_out.texture_view),
                (3, &im_out.texture_view),
            )),
        );

        commands.entity(entity).insert(FftBindGroups {
            common,
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
