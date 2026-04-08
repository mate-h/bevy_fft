use std::num::NonZero;

use bevy::{
    asset::{AssetServer, Assets, Handle, RenderAssetUsages},
    image::Image,
    log::{info, warn},
    prelude::*,
    render::{
        extract_component::{ComponentUniforms, ExtractComponent},
        globals::{GlobalsBuffer, GlobalsUniform},
        render_asset::RenderAssets,
        render_resource::{binding_types::*, *},
        renderer::{RenderDevice, RenderQueue},
        texture::GpuImage,
    },
    shader::ShaderDefVal,
    utils::once,
};

use super::{FftInputDomain, FftRoots, FftSettings, FftSource};
use crate::{complex::c32, fft::FftInputTexture};

#[derive(Resource)]
pub struct FftBindGroupLayouts {
    pub common: BindGroupLayoutDescriptor,
    pub resolve_outputs: BindGroupLayoutDescriptor,
}

impl FromWorld for FftBindGroupLayouts {
    fn from_world(_world: &mut World) -> Self {
        let entries = BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                uniform_buffer::<GlobalsUniform>(false),
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
        );

        let resolve_entries = BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                uniform_buffer::<FftSettings>(false),
                texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::ReadOnly),
                texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::ReadOnly),
                texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::ReadOnly),
                texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::WriteOnly),
                texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::WriteOnly),
            ),
        );

        Self {
            common: BindGroupLayoutDescriptor::new("fft_common_bind_group_layout", &entries),
            resolve_outputs: BindGroupLayoutDescriptor::new(
                "fft_resolve_outputs_bind_group_layout",
                &resolve_entries,
            ),
        }
    }
}

#[derive(Resource)]
pub(crate) struct FftPipelines {
    pub forward_br_horizontal: CachedComputePipelineId,
    pub forward_br_vertical: CachedComputePipelineId,
    pub radix2_dit: CachedComputePipelineId,
    pub fft_copy: CachedComputePipelineId,
    pub inverse_br_horizontal: CachedComputePipelineId,
    pub inverse_br_vertical: CachedComputePipelineId,
    pub resolve_spectrum: CachedComputePipelineId,
    pub resolve_spatial: CachedComputePipelineId,
}

impl FromWorld for FftPipelines {
    fn from_world(world: &mut World) -> Self {
        let pipeline_cache = world.resource::<PipelineCache>();
        let layouts = world.resource::<FftBindGroupLayouts>();
        let asset_server = world.resource::<AssetServer>();
        let fft = asset_server.load("fft.wgsl");
        let ifft = asset_server.load("ifft.wgsl");
        let resolve_shader = super::shaders::RESOLVE_OUTPUTS.clone();

        let base_shader_defs = vec![ShaderDefVal::UInt("CHANNELS".into(), 4)];

        let push_constant_range_20 = PushConstantRange {
            stages: ShaderStages::COMPUTE,
            range: 0..20,
        };

        let forward_br_horizontal = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("fft_forward_br_horizontal".into()),
            layout: vec![layouts.common.clone()],
            push_constant_ranges: vec![],
            shader: fft.clone(),
            shader_defs: base_shader_defs.clone(),
            entry_point: Some("fft_forward_br_horizontal".into()),
            zero_initialize_workgroup_memory: false,
        });

        let forward_br_vertical = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("fft_forward_br_vertical".into()),
            layout: vec![layouts.common.clone()],
            push_constant_ranges: vec![],
            shader: fft.clone(),
            shader_defs: base_shader_defs.clone(),
            entry_point: Some("fft_forward_br_vertical".into()),
            zero_initialize_workgroup_memory: false,
        });

        let radix2_dit = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("fft_radix2_dit".into()),
            layout: vec![layouts.common.clone()],
            push_constant_ranges: vec![push_constant_range_20.clone()],
            shader: fft.clone(),
            shader_defs: base_shader_defs.clone(),
            entry_point: Some("fft_radix2_dit".into()),
            zero_initialize_workgroup_memory: false,
        });

        let fft_copy = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("fft_copy_buffer".into()),
            layout: vec![layouts.common.clone()],
            push_constant_ranges: vec![push_constant_range_20],
            shader: fft.clone(),
            shader_defs: base_shader_defs.clone(),
            entry_point: Some("fft_copy_buffer".into()),
            zero_initialize_workgroup_memory: false,
        });

        let inverse_br_horizontal = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("ifft_br_horizontal".into()),
            layout: vec![layouts.common.clone()],
            push_constant_ranges: vec![],
            shader: ifft.clone(),
            shader_defs: base_shader_defs.clone(),
            entry_point: Some("ifft_br_horizontal".into()),
            zero_initialize_workgroup_memory: false,
        });

        let inverse_br_vertical = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("ifft_br_vertical".into()),
            layout: vec![layouts.common.clone()],
            push_constant_ranges: vec![],
            shader: ifft.clone(),
            shader_defs: base_shader_defs,
            entry_point: Some("ifft_br_vertical".into()),
            zero_initialize_workgroup_memory: false,
        });

        let resolve_spectrum = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("fft_resolve_spectrum_pipeline".into()),
            layout: vec![layouts.resolve_outputs.clone()],
            push_constant_ranges: vec![],
            shader: resolve_shader.clone(),
            shader_defs: vec![],
            entry_point: Some("resolve_power_spectrum_from_c".into()),
            zero_initialize_workgroup_memory: false,
        });

        let resolve_spatial = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("fft_resolve_spatial_pipeline".into()),
            layout: vec![layouts.resolve_outputs.clone()],
            push_constant_ranges: vec![],
            shader: resolve_shader,
            shader_defs: vec![],
            entry_point: Some("resolve_spatial_from_b".into()),
            zero_initialize_workgroup_memory: false,
        });

        Self {
            forward_br_horizontal,
            forward_br_vertical,
            radix2_dit,
            fft_copy,
            inverse_br_horizontal,
            inverse_br_vertical,
            resolve_spectrum,
            resolve_spatial,
        }
    }
}

/// All GPU textures that back a single [`FftSource`].
///
/// Resolve passes fill [`Self::spatial_output`] (after inverse FFT from **B**) and
/// [`Self::power_spectrum`] (after the spectrum stage from **C**). The
/// `buffer_*` handles are ping-pong storage used inside the FFT graph and are mainly interesting
/// when you author custom compute that plugs into those bindings.
#[derive(Component, ExtractComponent, Clone)]
pub struct FftTextures {
    pub buffer_a_re: Handle<Image>,
    pub buffer_a_im: Handle<Image>,
    pub buffer_b_re: Handle<Image>,
    pub buffer_b_im: Handle<Image>,
    pub buffer_c_re: Handle<Image>,
    pub buffer_c_im: Handle<Image>,
    pub buffer_d_re: Handle<Image>,
    pub buffer_d_im: Handle<Image>,
    /// Spatial result after the inverse FFT. Typical bind target for on-screen display.
    pub spatial_output: Handle<Image>,
    /// Log-magnitude spectrum with the DC term moved to the middle for easier viewing.
    pub power_spectrum: Handle<Image>,
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

        let spatial_output = images.add(image.clone());
        let power_spectrum = images.add(image.clone());

        commands.entity(entity).insert(FftTextures {
            buffer_a_re,
            buffer_a_im,
            buffer_b_re,
            buffer_b_im,
            buffer_c_re,
            buffer_c_im,
            buffer_d_re,
            buffer_d_im,
            spatial_output,
            power_spectrum,
        });
    }
}

#[derive(Component)]
pub struct FftBindGroups {
    pub common: BindGroup,
}

#[derive(Component)]
pub(crate) struct FftResolveBindGroups {
    pub group: BindGroup,
}

#[allow(clippy::too_many_arguments)]
/// Rebuilds the main FFT bind groups once per frame.
///
/// CPU uploads can rebuild [`GpuImage`] data for buffer **A**, which would leave stale views if we
/// cached bind groups forever. Refreshing every frame keeps the bindings aligned with whatever the
/// asset system just produced.
pub fn prepare_fft_bind_groups(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    pipeline_cache: Res<PipelineCache>,
    layouts: Res<FftBindGroupLayouts>,
    fft_uniforms: Res<ComponentUniforms<FftSettings>>,
    fft_roots_buffer: Res<FftRootsBuffer>,
    globals_buffer: Res<GlobalsBuffer>,
    gpu_images: Res<RenderAssets<GpuImage>>,
    query: Query<(Entity, &FftTextures, &FftSettings)>,
) {
    let Some(settings_binding) = fft_uniforms.binding() else {
        info!("FftSettings uniform buffer missing, skipping bind group creation");
        return;
    };

    let Some(roots_binding) = fft_roots_buffer.buffer.binding() else {
        info!("FftRootsBuffer buffer missing, skipping bind group creation");
        return;
    };

    let Some(globals_binding) = globals_buffer.buffer.binding() else {
        info!("GlobalsBuffer buffer missing, skipping bind group creation");
        return;
    };

    for (entity, textures, _settings) in &query {
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

        let common_layout = pipeline_cache.get_bind_group_layout(&layouts.common);
        let common = render_device.create_bind_group(
            "fft_bind_group",
            &common_layout,
            &BindGroupEntries::sequential((
                globals_binding.clone(),
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

        commands.entity(entity).insert(FftBindGroups { common });
    }
}

type PrepareFftResolveBindGroupsQuery<'w, 's> =
    Query<'w, 's, (Entity, &'static FftTextures), With<FftBindGroups>>;

#[allow(clippy::too_many_arguments)]
/// Rebuilds resolve bind groups every frame so `texture_view`s stay aligned with [`prepare_fft_bind_groups`].
pub(crate) fn prepare_fft_resolve_bind_groups(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    pipeline_cache: Res<PipelineCache>,
    layouts: Res<FftBindGroupLayouts>,
    fft_uniforms: Res<ComponentUniforms<FftSettings>>,
    gpu_images: Res<RenderAssets<GpuImage>>,
    query: PrepareFftResolveBindGroupsQuery,
) {
    let Some(settings_binding) = fft_uniforms.binding() else {
        return;
    };

    for (entity, textures) in &query {
        let Some(c_re) = gpu_images.get(&textures.buffer_c_re) else {
            continue;
        };
        let Some(c_im) = gpu_images.get(&textures.buffer_c_im) else {
            continue;
        };
        let Some(b_re) = gpu_images.get(&textures.buffer_b_re) else {
            continue;
        };
        let Some(power) = gpu_images.get(&textures.power_spectrum) else {
            continue;
        };
        let Some(spatial) = gpu_images.get(&textures.spatial_output) else {
            continue;
        };

        let layout = pipeline_cache.get_bind_group_layout(&layouts.resolve_outputs);
        let group = render_device.create_bind_group(
            "fft_resolve_outputs_bind_group",
            &layout,
            &BindGroupEntries::sequential((
                settings_binding.clone(),
                &c_re.texture_view,
                &c_im.texture_view,
                &b_re.texture_view,
                &power.texture_view,
                &spatial.texture_view,
            )),
        );

        commands
            .entity(entity)
            .insert(FftResolveBindGroups { group });
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
    let Ok((roots, _)) = query.single() else {
        return;
    };

    once!(info!("Prepared FFT roots buffer"));

    fft_roots_buffer.buffer.set(*roots);
    fft_roots_buffer.buffer.write_buffer(&device, &queue);
}

/// Copies [`FftInputTexture`] data into the FFT working images on the CPU.
///
/// When no imaginary texture is attached the corresponding buffer is cleared to zero.
pub(crate) fn copy_input_textures_to_fft_buffers(
    mut images: ResMut<Assets<Image>>,
    query: Query<(&FftTextures, &FftInputTexture, &FftSource)>,
) {
    for (textures, input, source) in &query {
        let Some(src_re) = images.get(&input.real) else {
            continue;
        };
        let src_re_data = src_re.data.clone();

        let expected_extent = Extent3d {
            width: source.size.x,
            height: source.size.y,
            depth_or_array_layers: 1,
        };
        if src_re.texture_descriptor.size != expected_extent {
            warn!(
                "Input real texture size {:?} does not match FFT size {:?}, skipping copy",
                src_re.texture_descriptor.size, expected_extent
            );
            continue;
        }

        let (dst_re_handle, dst_im_handle) = match source.input_domain {
            FftInputDomain::Spectrum => (&textures.buffer_c_re, &textures.buffer_c_im),
            FftInputDomain::Spatial => (&textures.buffer_a_re, &textures.buffer_a_im),
        };

        if let (Some(dst_re), Some(src_bytes)) =
            (images.get_mut(dst_re_handle), src_re_data.as_ref())
            && let Some(dst_bytes) = dst_re.data.as_mut()
        {
            if dst_bytes.len() == src_bytes.len() {
                dst_bytes.clone_from_slice(src_bytes);
            } else {
                warn!(
                    "Input real texture data length {} does not match destination {}",
                    src_bytes.len(),
                    dst_bytes.len()
                );
                continue;
            }
        }

        if let Some(imag_handle) = &input.imag {
            let src_im_data = images.get(imag_handle).and_then(|img| img.data.clone());
            if let (Some(dst_im), Some(src_bytes)) =
                (images.get_mut(dst_im_handle), src_im_data.as_ref())
                && let Some(dst_bytes) = dst_im.data.as_mut()
            {
                if dst_bytes.len() == src_bytes.len() {
                    dst_bytes.clone_from_slice(src_bytes);
                } else {
                    warn!(
                        "Input imag texture data length {} does not match destination {}",
                        src_bytes.len(),
                        dst_bytes.len()
                    );
                }
            }
        } else if let Some(dst_im) = images.get_mut(dst_im_handle)
            && let Some(dst_bytes) = dst_im.data.as_mut()
        {
            dst_bytes.fill(0);
        }
    }
}
