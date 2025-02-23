use std::num::NonZero;

use bevy::log;
use bevy_ecs::{
    component::Component,
    entity::Entity,
    query::With,
    resource::Resource,
    system::{Commands, Query, Res, ResMut},
    world::{FromWorld, World},
};
use bevy_render::{
    extract_component::ComponentUniforms,
    render_resource::{
        binding_types::{storage_buffer_sized, texture_2d, texture_storage_2d, uniform_buffer},
        *,
    },
    renderer::{RenderDevice, RenderQueue},
    texture::TextureCache,
};

use crate::complex::c32;

use super::{shaders, FftRoots, FftSettings, FftTextures};

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

        // let ifft = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        //     label: Some("ifft_pipeline".into()),
        //     layout: vec![layouts.compute.clone()],
        //     push_constant_ranges: vec![],
        //     shader: shaders::IFFT,
        //     shader_defs: vec![],
        //     entry_point: "ifft".into(),
        //     zero_initialize_workgroup_memory: false,
        // });

        Self { fft }
    }
}

pub(crate) fn prepare_fft_textures(
    mut commands: Commands,
    mut texture_cache: ResMut<TextureCache>,
    render_device: Res<RenderDevice>,
    matches: Query<(Entity, &FftSettings)>,
) {
    for (entity, settings) in &matches {
        let input = texture_cache.get(
            &render_device,
            TextureDescriptor {
                label: Some("fft_input"),
                size: Extent3d {
                    width: settings.size.x,
                    height: settings.size.y,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::Rgba32Float,
                usage: TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
        );

        let output = texture_cache.get(
            &render_device,
            TextureDescriptor {
                label: Some("fft_output"),
                size: Extent3d {
                    width: settings.size.x,
                    height: settings.size.y,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::Rgba32Uint,
                usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
        );

        commands
            .entity(entity)
            .insert(FftTextures { input, output });
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
    matches: Query<(Entity, &FftTextures), With<FftSettings>>,
) {
    let settings_binding = fft_uniforms
        .binding()
        .expect("Failed to prepare FFT bind groups. FftSettings uniform buffer missing");
    let roots_binding = fft_roots_buffer
        .buffer
        .binding()
        .expect("Failed to prepare FFT bind groups. FftRootsBuffer buffer missing");
    for (entity, textures) in &matches {
        let compute = render_device.create_bind_group(
            "fft_compute_bind_group",
            &layouts.compute,
            &BindGroupEntries::with_indices((
                (0, settings_binding.clone()),
                (1, roots_binding.clone()),
                (2, &textures.input.default_view),
                (3, &textures.output.default_view),
            )),
        );

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
    fft_entity: Query<(&FftRoots, &FftSettings), With<FftSettings>>,
    mut fft_roots_buffer: ResMut<FftRootsBuffer>,
) {
    let Ok((roots, _)) = fft_entity.get_single() else {
        return;
    };

    fft_roots_buffer.buffer.set(*roots);
    fft_roots_buffer.buffer.write_buffer(&device, &queue);
}
