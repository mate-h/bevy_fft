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
    render_resource::{
        binding_types::{texture_storage_2d, uniform_buffer},
        *,
    },
    renderer::RenderDevice,
    texture::TextureCache,
};

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
                    (0, uniform_buffer::<FftSettings>(true)),
                    (1, uniform_buffer::<FftRoots>(true)),
                    (
                        2,
                        texture_storage_2d(
                            TextureFormat::R32Float,
                            StorageTextureAccess::ReadWrite,
                        ),
                    ),
                    (
                        3,
                        texture_storage_2d(
                            TextureFormat::R32Float,
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

        let shader_defs = vec![ShaderDefVal::Int("CHANNELS".into(), 4)];

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
    views: Query<(Entity, &FftSettings)>,
) {
    log::info!("Preparing FFT textures");
    for (entity, settings) in &views {
        log::info!("Preparing FFT textures for {:?}", settings.size);
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
                format: TextureFormat::R32Float,
                usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
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
                format: TextureFormat::R32Float,
                usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
        );

        let temp = texture_cache.get(
            &render_device,
            TextureDescriptor {
                label: Some("fft_temp"),
                size: Extent3d {
                    width: settings.size.x,
                    height: settings.size.y,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::R32Float,
                usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
        );

        commands.entity(entity).insert(FftTextures {
            input,
            output,
            temp,
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
    views: Query<(Entity, &FftTextures), With<FftSettings>>,
) {
    for (entity, textures) in &views {
        let compute = render_device.create_bind_group(
            "fft_compute_bind_group",
            &layouts.compute,
            &BindGroupEntries::with_indices((
                (0, &textures.input.default_view),
                (1, &textures.output.default_view),
                (2, &textures.temp.default_view),
            )),
        );

        commands.entity(entity).insert(FftBindGroups { compute });
    }
}
