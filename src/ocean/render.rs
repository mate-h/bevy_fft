//! GPU init of `H0` and spectrum fill for buffer **C**, spliced into the root FFT graph.

use std::sync::Mutex;

use bevy::{
    asset::{Assets, Handle, RenderAssetUsages},
    ecs::{
        query::QueryState,
        system::lifetimeless::Read,
        world::{FromWorld, World},
    },
    image::Image,
    prelude::*,
    reflect::Reflect,
    render::{
        extract_component::{ComponentUniforms, ExtractComponent},
        render_asset::RenderAssets,
        render_graph::{Node, NodeRunError, RenderGraphContext, RenderLabel},
        render_resource::{
            binding_types::{texture_storage_2d, uniform_buffer},
            *,
        },
        renderer::RenderDevice,
        texture::GpuImage,
    },
    shader::ShaderDefVal,
};

use crate::fft::{
    FftSettings, FftSource,
    resources::{FftBindGroupLayouts, FftBindGroups, FftTextures},
};

/// Snapshot of [`OceanH0Uniform`] from the last `init_h0` dispatch.
///
/// `init_h0` bakes wind, JONSWAP, and amplitude into the `H0` texture. The compute pass runs again
/// when this value differs from the current uniform, including when [`OceanSimSettings::h0_serial`](crate::ocean::OceanSimSettings#structfield.h0_serial) changes.
#[derive(Resource, Default)]
pub struct OceanInitTracker {
    last_init_h0_uniform: Mutex<Option<OceanH0Uniform>>,
}

#[repr(C)]
#[derive(Component, Clone, Copy, Default, Reflect, ShaderType, PartialEq)]
pub struct OceanH0Uniform {
    pub texture_size: u32,
    pub _pad0: u32,
    pub tile_size: f32,
    pub wind_direction: f32,
    pub wind_speed: f32,
    pub peak_enhancement: f32,
    pub directional_spread: f32,
    pub small_wave_cutoff: f32,
    pub gravity: f32,
    pub amplitude_scale: f32,
    pub h0_serial: u32,
    pub _pad1: u32,
    pub _pad2: u32,
    pub _pad3: u32,
}

impl ExtractComponent for OceanH0Uniform {
    type QueryData = Read<Self>;
    type QueryFilter = ();
    type Out = Self;

    fn extract_component(
        item: bevy::ecs::query::QueryItem<'_, '_, Self::QueryData>,
    ) -> Option<Self::Out> {
        Some(*item)
    }
}

#[repr(C)]
#[derive(Component, Clone, Copy, Default, Reflect, ShaderType)]
pub struct OceanDynamicUniform {
    pub texture_size: u32,
    pub _pad0: u32,
    pub tile_size: f32,
    pub elapsed_seconds: f32,
    pub gravity: f32,
    /// Same radians as [`crate::ocean::OceanSimSettings::wind_direction`] and `init_h0` (`atan2(k.y, k.x)` on the XZ plane; texture row is Z).
    pub wind_direction: f32,
    pub _pad2: f32,
    pub _pad3: f32,
}

impl ExtractComponent for OceanDynamicUniform {
    type QueryData = Read<Self>;
    type QueryFilter = ();
    type Out = Self;

    fn extract_component(
        item: bevy::ecs::query::QueryItem<'_, '_, Self::QueryData>,
    ) -> Option<Self::Out> {
        Some(*item)
    }
}

/// GPU `H0` storage texture handle, extracted from the main world via [`ExtractComponent`].
///
/// The render-world ocean bind-group setup pairs this handle with [`FftTextures`] when wiring ocean compute.
#[derive(Component, Clone, Reflect)]
pub struct OceanH0Image {
    pub texture: Handle<Image>,
}

impl ExtractComponent for OceanH0Image {
    type QueryData = Read<Self>;
    type QueryFilter = ();
    type Out = Self;

    fn extract_component(
        item: bevy::ecs::query::QueryItem<'_, '_, Self::QueryData>,
    ) -> Option<Self::Out> {
        Some(item.clone())
    }
}

/// Per-entity bind groups for ocean compute.
///
/// The prepare system overwrites this each frame so bindings track the current [`GpuImage`] views
/// after a resize or other asset refresh (same idea as [`crate::fft::prepare_fft_bind_groups`]).
#[derive(Component)]
pub struct OceanComputeBindGroups {
    pub init: BindGroup,
    pub spectrum_dynamic: BindGroup,
    pub spectrum_h0_read: BindGroup,
}

#[derive(PartialEq, Eq, Debug, Copy, Clone, Hash, RenderLabel)]
pub struct OceanSpectrumLabel;

#[derive(Resource)]
pub struct OceanComputePipelines {
    pub init_layout: BindGroupLayoutDescriptor,
    pub init: CachedComputePipelineId,
    pub spectrum_layout_dynamic: BindGroupLayoutDescriptor,
    pub spectrum_layout_h0: BindGroupLayoutDescriptor,
    pub spectrum: CachedComputePipelineId,
}

impl FromWorld for OceanComputePipelines {
    fn from_world(world: &mut World) -> Self {
        let pipeline_cache = world.resource::<PipelineCache>();
        let asset_server = world.resource::<AssetServer>();

        let init_entries = BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::WriteOnly),
                uniform_buffer::<OceanH0Uniform>(false),
            ),
        );
        let init_layout = BindGroupLayoutDescriptor::new("ocean_init_h0_layout", &init_entries);

        let spectrum_dyn_entries = BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (uniform_buffer::<OceanDynamicUniform>(false),),
        );
        let spectrum_layout_dynamic =
            BindGroupLayoutDescriptor::new("ocean_spectrum_dynamic_layout", &spectrum_dyn_entries);

        let spectrum_h0_entries = BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (texture_storage_2d(
                TextureFormat::Rgba32Float,
                StorageTextureAccess::ReadOnly,
            ),),
        );
        let spectrum_layout_h0 =
            BindGroupLayoutDescriptor::new("ocean_spectrum_h0_read_layout", &spectrum_h0_entries);

        let init_shader = asset_server.load("ocean/init_h0.wgsl");
        let spectrum_shader = asset_server.load("ocean/ocean_spectrum.wgsl");

        let init = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("ocean_init_h0".into()),
            layout: vec![init_layout.clone()],
            push_constant_ranges: vec![],
            shader: init_shader,
            shader_defs: vec![],
            entry_point: Some("init_h0".into()),
            zero_initialize_workgroup_memory: false,
        });

        let fft_layouts = world.resource::<FftBindGroupLayouts>();

        let spectrum = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("ocean_spectrum_to_c".into()),
            layout: vec![
                fft_layouts.common.clone(),
                spectrum_layout_dynamic.clone(),
                spectrum_layout_h0.clone(),
            ],
            push_constant_ranges: vec![],
            shader: spectrum_shader,
            shader_defs: vec![ShaderDefVal::UInt("CHANNELS".into(), 4)],
            entry_point: Some("ocean_fill_spectrum_c".into()),
            zero_initialize_workgroup_memory: false,
        });

        Self {
            init_layout,
            init,
            spectrum_layout_dynamic,
            spectrum_layout_h0,
            spectrum,
        }
    }
}

pub(super) fn prepare_ocean_h0_image(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    query: Query<
        (Entity, &FftSource),
        (With<crate::ocean::OceanSimSettings>, Without<OceanH0Image>),
    >,
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
            | TextureUsages::COPY_DST;
        let h = images.add(image);
        commands.entity(entity).insert(OceanH0Image { texture: h });
    }
}

pub(super) fn sync_ocean_h0_uniform(
    mut q: Query<(&crate::ocean::OceanSimSettings, &mut OceanH0Uniform)>,
) {
    for (s, mut u) in &mut q {
        u.texture_size = s.texture_size;
        u.tile_size = s.tile_size;
        u.wind_direction = s.wind_direction;
        u.wind_speed = s.wind_speed;
        u.peak_enhancement = s.peak_enhancement;
        u.directional_spread = s.directional_spread;
        u.small_wave_cutoff = s.small_wave_cutoff;
        u.gravity = s.gravity;
        u.amplitude_scale = s.amplitude_scale;
        u.h0_serial = s.h0_serial;
    }
}

pub(super) fn sync_ocean_dynamic_uniform(
    time: Res<Time>,
    mut q: Query<(&crate::ocean::OceanSimSettings, &mut OceanDynamicUniform)>,
) {
    for (s, mut d) in &mut q {
        d.texture_size = s.texture_size;
        d.tile_size = s.tile_size;
        d.elapsed_seconds = time.elapsed_secs() * s.time_scale;
        d.gravity = s.gravity;
        d.wind_direction = s.wind_direction;
    }
}

type PrepareOceanBgQuery<'w, 's> = Query<
    'w,
    's,
    (
        Entity,
        &'static FftTextures,
        &'static OceanH0Image,
        &'static FftSettings,
    ),
    (
        With<FftBindGroups>,
        With<OceanH0Uniform>,
        With<OceanDynamicUniform>,
    ),
>;

pub(super) fn prepare_ocean_compute_bind_groups(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    pipeline_cache: Res<PipelineCache>,
    ocean_pipelines: Res<OceanComputePipelines>,
    h0_uniforms: Res<ComponentUniforms<OceanH0Uniform>>,
    dyn_uniforms: Res<ComponentUniforms<OceanDynamicUniform>>,
    gpu_images: Res<RenderAssets<GpuImage>>,
    query: PrepareOceanBgQuery,
) {
    let Some(h0_binding) = h0_uniforms.binding() else {
        return;
    };
    let Some(dyn_binding) = dyn_uniforms.binding() else {
        return;
    };

    let init_layout = pipeline_cache.get_bind_group_layout(&ocean_pipelines.init_layout);
    let dyn_layout = pipeline_cache.get_bind_group_layout(&ocean_pipelines.spectrum_layout_dynamic);
    let h0_layout = pipeline_cache.get_bind_group_layout(&ocean_pipelines.spectrum_layout_h0);

    for (entity, _textures, ocean_h0, _fft_settings) in &query {
        let Some(h0_view) = gpu_images.get(&ocean_h0.texture).map(|g| &g.texture_view) else {
            continue;
        };

        let init = render_device.create_bind_group(
            "ocean_init_bind_group",
            &init_layout,
            &BindGroupEntries::sequential((h0_view, h0_binding.clone())),
        );

        let spectrum_dynamic = render_device.create_bind_group(
            "ocean_spectrum_dyn_bind_group",
            &dyn_layout,
            &BindGroupEntries::sequential((dyn_binding.clone(),)),
        );

        let spectrum_h0_read = render_device.create_bind_group(
            "ocean_spectrum_h0_read_bind_group",
            &h0_layout,
            &BindGroupEntries::sequential((h0_view,)),
        );

        commands.entity(entity).insert(OceanComputeBindGroups {
            init,
            spectrum_dynamic,
            spectrum_h0_read,
        });
    }
}

pub struct OceanSpectrumNode {
    query: QueryState<(
        &'static FftBindGroups,
        &'static OceanComputeBindGroups,
        &'static FftSettings,
        &'static OceanH0Uniform,
    )>,
}

impl FromWorld for OceanSpectrumNode {
    fn from_world(world: &mut World) -> Self {
        Self {
            query: world.query(),
        }
    }
}

impl Node for OceanSpectrumNode {
    fn update(&mut self, world: &mut World) {
        self.query.update_archetypes(world);
    }

    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut bevy::render::renderer::RenderContext,
        world: &World,
    ) -> Result<(), NodeRunError> {
        let pl = world.resource::<OceanComputePipelines>();
        let cache = world.resource::<PipelineCache>();
        let tracker = world.resource::<OceanInitTracker>();

        let Some(init_pl) = cache.get_compute_pipeline(pl.init) else {
            return Ok(());
        };
        let Some(spec_pl) = cache.get_compute_pipeline(pl.spectrum) else {
            return Ok(());
        };

        let wg = 8u32;
        let enc = render_context.command_encoder();

        for (fft_bg, ocean_bg, settings, h0_uni) in self.query.iter_manual(world) {
            let nx = settings.size.x.div_ceil(wg);
            let ny = settings.size.y.div_ceil(wg);

            let current = *h0_uni;
            let needs_h0_init = {
                let guard = tracker.last_init_h0_uniform.lock().unwrap();
                guard.as_ref() != Some(&current)
            };
            if needs_h0_init {
                {
                    let mut pass = enc.begin_compute_pass(&ComputePassDescriptor {
                        label: Some("ocean_init_h0_pass"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(init_pl);
                    pass.set_bind_group(0, &ocean_bg.init, &[]);
                    pass.dispatch_workgroups(nx, ny, 1);
                }
                *tracker.last_init_h0_uniform.lock().unwrap() = Some(current);
            }

            {
                let mut pass = enc.begin_compute_pass(&ComputePassDescriptor {
                    label: Some("ocean_spectrum_to_c_pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(spec_pl);
                pass.set_bind_group(0, &fft_bg.common, &[]);
                pass.set_bind_group(1, &ocean_bg.spectrum_dynamic, &[]);
                pass.set_bind_group(2, &ocean_bg.spectrum_h0_read, &[]);
                pass.dispatch_workgroups(nx, ny, 1);
            }
        }

        Ok(())
    }
}
