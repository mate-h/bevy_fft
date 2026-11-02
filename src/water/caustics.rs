//! Caustic map from the water surface onto a level receiver.
//!
//! Waves act as lenses. Where the surface curves up, refracted sunlight converges, and at the
//! right depth it focuses into bright lines. [`WaterCaustics::map`] holds that light on a few
//! level receivers under the water, over one slice of the [`WaterLodData`]. A compute pass sends a
//! grid of sun rays through the displaced surface, refracts each by Snell's law, and adds its light
//! to the texels around its hit on every receiver. The map is intensity relative to a level
//! surface, so a calm sea gives one everywhere, and the ratio of the area a bundle of rays leaves
//! to the area it lands on sets how bright each texel gets.
//!
//! The receivers sit at depths that grow by a constant factor, the way the cascaded caustic maps
//! of Kougianos et al. (2021, §3.1) grow their extent by the level scale factor `s`. A point
//! between two receivers reads both where its refracted beam crosses them and blends the two by
//! depth.

use bevy::{
    asset::RenderAssetUsages,
    prelude::*,
    render::{
        RenderApp,
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        render_asset::RenderAssets,
        render_resource::{binding_types::*, *},
        renderer::{RenderContext, RenderDevice, RenderQueue},
        texture::GpuImage,
    },
};

use super::{WaterLodData, WaterLodGpu, WaterLodSettings, WaterLodSlices};

pub const WATER_CAUSTICS_SHADER: &str = "embedded://bevy_fft/water/caustics.wesl";

/// Format of [`WaterCaustics::map`].
pub const WATER_CAUSTICS_FORMAT: TextureFormat = TextureFormat::R32Float;

/// Layers of [`WaterCaustics::map`], the most receivers it holds.
pub const WATER_CAUSTICS_MAX_RECEIVERS: u32 = 4;

/// Where the caustic map is measured.
#[derive(Resource, Clone, Copy, Debug, Reflect)]
pub struct WaterCausticsSettings {
    /// Runs the pass. Off by default, since it traces `rays_per_texel` squared rays per texel of
    /// the slice every frame.
    pub enabled: bool,
    /// LOD slice the map covers. Coarser slices reach further but resolve only longer waves.
    pub slice: u32,
    /// Depth of the shallowest receiving plane below the water mesh plane, in meters.
    pub receiver_depth: f32,
    /// Receiving planes, from 1 to [`WATER_CAUSTICS_MAX_RECEIVERS`]. Each one traces the rays
    /// again, so the pass costs about this many times one receiver.
    pub receiver_count: u32,
    /// Ratio between the depths of neighbouring receivers, above one. Points between two receivers
    /// blend them, and points past the last one read the deepest.
    pub receiver_scale: f32,
    /// Rays along each edge of a texel. More rays smooth the map.
    pub rays_per_texel: u32,
    /// Index of refraction of the water.
    pub ior: f32,
}

impl Default for WaterCausticsSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            slice: 2,
            receiver_depth: 2.0,
            receiver_count: 3,
            receiver_scale: 2.0,
            rays_per_texel: 4,
            ior: 1.333,
        }
    }
}

/// Caustic intensity on each receiver, one layer per receiver from the shallowest down, and one
/// texel per texel of the chosen LOD slice, placed like it. Rewritten every frame before the
/// cameras draw. The underwater medium pass of
/// [`super::WaterUnderwater`] projects it onto the scene, and the single-scattered sunlight under
/// water follows it, which draws the light shafts below focusing waves.
impl WaterCausticsSettings {
    /// Depth of each receiver in meters, shallowest first. Lanes past
    /// [`Self::receiver_count`] repeat the deepest.
    pub fn receiver_depths(&self) -> Vec4 {
        let count = self.receiver_count.clamp(1, WATER_CAUSTICS_MAX_RECEIVERS);
        let scale = self.receiver_scale.max(1.0);
        Vec4::from_array(std::array::from_fn(|i| {
            self.receiver_depth * scale.powi(i.min(count as usize - 1) as i32)
        }))
    }
}

#[derive(Resource, Clone)]
pub struct WaterCaustics {
    pub map: Handle<Image>,
}

/// Layout matches `CausticsUniform` in `caustics.wesl`.
#[derive(Clone, Copy, Default, ShaderType)]
pub struct CausticsUniform {
    pub direction_to_light: Vec3,
    pub receiver_count: u32,
    pub receiver_depths: Vec4,
    pub slice: u32,
    pub rays: u32,
    pub ior: f32,
    pub _pad: f32,
}

#[derive(Resource, Clone, Default, ExtractResource)]
#[extract_app(RenderApp)]
pub struct WaterCausticsFrame {
    enabled: bool,
    uniform: CausticsUniform,
    lookup: CausticsLookup,
    map: Handle<Image>,
}

/// Where readers find the caustic map. Layout matches `CausticsLookup` in `underside.wesl`.
#[derive(Clone, Copy, Debug, Default, PartialEq, ShaderType)]
pub struct CausticsLookup {
    /// The covered slice: centre X, centre Z, texel width, and width in meters.
    pub rect: Vec4,
    /// One when the map was traced this frame, then the receiver count.
    pub params: Vec4,
    /// Depth of each receiver in meters, shallowest first.
    pub receiver_depths: Vec4,
}

impl WaterCausticsFrame {
    pub(super) fn lookup(&self) -> CausticsLookup {
        self.lookup
    }

    pub(super) fn map(&self) -> &Handle<Image> {
        &self.map
    }
}

pub(super) fn create_water_caustics(
    mut commands: Commands,
    settings: Res<WaterLodSettings>,
    mut images: ResMut<Assets<Image>>,
) {
    let mut image = Image::new_uninit(
        Extent3d {
            width: settings.resolution,
            height: settings.resolution,
            depth_or_array_layers: WATER_CAUSTICS_MAX_RECEIVERS,
        },
        TextureDimension::D2,
        WATER_CAUSTICS_FORMAT,
        RenderAssetUsages::RENDER_WORLD,
    );
    image.texture_descriptor.usage =
        TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_SRC;
    commands.insert_resource(WaterCaustics {
        map: images.add(image),
    });
}

/// Takes the sun from the first directional light.
pub(super) fn gather_water_caustics(
    settings: Res<WaterCausticsSettings>,
    caustics: Option<Res<WaterCaustics>>,
    lod: Option<Res<WaterLodData>>,
    lights: Query<&GlobalTransform, With<DirectionalLight>>,
    mut frame: ResMut<WaterCausticsFrame>,
) {
    frame.enabled = false;
    frame.lookup = CausticsLookup::default();
    let Some(caustics) = caustics else {
        return;
    };
    frame.map = caustics.map.clone();
    let (true, Some(lod), Some(sun)) = (settings.enabled, lod, lights.iter().next()) else {
        return;
    };
    let direction_to_light = sun.back().as_vec3();
    if direction_to_light.y <= 0.0 {
        return;
    }
    frame.enabled = true;
    let slice = settings.slice.min(lod.slices.count.saturating_sub(1));
    let receiver_count = settings
        .receiver_count
        .clamp(1, WATER_CAUSTICS_MAX_RECEIVERS);
    let receiver_depths = settings.receiver_depths();
    frame.uniform = CausticsUniform {
        direction_to_light,
        receiver_count,
        receiver_depths,
        slice,
        rays: settings.rays_per_texel.max(1),
        ior: settings.ior,
        _pad: 0.0,
    };
    frame.lookup = CausticsLookup {
        rect: lod.slices.slices[slice as usize],
        params: Vec4::new(1.0, receiver_count as f32, 0.0, 0.0),
        receiver_depths,
    };
}

#[derive(Resource)]
pub struct WaterCausticsPipeline {
    layout: BindGroupLayoutDescriptor,
    splat: CachedComputePipelineId,
    resolve: CachedComputePipelineId,
}

impl FromWorld for WaterCausticsPipeline {
    fn from_world(world: &mut World) -> Self {
        let pipeline_cache = world.resource::<PipelineCache>();
        let asset_server = world.resource::<AssetServer>();
        let entries = BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                uniform_buffer::<CausticsUniform>(false),
                uniform_buffer::<WaterLodSlices>(false),
                texture_2d_array(TextureSampleType::Float { filterable: true }),
                texture_2d_array(TextureSampleType::Float { filterable: true }),
                sampler(SamplerBindingType::Filtering),
                storage_buffer_sized(false, None),
                texture_storage_2d_array(WATER_CAUSTICS_FORMAT, StorageTextureAccess::WriteOnly),
            ),
        );
        let layout = BindGroupLayoutDescriptor::new("water_caustics_layout", &entries);
        let shader = asset_server.load(WATER_CAUSTICS_SHADER);
        let queue = |entry: &'static str| {
            pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some(entry.into()),
                layout: vec![layout.clone()],
                immediate_size: 0,
                shader: shader.clone(),
                shader_defs: vec![],
                entry_point: Some(entry.into()),
                zero_initialize_workgroup_memory: false,
                constants: vec![],
            })
        };
        Self {
            splat: queue("caustics_splat"),
            resolve: queue("caustics_resolve"),
            layout,
        }
    }
}

#[derive(Resource, Default)]
pub struct WaterCausticsGpu {
    uniform: UniformBuffer<CausticsUniform>,
    energy: Option<Buffer>,
    group: Option<BindGroup>,
    resolution: u32,
    rays: u32,
    receivers: u32,
}

#[allow(clippy::too_many_arguments)]
pub(super) fn prepare_water_caustics(
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    pipeline_cache: Res<PipelineCache>,
    pipeline: Res<WaterCausticsPipeline>,
    frame: Res<WaterCausticsFrame>,
    lod: Option<Res<WaterLodData>>,
    lod_gpu: Res<WaterLodGpu>,
    images: Res<RenderAssets<GpuImage>>,
    mut gpu: ResMut<WaterCausticsGpu>,
) {
    gpu.group = None;
    let Some(lod) = lod else {
        return;
    };
    if !frame.enabled {
        return;
    }
    let (Some(map), Some(displacement), Some(foam), Some(slices)) = (
        images.get(&frame.map),
        images.get(&lod.displacement),
        images.get(&lod.foam),
        lod_gpu.slices.binding(),
    ) else {
        return;
    };
    let resolution = lod.slices.resolution;
    let size =
        u64::from(resolution) * u64::from(resolution) * u64::from(WATER_CAUSTICS_MAX_RECEIVERS) * 4;
    if gpu.energy.as_ref().is_none_or(|b| b.size() != size) {
        gpu.energy = Some(device.create_buffer(&BufferDescriptor {
            label: Some("water_caustics_energy"),
            size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
    }
    gpu.uniform.set(frame.uniform);
    gpu.uniform.write_buffer(&device, &queue);
    let (Some(uniform), Some(energy)) = (gpu.uniform.binding(), gpu.energy.as_ref()) else {
        return;
    };
    let layout = pipeline_cache.get_bind_group_layout(&pipeline.layout);
    let group = device.create_bind_group(
        "water_caustics_bind_group",
        &layout,
        &BindGroupEntries::sequential((
            uniform,
            slices,
            &displacement.texture_view,
            &foam.texture_view,
            &displacement.sampler,
            energy.as_entire_binding(),
            &map.texture_view,
        )),
    );
    gpu.group = Some(group);
    gpu.resolution = resolution;
    gpu.rays = frame.uniform.rays;
    gpu.receivers = frame.uniform.receiver_count;
}

pub(super) fn run_water_caustics(
    mut ctx: RenderContext,
    pipeline: Res<WaterCausticsPipeline>,
    cache: Res<PipelineCache>,
    gpu: Res<WaterCausticsGpu>,
) {
    let (Some(group), Some(energy)) = (&gpu.group, &gpu.energy) else {
        return;
    };
    let (Some(splat), Some(resolve)) = (
        cache.get_compute_pipeline(pipeline.splat),
        cache.get_compute_pipeline(pipeline.resolve),
    ) else {
        return;
    };
    let encoder = ctx.command_encoder();
    encoder.clear_buffer(energy, 0, None);
    let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
        label: Some("water_caustics_pass"),
        timestamp_writes: None,
    });
    pass.set_bind_group(0, group, &[]);
    let rays = (gpu.resolution * gpu.rays).div_ceil(8);
    pass.set_pipeline(splat);
    pass.dispatch_workgroups(rays, rays, gpu.receivers);
    let texels = gpu.resolution.div_ceil(8);
    pass.set_pipeline(resolve);
    pass.dispatch_workgroups(texels, texels, gpu.receivers);
}

pub(super) fn configure(app: &mut App) {
    bevy::asset::embedded_asset!(app, "caustics.wesl");
    app.register_type::<WaterCausticsSettings>()
        .init_resource::<WaterCausticsSettings>()
        .init_resource::<WaterCausticsFrame>()
        .add_plugins(ExtractResourcePlugin::<WaterCausticsFrame>::default())
        .add_systems(PreStartup, create_water_caustics)
        .add_systems(
            PostUpdate,
            gather_water_caustics
                .after(super::WaterLodSystems)
                .after(bevy::transform::TransformSystems::Propagate),
        );
    if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
        render_app
            .init_resource::<WaterCausticsGpu>()
            .add_systems(
                bevy::render::Render,
                prepare_water_caustics.in_set(bevy::render::RenderSystems::PrepareBindGroups),
            )
            .add_systems(
                bevy::render::renderer::RenderGraph,
                run_water_caustics.in_set(super::WaterLodGraphSystems::Read),
            );
    }
}

pub(super) fn finish(app: &mut App) {
    if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
        render_app.init_resource::<WaterCausticsPipeline>();
    }
}
