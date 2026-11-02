//! Writes the ocean FFT cascades into the shared [`WaterLodData`] slices each frame.
//!
//! Flow comes from the change in each cascade's displacement since the previous frame. In linear
//! wave theory a surface particle moves with the time derivative of its displacement, so the
//! difference of two frames gives the orbital velocity without extra FFT lanes. The writer keeps a
//! copy of every cascade's output for the next frame.
//!
//! Drift is the summed Stokes drift of the cascades, the same in every texel.

use bevy::{
    prelude::*,
    render::{
        RenderApp,
        diagnostic::RecordDiagnostics,
        extract_resource::ExtractResource,
        render_asset::RenderAssets,
        render_resource::{binding_types::*, *},
        renderer::{RenderContext, RenderDevice, RenderQueue},
        texture::GpuImage,
    },
};

use crate::{
    fft::FftTextures,
    ocean::{
        OCEAN_MAX_CASCADES, OceanCascade, OceanFoamMask, OceanFoamPhase, OceanSimSettings,
        OceanStokesDrift, OceanSurfaceSettings,
    },
    water::{
        WATER_LOD_DRIFT_FORMAT, WATER_LOD_SURFACE_FORMAT, WaterLodData, WaterLodGpu, WaterLodSlices,
    },
};

/// Uniform for `lod_write.wesl`.
#[derive(Clone, Copy, Default, ShaderType)]
pub struct OceanLodWriteParams {
    pub cascade_tile_sizes: Vec4,
    /// One for cascades whose waves count toward breaking, zero for the rest. See
    /// [`OceanSurfaceSettings::breaking_wavelength`].
    pub cascade_breaking: Vec4,
    pub cascade_count: u32,
    pub amplitude: f32,
    pub choppiness: f32,
    /// Water depth in meters, written to the foam depth lane.
    pub depth: f32,
    /// One over the simulated time since the previous frame, or zero when there is no previous
    /// frame to difference against.
    pub inv_dt: f32,
    pub _pad0: f32,
    /// Surface Stokes drift of all cascades in meters per second, after amplitude shaping.
    pub stokes_drift: Vec2,
    /// [`OceanSurfaceSettings::foam_cutoff`].
    pub foam_cutoff: f32,
    /// [`OceanSurfaceSettings::foam_falloff`].
    pub foam_falloff: f32,
    /// [`OceanSurfaceSettings::far_foam_cutoff`].
    pub far_foam_cutoff: f32,
    /// [`OceanSurfaceSettings::far_foam_texel`].
    pub far_foam_texel: f32,
}

/// What the writer needs from the main world this frame.
#[derive(Resource, Clone, Default, ExtractResource)]
#[extract_app(RenderApp)]
pub struct OceanLodWriteFrame {
    pub(crate) params: OceanLodWriteParams,
    pub(crate) displacement: Vec<Handle<Image>>,
    pub(crate) foam: Vec<Handle<Image>>,
    pub(crate) jacobian: Vec<Handle<Image>>,
}

#[allow(clippy::type_complexity)]
pub(super) fn gather_ocean_lod_write(
    mut frame: ResMut<OceanLodWriteFrame>,
    cascades: Query<(
        &OceanSimSettings,
        &FftTextures,
        &OceanFoamMask,
        &OceanFoamPhase,
        Option<&OceanCascade>,
        Option<&OceanStokesDrift>,
    )>,
    settings: Res<OceanSurfaceSettings>,
    time: Res<Time>,
) {
    frame.displacement.clear();
    frame.foam.clear();
    frame.jacobian.clear();
    frame.params.cascade_count = 0;

    let mut slots: Vec<_> = cascades
        .iter()
        .filter_map(|(sim, tex, mask, phase, cascade, drift)| {
            let slot = cascade.map_or(0, |c| c.index as usize);
            // Same half that `sync_ocean_foam_display` treats as most recent.
            let latest = if phase.0 % 2 == 0 {
                &mask.texture_b
            } else {
                &mask.texture_a
            };
            (slot < OCEAN_MAX_CASCADES).then(|| {
                (
                    slot,
                    sim.tile_size,
                    tex.spatial_output.clone(),
                    latest.clone(),
                    sim.depth,
                    sim.time_scale,
                    drift.map_or(Vec2::ZERO, |d| d.0),
                    sim.band_k_min,
                    mask.jacobian.clone(),
                )
            })
        })
        .collect();
    slots.sort_by_key(|s| s.0);
    let Some(first) = slots.first().cloned() else {
        return;
    };

    let mut tiles = Vec4::ONE;
    let mut breaking = Vec4::ZERO;
    let mut drift = Vec2::ZERO;
    let breaking_k = std::f32::consts::TAU / settings.breaking_wavelength.max(1e-6);
    for (i, (_, tile, disp, foam, _, _, cascade_drift, k_min, jacobian)) in slots.iter().enumerate()
    {
        tiles[i] = *tile;
        breaking[i] = if *k_min < breaking_k { 1.0 } else { 0.0 };
        drift += *cascade_drift;
        frame.displacement.push(disp.clone());
        frame.foam.push(foam.clone());
        frame.jacobian.push(jacobian.clone());
    }
    // Unused bindings repeat cascade 0 so the bind group is always complete.
    while frame.displacement.len() < OCEAN_MAX_CASCADES {
        frame.displacement.push(first.2.clone());
        frame.foam.push(first.3.clone());
        frame.jacobian.push(first.8.clone());
    }
    let dt = time.delta_secs() * first.5;
    frame.params = OceanLodWriteParams {
        cascade_tile_sizes: tiles,
        cascade_breaking: breaking,
        cascade_count: slots.len() as u32,
        amplitude: settings.amplitude,
        choppiness: settings.choppiness,
        depth: first.4,
        inv_dt: if dt > 0.0 { dt.recip() } else { 0.0 },
        // Drift goes with the square of wave height, so it takes the height gain twice.
        stokes_drift: drift * settings.amplitude * settings.amplitude,
        foam_cutoff: settings.foam_cutoff,
        foam_falloff: settings.foam_falloff,
        far_foam_cutoff: settings.far_foam_cutoff,
        far_foam_texel: settings.far_foam_texel,
        ..default()
    };
}

#[derive(Resource)]
pub struct OceanLodWritePipeline {
    layout: BindGroupLayoutDescriptor,
    pipeline: CachedComputePipelineId,
    /// Trilinear and repeating, for the cascades' Jacobian statistics.
    sampler: Sampler,
}

impl FromWorld for OceanLodWritePipeline {
    fn from_world(world: &mut World) -> Self {
        let pipeline_cache = world.resource::<PipelineCache>();
        let asset_server = world.resource::<AssetServer>();
        let tex = || texture_2d(TextureSampleType::Float { filterable: false });
        let filtered = || texture_2d(TextureSampleType::Float { filterable: true });
        let out = |format| texture_storage_2d_array(format, StorageTextureAccess::WriteOnly);
        let entries = BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                uniform_buffer::<OceanLodWriteParams>(false),
                uniform_buffer::<WaterLodSlices>(false),
                out(WATER_LOD_SURFACE_FORMAT),
                out(WATER_LOD_SURFACE_FORMAT),
                out(TextureFormat::Rgba32Float),
                out(WATER_LOD_DRIFT_FORMAT),
                tex(),
                tex(),
                tex(),
                tex(),
                tex(),
                tex(),
                tex(),
                tex(),
                tex(),
                tex(),
                tex(),
                tex(),
                filtered(),
                filtered(),
                filtered(),
                filtered(),
                sampler(SamplerBindingType::Filtering),
            ),
        );
        let layout = BindGroupLayoutDescriptor::new("ocean_lod_write_layout", &entries);
        let sampler = world
            .resource::<RenderDevice>()
            .create_sampler(&SamplerDescriptor {
                label: Some("ocean_jacobian_sampler"),
                address_mode_u: AddressMode::Repeat,
                address_mode_v: AddressMode::Repeat,
                mag_filter: FilterMode::Linear,
                min_filter: FilterMode::Linear,
                mipmap_filter: MipmapFilterMode::Linear,
                ..default()
            });
        let pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("ocean_lod_write".into()),
            layout: vec![layout.clone()],
            immediate_size: 0,
            shader: asset_server.load(super::shaders::OCEAN_LOD_WRITE),
            shader_defs: vec![],
            entry_point: Some("ocean_lod_write".into()),
            zero_initialize_workgroup_memory: false,
            constants: vec![],
        });
        Self {
            layout,
            pipeline,
            sampler,
        }
    }
}

/// Last frame's displacement for one cascade slot.
struct PreviousDisplacement {
    texture: Texture,
    view: TextureView,
    size: Extent3d,
}

#[derive(Resource, Default)]
pub struct OceanLodWriteBindGroup {
    group: Option<BindGroup>,
    params: UniformBuffer<OceanLodWriteParams>,
    dispatch: UVec3,
    previous: Vec<PreviousDisplacement>,
    /// Pairs of cascade output and previous copy to refresh after the pass.
    copies: Vec<(Texture, usize)>,
}

#[allow(clippy::too_many_arguments)]
pub(super) fn prepare_ocean_lod_write(
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    pipeline_cache: Res<PipelineCache>,
    pipeline: Res<OceanLodWritePipeline>,
    frame: Res<OceanLodWriteFrame>,
    lod: Option<Res<WaterLodData>>,
    lod_gpu: Res<WaterLodGpu>,
    images: Res<RenderAssets<GpuImage>>,
    mut state: ResMut<OceanLodWriteBindGroup>,
) {
    state.group = None;
    state.copies.clear();
    let Some(lod) = lod else {
        return;
    };
    if frame.params.cascade_count == 0 || frame.displacement.len() < OCEAN_MAX_CASCADES {
        return;
    }
    let (Some(disp_out), Some(foam_out), Some(flow_out), Some(drift_out)) = (
        images.get(&lod.displacement),
        images.get(&lod.foam),
        images.get(&lod.flow),
        images.get(&lod.drift),
    ) else {
        return;
    };
    let (Some(disp), Some(foam), Some(jacobian)) = (
        frame
            .displacement
            .iter()
            .map(|h| images.get(h))
            .collect::<Option<Vec<_>>>(),
        frame
            .foam
            .iter()
            .map(|h| images.get(h).map(|g| &g.texture_view))
            .collect::<Option<Vec<_>>>(),
        frame
            .jacobian
            .iter()
            .map(|h| images.get(h).map(|g| &g.texture_view))
            .collect::<Option<Vec<_>>>(),
    ) else {
        return;
    };
    let Some(slices) = lod_gpu.slices.binding() else {
        return;
    };

    // A new or resized copy holds no previous frame, so flow stays zero until the next one.
    let mut params = frame.params;
    let active = frame.params.cascade_count as usize;
    state.previous.truncate(OCEAN_MAX_CASCADES);
    for (i, source) in disp.iter().enumerate().take(OCEAN_MAX_CASCADES) {
        let size = source.texture_descriptor.size;
        if state.previous.get(i).is_some_and(|p| p.size == size) {
            continue;
        }
        let texture = device.create_texture(&TextureDescriptor {
            label: Some("ocean_lod_previous_displacement"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: source.texture_descriptor.format,
            usage: TextureUsages::COPY_DST | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&TextureViewDescriptor::default());
        let previous = PreviousDisplacement {
            texture,
            view,
            size,
        };
        if i < state.previous.len() {
            state.previous[i] = previous;
        } else {
            state.previous.push(previous);
        }
        if i < active {
            params.inv_dt = 0.0;
        }
    }
    state.copies = disp
        .iter()
        .take(active)
        .enumerate()
        .map(|(i, g)| (g.texture.clone(), i))
        .collect();

    state.params.set(params);
    state.params.write_buffer(&device, &queue);
    let Some(params) = state.params.binding() else {
        return;
    };
    let prev = &state.previous;
    let layout = pipeline_cache.get_bind_group_layout(&pipeline.layout);
    let group = device.create_bind_group(
        "ocean_lod_write_bind_group",
        &layout,
        &BindGroupEntries::sequential((
            params,
            slices,
            &disp_out.texture_view,
            &foam_out.texture_view,
            &flow_out.texture_view,
            &drift_out.texture_view,
            &disp[0].texture_view,
            &disp[1].texture_view,
            &disp[2].texture_view,
            &disp[3].texture_view,
            foam[0],
            foam[1],
            foam[2],
            foam[3],
            &prev[0].view,
            &prev[1].view,
            &prev[2].view,
            &prev[3].view,
            jacobian[0],
            jacobian[1],
            jacobian[2],
            jacobian[3],
            &pipeline.sampler,
        )),
    );
    state.group = Some(group);
    let res = lod.slices.resolution;
    state.dispatch = UVec3::new(res.div_ceil(8), res.div_ceil(8), lod.slices.count);
}

pub fn run_ocean_lod_write(
    mut ctx: RenderContext,
    pipeline: Res<OceanLodWritePipeline>,
    cache: Res<PipelineCache>,
    state: Res<OceanLodWriteBindGroup>,
) {
    let Some(group) = &state.group else {
        return;
    };
    let diagnostics = ctx.diagnostic_recorder();
    let diagnostics = diagnostics.as_deref();
    let encoder = ctx.command_encoder();
    let time_span = diagnostics.time_span(encoder, "ocean_lod_write");
    if let Some(compute) = cache.get_compute_pipeline(pipeline.pipeline) {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("ocean_lod_write_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(compute);
        pass.set_bind_group(0, group, &[]);
        pass.dispatch_workgroups(state.dispatch.x, state.dispatch.y, state.dispatch.z);
    }
    for (source, slot) in &state.copies {
        let previous = &state.previous[*slot];
        encoder.copy_texture_to_texture(
            source.as_image_copy(),
            previous.texture.as_image_copy(),
            previous.size,
        );
    }
    time_span.end(encoder);
}
