//! Writes the ocean FFT cascades into the shared [`WaterLodData`] slices each frame.

use bevy::{
    prelude::*,
    render::{
        RenderApp,
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
        OceanSurfaceMaterial, OceanSurfaceTag,
    },
    water::{WaterLodData, WaterLodGpu, WaterLodSlices},
};

/// Uniform for `lod_write.wesl`.
#[derive(Clone, Copy, Default, ShaderType)]
pub struct OceanLodWriteParams {
    pub cascade_tile_sizes: Vec4,
    pub cascade_count: u32,
    pub amplitude: f32,
    pub choppiness: f32,
    /// Water depth in meters, written to the foam depth lane.
    pub depth: f32,
}

/// What the writer needs from the main world this frame.
#[derive(Resource, Clone, Default, ExtractResource)]
#[extract_app(RenderApp)]
pub struct OceanLodWriteFrame {
    pub(super) params: OceanLodWriteParams,
    pub(super) displacement: Vec<Handle<Image>>,
    pub(super) foam: Vec<Handle<Image>>,
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
    )>,
    surface: Query<&MeshMaterial3d<OceanSurfaceMaterial>, With<OceanSurfaceTag>>,
    materials: Res<Assets<OceanSurfaceMaterial>>,
) {
    frame.displacement.clear();
    frame.foam.clear();
    frame.params.cascade_count = 0;
    let Some(material) = surface.iter().next().and_then(|m| materials.get(&m.0)) else {
        return;
    };

    let mut slots: Vec<_> = cascades
        .iter()
        .filter_map(|(sim, tex, mask, phase, cascade)| {
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
                )
            })
        })
        .collect();
    slots.sort_by_key(|s| s.0);
    let Some(first) = slots.first().cloned() else {
        return;
    };

    let mut tiles = Vec4::ONE;
    for (i, (_, tile, disp, foam, _)) in slots.iter().enumerate() {
        tiles[i] = *tile;
        frame.displacement.push(disp.clone());
        frame.foam.push(foam.clone());
    }
    // Unused bindings repeat cascade 0 so the bind group is always complete.
    while frame.displacement.len() < OCEAN_MAX_CASCADES {
        frame.displacement.push(first.2.clone());
        frame.foam.push(first.3.clone());
    }
    let settings = &material.extension.settings;
    frame.params = OceanLodWriteParams {
        cascade_tile_sizes: tiles,
        cascade_count: slots.len() as u32,
        amplitude: settings.amplitude,
        choppiness: settings.choppiness,
        depth: first.4,
    };
}

#[derive(Resource)]
pub struct OceanLodWritePipeline {
    layout: BindGroupLayoutDescriptor,
    pipeline: CachedComputePipelineId,
}

impl FromWorld for OceanLodWritePipeline {
    fn from_world(world: &mut World) -> Self {
        let pipeline_cache = world.resource::<PipelineCache>();
        let asset_server = world.resource::<AssetServer>();
        let tex = || texture_2d(TextureSampleType::Float { filterable: false });
        let entries = BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                uniform_buffer::<OceanLodWriteParams>(false),
                uniform_buffer::<WaterLodSlices>(false),
                texture_storage_2d_array(
                    TextureFormat::Rgba32Float,
                    StorageTextureAccess::WriteOnly,
                ),
                texture_storage_2d_array(
                    TextureFormat::Rgba32Float,
                    StorageTextureAccess::WriteOnly,
                ),
                tex(),
                tex(),
                tex(),
                tex(),
                tex(),
                tex(),
                tex(),
                tex(),
            ),
        );
        let layout = BindGroupLayoutDescriptor::new("ocean_lod_write_layout", &entries);
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
        Self { layout, pipeline }
    }
}

#[derive(Resource, Default)]
pub struct OceanLodWriteBindGroup {
    group: Option<BindGroup>,
    params: UniformBuffer<OceanLodWriteParams>,
    dispatch: UVec3,
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
    let Some(lod) = lod else {
        return;
    };
    if frame.params.cascade_count == 0 || frame.displacement.len() < OCEAN_MAX_CASCADES {
        return;
    }
    let (Some(disp_out), Some(foam_out)) = (images.get(&lod.displacement), images.get(&lod.foam))
    else {
        return;
    };
    let views: Option<Vec<_>> = frame
        .displacement
        .iter()
        .chain(frame.foam.iter())
        .map(|h| images.get(h).map(|g| &g.texture_view))
        .collect();
    let (Some(views), Some(slices)) = (views, lod_gpu.slices.binding()) else {
        return;
    };
    state.params.set(frame.params);
    state.params.write_buffer(&device, &queue);
    let Some(params) = state.params.binding() else {
        return;
    };
    let layout = pipeline_cache.get_bind_group_layout(&pipeline.layout);
    state.group = Some(device.create_bind_group(
        "ocean_lod_write_bind_group",
        &layout,
        &BindGroupEntries::sequential((
            params,
            slices,
            &disp_out.texture_view,
            &foam_out.texture_view,
            views[0],
            views[1],
            views[2],
            views[3],
            views[4],
            views[5],
            views[6],
            views[7],
        )),
    ));
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
    let Some(compute) = cache.get_compute_pipeline(pipeline.pipeline) else {
        return;
    };
    let mut pass = ctx
        .command_encoder()
        .begin_compute_pass(&ComputePassDescriptor {
            label: Some("ocean_lod_write_pass"),
            timestamp_writes: None,
        });
    pass.set_pipeline(compute);
    pass.set_bind_group(0, group, &[]);
    pass.dispatch_workgroups(state.dispatch.x, state.dispatch.y, state.dispatch.z);
}
