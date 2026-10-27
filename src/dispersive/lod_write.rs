//! Writes the dispersive patch into the shared [`WaterLodData`] slices each frame.

use bevy::{
    prelude::*,
    render::{
        render_asset::RenderAssets,
        render_resource::{binding_types::*, *},
        renderer::{RenderContext, RenderDevice, RenderQueue},
        texture::GpuImage,
    },
};

use super::{DispersiveController, DispersiveGridImages, DispersiveLodBlend, DispersiveSimRoot};
use crate::water::{
    WATER_LOD_SURFACE_FORMAT, WaterLodData, WaterLodGpu, WaterLodScratch, WaterLodSlices,
};

/// Uniform for `lod_write.wesl`.
#[derive(Clone, Copy, Default, ShaderType)]
pub struct DispersiveLodParams {
    pub center: Vec2,
    pub cell_size: f32,
    pub datum: f32,
    pub cells: u32,
    pub blend_cells: f32,
    pub blend: u32,
    pub _pad: u32,
}

#[derive(Resource)]
pub struct DispersiveLodPipeline {
    layout: BindGroupLayoutDescriptor,
    pipeline: CachedComputePipelineId,
}

impl FromWorld for DispersiveLodPipeline {
    fn from_world(world: &mut World) -> Self {
        let pipeline_cache = world.resource::<PipelineCache>();
        let asset_server = world.resource::<AssetServer>();
        let unfiltered = TextureSampleType::Float { filterable: false };
        let entries = BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                uniform_buffer::<DispersiveLodParams>(false),
                uniform_buffer::<WaterLodSlices>(false),
                texture_storage_2d_array(WATER_LOD_SURFACE_FORMAT, StorageTextureAccess::WriteOnly),
                texture_storage_2d_array(WATER_LOD_SURFACE_FORMAT, StorageTextureAccess::WriteOnly),
                texture_storage_2d_array(
                    TextureFormat::Rgba32Float,
                    StorageTextureAccess::WriteOnly,
                ),
                texture_2d_array(unfiltered),
                texture_2d_array(unfiltered),
                texture_2d_array(unfiltered),
                texture_2d(unfiltered),
                texture_2d(unfiltered),
            ),
        );
        let layout = BindGroupLayoutDescriptor::new("dispersive_lod_write_layout", &entries);
        let pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("dispersive_lod_write".into()),
            layout: vec![layout.clone()],
            immediate_size: 0,
            shader: asset_server.load(super::shaders::DISPERSIVE_LOD_WRITE),
            shader_defs: vec![],
            entry_point: Some("dispersive_lod_write".into()),
            zero_initialize_workgroup_memory: false,
            constants: vec![],
        });
        Self { layout, pipeline }
    }
}

#[derive(Resource, Default)]
pub struct DispersiveLodBindGroup {
    group: Option<BindGroup>,
    params: UniformBuffer<DispersiveLodParams>,
    dispatch: UVec3,
}

#[allow(clippy::too_many_arguments)]
pub(super) fn prepare_dispersive_lod_write(
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    pipeline_cache: Res<PipelineCache>,
    pipeline: Res<DispersiveLodPipeline>,
    controller: Option<Res<DispersiveController>>,
    grids: Query<&DispersiveGridImages, With<DispersiveSimRoot>>,
    lod: Option<Res<WaterLodData>>,
    lod_gpu: Res<WaterLodGpu>,
    scratch: Res<WaterLodScratch>,
    images: Res<RenderAssets<GpuImage>>,
    mut state: ResMut<DispersiveLodBindGroup>,
) {
    state.group = None;
    let (Some(controller), Some(lod), Ok(grid)) = (controller, lod, grids.single()) else {
        return;
    };
    let Some(placement) = controller.placement else {
        return;
    };
    let (Some(disp_out), Some(foam_out), Some(flow_out), Some(sim_state), Some(bed)) = (
        images.get(&lod.displacement),
        images.get(&lod.foam),
        images.get(&lod.flow),
        images.get(&grid.state),
        images.get(&grid.bed),
    ) else {
        return;
    };
    let (Some(disp_in), Some(foam_in), Some(flow_in), Some(slices)) = (
        scratch.displacement(),
        scratch.foam(),
        scratch.flow(),
        lod_gpu.slices.binding(),
    ) else {
        return;
    };
    state.params.set(DispersiveLodParams {
        center: placement.center,
        cell_size: controller.cell_size(),
        datum: placement.datum,
        cells: controller.n,
        blend_cells: placement.blend_cells,
        blend: match placement.blend {
            DispersiveLodBlend::Add => 0,
            DispersiveLodBlend::Replace => 1,
        },
        _pad: 0,
    });
    state.params.write_buffer(&device, &queue);
    let Some(params) = state.params.binding() else {
        return;
    };
    let layout = pipeline_cache.get_bind_group_layout(&pipeline.layout);
    state.group = Some(device.create_bind_group(
        "dispersive_lod_write_bind_group",
        &layout,
        &BindGroupEntries::sequential((
            params,
            slices,
            &disp_out.texture_view,
            &foam_out.texture_view,
            &flow_out.texture_view,
            disp_in,
            foam_in,
            flow_in,
            &sim_state.texture_view,
            &bed.texture_view,
        )),
    ));
    let res = lod.slices.resolution;
    state.dispatch = UVec3::new(res.div_ceil(8), res.div_ceil(8), lod.slices.count);
}

pub(super) fn run_dispersive_lod_write(
    mut ctx: RenderContext,
    pipeline: Res<DispersiveLodPipeline>,
    cache: Res<PipelineCache>,
    state: Res<DispersiveLodBindGroup>,
    scratch: Res<WaterLodScratch>,
    lod: Option<Res<WaterLodData>>,
    images: Res<RenderAssets<GpuImage>>,
) {
    let (Some(group), Some(lod)) = (&state.group, lod) else {
        return;
    };
    let Some(compute) = cache.get_compute_pipeline(pipeline.pipeline) else {
        return;
    };
    let encoder = ctx.command_encoder();
    if !scratch.copy_from_lod(encoder, &images, &lod) {
        return;
    }
    let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
        label: Some("dispersive_lod_write_pass"),
        timestamp_writes: None,
    });
    pass.set_pipeline(compute);
    pass.set_bind_group(0, group, &[]);
    pass.dispatch_workgroups(state.dispatch.x, state.dispatch.y, state.dispatch.z);
}
