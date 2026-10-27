//! Writes the ocean's long-wave height over a placed shallow water grid.
//!
//! When [`ShallowWaterController::ocean_forcing`] is on and the grid has a
//! [`ShallowWaterPlacement`](crate::shallow_water::ShallowWaterPlacement), this pass fills
//! [`ShallowWaterController::ocean_eta`] each frame before the solver steps. The solver's
//! [`ShallowWaterBorder::Ocean`](crate::shallow_water::ShallowWaterBorder::Ocean) band and its PML
//! blend toward that height, so ocean waves enter the grid and leave it without reflecting.

use bevy::{
    prelude::*,
    render::{
        diagnostic::RecordDiagnostics,
        render_asset::RenderAssets,
        render_resource::{binding_types::*, *},
        renderer::{RenderContext, RenderDevice, RenderQueue},
        texture::GpuImage,
    },
};

use super::{OCEAN_MAX_CASCADES, lod_write::OceanLodWriteFrame};
use crate::{shallow_water::ShallowWaterController, water::WaterLodData};

/// Uniform for `shallow_water_forcing.wesl`.
#[derive(Clone, Copy, Default, ShaderType)]
pub struct ShallowWaterForcingParams {
    pub cascade_tile_sizes: Vec4,
    pub grid_center: Vec2,
    pub chop_dir: Vec2,
    pub cells: UVec2,
    pub cell_size: f32,
    pub datum: f32,
    pub amplitude: f32,
    pub choppiness: f32,
    pub cascade_count: u32,
    pub edge_cells: u32,
}

#[derive(Resource)]
pub struct ShallowWaterForcingPipeline {
    layout: BindGroupLayoutDescriptor,
    pipeline: CachedComputePipelineId,
}

impl FromWorld for ShallowWaterForcingPipeline {
    fn from_world(world: &mut World) -> Self {
        let pipeline_cache = world.resource::<PipelineCache>();
        let asset_server = world.resource::<AssetServer>();
        let tex = || texture_2d(TextureSampleType::Float { filterable: false });
        let entries = BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                uniform_buffer::<ShallowWaterForcingParams>(false),
                texture_storage_2d(TextureFormat::R32Float, StorageTextureAccess::WriteOnly),
                tex(),
                tex(),
                tex(),
                tex(),
            ),
        );
        let layout = BindGroupLayoutDescriptor::new("shallow_water_forcing_layout", &entries);
        let pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("shallow_water_forcing".into()),
            layout: vec![layout.clone()],
            immediate_size: 0,
            shader: asset_server.load(super::shaders::OCEAN_SHALLOW_WATER_FORCING),
            shader_defs: vec![],
            entry_point: Some("shallow_water_forcing".into()),
            zero_initialize_workgroup_memory: false,
            constants: vec![],
        });
        Self { layout, pipeline }
    }
}

#[derive(Resource, Default)]
pub struct ShallowWaterForcingBindGroup {
    group: Option<BindGroup>,
    params: UniformBuffer<ShallowWaterForcingParams>,
    dispatch: UVec2,
}

#[allow(clippy::too_many_arguments)]
pub(super) fn prepare_shallow_water_forcing(
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    pipeline_cache: Res<PipelineCache>,
    pipeline: Res<ShallowWaterForcingPipeline>,
    frame: Res<OceanLodWriteFrame>,
    controller: Option<Res<ShallowWaterController>>,
    lod: Option<Res<WaterLodData>>,
    images: Res<RenderAssets<GpuImage>>,
    mut state: ResMut<ShallowWaterForcingBindGroup>,
) {
    state.group = None;
    let Some(controller) = controller else {
        return;
    };
    let Some(placement) = controller.placement else {
        return;
    };
    if !controller.ocean_forcing
        || frame.params.cascade_count == 0
        || frame.displacement.len() < OCEAN_MAX_CASCADES
    {
        return;
    }
    let Some(eta_out) = images.get(&controller.ocean_eta) else {
        return;
    };
    let Some(disp) = frame
        .displacement
        .iter()
        .map(|h| images.get(h).map(|g| &g.texture_view))
        .collect::<Option<Vec<_>>>()
    else {
        return;
    };

    let chop = lod.map_or(0.0, |l| l.slices.chop_direction);
    state.params.set(ShallowWaterForcingParams {
        cascade_tile_sizes: frame.params.cascade_tile_sizes,
        grid_center: placement.center,
        chop_dir: Vec2::from_angle(chop),
        cells: UVec2::new(controller.cells_x, controller.cells_y),
        cell_size: placement.cell_size,
        datum: placement.datum,
        amplitude: frame.params.amplitude,
        choppiness: frame.params.choppiness,
        cascade_count: frame.params.cascade_count,
        // The solver reads the target only in the PML and the ocean band.
        edge_cells: controller.pml_width + controller.ocean_band_cells.max(0.0).ceil() as u32 + 1,
    });
    state.params.write_buffer(&device, &queue);
    let Some(params) = state.params.binding() else {
        return;
    };
    let layout = pipeline_cache.get_bind_group_layout(&pipeline.layout);
    let group = device.create_bind_group(
        "shallow_water_forcing_bind_group",
        &layout,
        &BindGroupEntries::sequential((
            params,
            &eta_out.texture_view,
            disp[0],
            disp[1],
            disp[2],
            disp[3],
        )),
    );
    state.group = Some(group);
    state.dispatch = UVec2::new(
        controller.cells_x.div_ceil(16),
        controller.cells_y.div_ceil(16),
    );
}

pub fn run_shallow_water_forcing(
    mut ctx: RenderContext,
    pipeline: Res<ShallowWaterForcingPipeline>,
    cache: Res<PipelineCache>,
    state: Res<ShallowWaterForcingBindGroup>,
) {
    let Some(group) = &state.group else {
        return;
    };
    let Some(compute) = cache.get_compute_pipeline(pipeline.pipeline) else {
        return;
    };
    let diagnostics = ctx.diagnostic_recorder();
    let diagnostics = diagnostics.as_deref();
    let encoder = ctx.command_encoder();
    let time_span = diagnostics.time_span(encoder, "shallow_water_forcing");
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("shallow_water_forcing_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(compute);
        pass.set_bind_group(0, group, &[]);
        pass.dispatch_workgroups(state.dispatch.x, state.dispatch.y, 1);
    }
    time_span.end(encoder);
}
