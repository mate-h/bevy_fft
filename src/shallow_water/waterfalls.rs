//! Spray at waterfall lips: a compute pass over the controller's grid that finds the faces where
//! water pours over a drop in the bed, with the CMF10 waterfall test, and appends spray to the
//! [`WaterParticleQueue`] of the water particles. See `waterfalls.wesl`.

use bevy::{
    prelude::*,
    render::{
        render_asset::RenderAssets,
        render_resource::{
            binding_types::{storage_buffer_sized, texture_2d, uniform_buffer},
            encase::internal::{WriteInto, Writer},
            *,
        },
        renderer::{RenderContext, RenderDevice, RenderQueue},
        texture::GpuImage,
    },
};

use super::{ShallowWaterController, shaders};
use crate::particles::WaterParticleQueue;

/// Spray thrown off waterfalls in a [`ShallowWaterController`] grid, into the water particles.
/// Needs [`WaterParticlesPlugin`](crate::particles::WaterParticlesPlugin) and a placement.
#[derive(Clone, Copy, Debug, Reflect)]
pub struct ShallowWaterWaterfalls {
    /// Particles per cubic meter of water that pours over a lip. Zero turns the spray off.
    pub rate: f32,
    /// Bed drop across a face, in cell widths, beyond which water falls instead of flowing:
    /// 1.5 in CMF10.
    pub slope_cap: f32,
    /// Random speed added in any direction, as a share of the water's speed over the lip.
    pub spread: f32,
    /// Meters across a drop.
    pub size: f32,
    /// Seconds a drop lives.
    pub lifetime: f32,
}

impl Default for ShallowWaterWaterfalls {
    fn default() -> Self {
        Self {
            rate: 40.0,
            slope_cap: 1.5,
            spread: 0.3,
            size: 0.08,
            lifetime: 1.5,
        }
    }
}

#[derive(Clone, Copy, Default, ShaderType)]
struct GpuWaterfallParams {
    size: UVec2,
    origin: Vec2,
    cell_size: f32,
    datum: f32,
    dt: f32,
    slope_cap: f32,
    rate: f32,
    spread: f32,
    size_m: f32,
    lifetime: f32,
    seed: u32,
    _pad: UVec3,
}

#[derive(Resource)]
pub(crate) struct ShallowWaterWaterfallPipeline {
    layout: BindGroupLayoutDescriptor,
    emit: CachedComputePipelineId,
}

impl FromWorld for ShallowWaterWaterfallPipeline {
    fn from_world(world: &mut World) -> Self {
        let layout = BindGroupLayoutDescriptor::new(
            "shallow_water_waterfalls_layout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (
                    uniform_buffer::<GpuWaterfallParams>(false),
                    texture_2d(TextureSampleType::Float { filterable: false }),
                    texture_2d(TextureSampleType::Float { filterable: false }),
                    texture_2d(TextureSampleType::Float { filterable: false }),
                    storage_buffer_sized(false, None),
                ),
            ),
        );
        let shader = world
            .resource::<AssetServer>()
            .load(shaders::SHALLOW_WATER_WATERFALLS);
        let emit =
            world
                .resource::<PipelineCache>()
                .queue_compute_pipeline(ComputePipelineDescriptor {
                    label: Some("shallow_water_waterfalls".into()),
                    layout: vec![layout.clone()],
                    immediate_size: 0,
                    shader,
                    shader_defs: vec![],
                    entry_point: Some("emitWaterfalls".into()),
                    zero_initialize_workgroup_memory: false,
                    constants: vec![],
                });
        Self { layout, emit }
    }
}

/// This frame's bind group and workgroups, when the spray runs.
#[derive(Resource, Default)]
pub(crate) struct ShallowWaterWaterfallGpu {
    params: Option<Buffer>,
    group: Option<BindGroup>,
    groups: UVec2,
    frame: u32,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn prepare_shallow_water_waterfalls(
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    cache: Res<PipelineCache>,
    pipeline: Res<ShallowWaterWaterfallPipeline>,
    images: Res<RenderAssets<GpuImage>>,
    controller: Res<ShallowWaterController>,
    particle_queue: Option<Res<WaterParticleQueue>>,
    mut gpu: ResMut<ShallowWaterWaterfallGpu>,
) {
    gpu.group = None;
    let (Some(waterfalls), Some(placement), Some(particle_queue)) =
        (controller.waterfalls, controller.placement, particle_queue)
    else {
        return;
    };
    if waterfalls.rate <= 0.0 || controller.paused {
        return;
    }
    let (Some(bed_water), Some(face_u), Some(face_w)) = (
        images.get(&controller.bed_water),
        images.get(&controller.flow_x),
        images.get(&controller.flow_y),
    ) else {
        return;
    };
    let cells = UVec2::new(controller.cells_x, controller.cells_y);
    let params = GpuWaterfallParams {
        size: cells,
        origin: placement.center - 0.5 * cells.as_vec2() * placement.cell_size,
        cell_size: placement.cell_size,
        datum: placement.datum,
        dt: controller.dt,
        slope_cap: waterfalls.slope_cap,
        rate: waterfalls.rate,
        spread: waterfalls.spread,
        size_m: waterfalls.size,
        lifetime: waterfalls.lifetime,
        seed: gpu.frame.wrapping_mul(0x9e37_79b9),
        _pad: UVec3::ZERO,
    };
    gpu.frame = gpu.frame.wrapping_add(1);
    let buffer = gpu
        .params
        .get_or_insert_with(|| {
            device.create_buffer(&BufferDescriptor {
                label: Some("shallow_water_waterfall_params"),
                size: GpuWaterfallParams::min_size().get(),
                usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        })
        .clone();
    let mut bytes = Vec::new();
    params.write_into(&mut Writer::new(&params, &mut bytes, 0).unwrap());
    queue.write_buffer(&buffer, 0, &bytes);
    gpu.groups = UVec2::new(cells.x.div_ceil(16), cells.y.div_ceil(16));
    gpu.group = Some(device.create_bind_group(
        "shallow_water_waterfalls",
        &cache.get_bind_group_layout(&pipeline.layout),
        &BindGroupEntries::sequential((
            buffer.as_entire_buffer_binding(),
            &bed_water.texture_view,
            &face_u.texture_view,
            &face_w.texture_view,
            particle_queue.buffer.as_entire_buffer_binding(),
        )),
    ));
}

pub(crate) fn run_shallow_water_waterfalls(
    mut ctx: RenderContext,
    pipeline: Res<ShallowWaterWaterfallPipeline>,
    cache: Res<PipelineCache>,
    gpu: Res<ShallowWaterWaterfallGpu>,
) {
    let (Some(group), Some(emit)) = (&gpu.group, cache.get_compute_pipeline(pipeline.emit)) else {
        return;
    };
    let mut pass = ctx
        .command_encoder()
        .begin_compute_pass(&ComputePassDescriptor {
            label: Some("shallow_water_waterfalls_pass"),
            timestamp_writes: None,
        });
    pass.set_pipeline(emit);
    pass.set_bind_group(0, group, &[]);
    pass.dispatch_workgroups(gpu.groups.x, gpu.groups.y, 1);
}
