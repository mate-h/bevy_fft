//! GPU height queries against the summed ocean cascades.
//!
//! Gameplay and physics code often needs the water surface at a few points: a boat's hull probes,
//! a swimmer's head, a camera that must stay above the waves. [`OceanQueries`] holds those points.
//! Each frame they are uploaded, a compute pass samples the shared
//! [`WaterLodData`](crate::water::WaterLodData) that the [`crate::ocean::OceanSurfaceMaterial`]
//! draws, and Bevy's [`Readback`] returns the results asynchronously.
//!
//! Results arrive one to three frames after the points were submitted. [`OceanQueryResults`]
//! carries the positions that were actually sampled and the frame they belong to, so callers can
//! compensate for the latency.

use std::collections::VecDeque;

use bevy::{
    asset::RenderAssetUsages,
    prelude::*,
    render::{
        RenderApp,
        extract_resource::ExtractResource,
        gpu_readback::{Readback, ReadbackComplete},
        render_asset::RenderAssets,
        render_resource::{binding_types::*, *},
        renderer::{RenderContext, RenderDevice},
        storage::{GpuShaderBuffer, ShaderBuffer, ShaderBufferData},
        texture::GpuImage,
    },
};

use crate::{
    ocean::{OceanSurfaceMaterial, OceanSurfaceTag},
    water::{WaterLodData, WaterLodGpu, WaterLodSlices},
};

/// Most points one frame can query.
pub const OCEAN_QUERY_MAX_POINTS: usize = 1024;

/// Byte size of the input header in `query.wesl` (`QueryHeader`).
const INPUT_HEADER_BYTES: usize = 16;
/// Byte size of the output header in `query.wesl` (`vec4<u32>`).
const OUTPUT_HEADER_BYTES: usize = 16;
/// Submitted frames kept for pairing results with their positions. Covers the readback latency.
const PENDING_FRAMES: usize = 8;

/// The ocean surface at one query point.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct OceanSample {
    /// Water surface height in meters above the ocean's rest level.
    pub height: f32,
    /// Surface slope `(∂h/∂x, ∂h/∂z)`.
    pub slope: Vec2,
    /// Horizontal chop offset in meters along the wind direction at the sampled point.
    pub chop: f32,
    /// Rate of change of [`Self::height`] in m/s, from this result and the previous one for the
    /// same point. Zero until two results for the point have arrived.
    pub vertical_velocity: f32,
}

impl OceanSample {
    /// Height extrapolated `seconds` past the time the sample was taken. Use it with
    /// the time since [`OceanQueryResults::time`] to hide the readback latency.
    pub fn height_after(&self, seconds: f32) -> f32 {
        self.height + self.vertical_velocity * seconds
    }

    /// Unit surface normal from the slope.
    pub fn normal(&self) -> Vec3 {
        Vec3::new(-self.slope.x, 1.0, -self.slope.y).normalize()
    }
}

/// One frame of returned samples, paired with the world `xz` positions they were taken at.
pub struct OceanQueryResults<'a> {
    /// Value of [`OceanQueries::frame`] when these points were submitted.
    pub frame: u32,
    /// [`Time::elapsed_secs`] when these points were submitted.
    pub time: f32,
    pub points: &'a [Vec2],
    pub samples: &'a [OceanSample],
}

/// Points to sample on the ocean surface each frame, and the latest results.
///
/// Set points with [`Self::set_points`]. They stay until replaced or cleared, so a fixed set of
/// probes needs to be set only when it moves. Read results with [`Self::results`].
#[derive(Resource)]
pub struct OceanQueries {
    points: Vec<Vec2>,
    frame: u32,
    pending: VecDeque<(u32, f32, Vec<Vec2>)>,
    results_frame: Option<u32>,
    results_time: f32,
    results_points: Vec<Vec2>,
    results: Vec<OceanSample>,
    input: Handle<ShaderBuffer>,
    output: Handle<ShaderBuffer>,
}

impl OceanQueries {
    /// Replaces the query points (world `x`, `z` in meters). Extra points past
    /// [`OCEAN_QUERY_MAX_POINTS`] are dropped.
    pub fn set_points(&mut self, points: impl IntoIterator<Item = Vec2>) {
        self.points.clear();
        self.points
            .extend(points.into_iter().take(OCEAN_QUERY_MAX_POINTS));
    }

    /// Current query points.
    pub fn points(&self) -> &[Vec2] {
        &self.points
    }

    /// Frame counter of the most recent submission. Compare with [`OceanQueryResults::frame`] to
    /// measure latency in frames.
    pub fn frame(&self) -> u32 {
        self.frame
    }

    /// Latest returned samples, or `None` before the first readback arrives.
    pub fn results(&self) -> Option<OceanQueryResults<'_>> {
        self.results_frame.map(|frame| OceanQueryResults {
            frame,
            time: self.results_time,
            points: &self.results_points,
            samples: &self.results,
        })
    }

    fn accept_readback(&mut self, data: &[u8]) {
        let Some(header) = data.get(..OUTPUT_HEADER_BYTES) else {
            return;
        };
        let frame = read_u32(header, 0);
        let count = read_u32(header, 4) as usize;
        if self.results_frame.is_some_and(|f| f >= frame) {
            return;
        }
        let Some(pos) = self.pending.iter().position(|(f, _, _)| *f == frame) else {
            return;
        };
        let (_, time, points) = self.pending.remove(pos).unwrap();
        self.pending.retain(|(f, _, _)| *f > frame);

        let samples = &data[OUTPUT_HEADER_BYTES..];
        let count = count.min(points.len()).min(samples.len() / 16);
        let dt = time - self.results_time;
        let previous = std::mem::take(&mut self.results);
        let previous_points = std::mem::take(&mut self.results_points);
        let mut points = points;
        points.truncate(count);
        self.results.extend((0..count).map(|i| {
            let o = i * 16;
            let height = read_f32(samples, o);
            let vertical_velocity = match previous.get(i) {
                Some(prev) if dt > 0.0 && previous_points.get(i) == points.get(i) => {
                    (height - prev.height) / dt
                }
                _ => 0.0,
            };
            OceanSample {
                height,
                slope: Vec2::new(read_f32(samples, o + 4), read_f32(samples, o + 8)),
                chop: read_f32(samples, o + 12),
                vertical_velocity,
            }
        }));
        self.results_points = points;
        self.results_frame = Some(frame);
        self.results_time = time;
    }
}

fn read_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
}

fn read_f32(bytes: &[u8], offset: usize) -> f32 {
    f32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
}

/// Marks the entity that carries the query output [`Readback`].
#[derive(Component)]
pub struct OceanQueryReadback;

pub(super) fn setup_ocean_queries(
    mut commands: Commands,
    mut buffers: ResMut<Assets<ShaderBuffer>>,
) {
    let input_bytes = INPUT_HEADER_BYTES + OCEAN_QUERY_MAX_POINTS * 8;
    let output_bytes = OUTPUT_HEADER_BYTES + OCEAN_QUERY_MAX_POINTS * 16;
    let input = buffers.add(ShaderBuffer::new(
        vec![0u8; input_bytes],
        RenderAssetUsages::default(),
    ));
    let output = buffers.add(ShaderBuffer::with_size(
        output_bytes as u64,
        RenderAssetUsages::RENDER_WORLD,
    ));
    commands
        .spawn((OceanQueryReadback, Name::new("Ocean query readback")))
        .observe(
            |event: On<ReadbackComplete>, mut queries: ResMut<OceanQueries>| {
                queries.accept_readback(&event.data);
            },
        );
    commands.insert_resource(OceanQueries {
        points: Vec::new(),
        frame: 0,
        pending: VecDeque::new(),
        results_frame: None,
        results_time: 0.0,
        results_points: Vec::new(),
        results: Vec::new(),
        input,
        output,
    });
}

/// Render-world view of the query state for one frame.
#[derive(Resource, Clone, Default, ExtractResource)]
#[extract_app(RenderApp)]
pub struct OceanQueryFrame {
    pub(super) count: u32,
    pub(super) input: Handle<ShaderBuffer>,
    pub(super) output: Handle<ShaderBuffer>,
    pub(super) lod_displacement: Handle<Image>,
}

#[allow(clippy::too_many_arguments)]
pub(super) fn submit_ocean_queries(
    mut commands: Commands,
    mut queries: ResMut<OceanQueries>,
    mut frame_out: ResMut<OceanQueryFrame>,
    mut buffers: ResMut<Assets<ShaderBuffer>>,
    lod: Option<Res<WaterLodData>>,
    surface: Query<&MeshMaterial3d<OceanSurfaceMaterial>, With<OceanSurfaceTag>>,
    materials: Res<Assets<OceanSurfaceMaterial>>,
    readback: Query<(Entity, Has<Readback>), With<OceanQueryReadback>>,
    time: Res<Time>,
) {
    frame_out.count = 0;
    let Ok((readback_entity, has_readback)) = readback.single() else {
        return;
    };
    let material = surface.iter().next().and_then(|m| materials.get(&m.0));
    let (Some(material), Some(lod), false) = (material, lod, queries.points.is_empty()) else {
        if has_readback {
            commands.entity(readback_entity).remove::<Readback>();
        }
        return;
    };

    queries.frame = queries.frame.wrapping_add(1).max(1);
    let frame = queries.frame;
    let count = queries.points.len() as u32;

    let mut bytes = Vec::with_capacity(INPUT_HEADER_BYTES + queries.points.len() * 8);
    bytes.extend_from_slice(&frame.to_le_bytes());
    bytes.extend_from_slice(&count.to_le_bytes());
    bytes.extend_from_slice(&material.extension.settings.wind_direction.to_le_bytes());
    bytes.extend_from_slice(&0u32.to_le_bytes());
    for p in &queries.points {
        bytes.extend_from_slice(&p.x.to_le_bytes());
        bytes.extend_from_slice(&p.y.to_le_bytes());
    }
    if let Some(mut buffer) = buffers.get_mut(&queries.input) {
        buffer.clear();
        buffer.extend_from_slice(&bytes);
        if let ShaderBufferData::Initialized { buffer_size, .. } = &mut buffer.data {
            *buffer_size = (INPUT_HEADER_BYTES + OCEAN_QUERY_MAX_POINTS * 8) as u64;
        }
    }

    let points = queries.points.clone();
    queries
        .pending
        .push_back((frame, time.elapsed_secs(), points));
    while queries.pending.len() > PENDING_FRAMES {
        queries.pending.pop_front();
    }

    frame_out.count = count;
    frame_out.input = queries.input.clone();
    frame_out.output = queries.output.clone();
    frame_out.lod_displacement = lod.displacement.clone();

    if !has_readback {
        commands
            .entity(readback_entity)
            .insert(Readback::buffer(queries.output.clone()));
    }
}

#[derive(Resource)]
pub struct OceanQueryPipeline {
    layout: BindGroupLayoutDescriptor,
    pipeline: CachedComputePipelineId,
}

impl FromWorld for OceanQueryPipeline {
    fn from_world(world: &mut World) -> Self {
        let pipeline_cache = world.resource::<PipelineCache>();
        let asset_server = world.resource::<AssetServer>();
        let entries = BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                storage_buffer_read_only_sized(false, None),
                storage_buffer_sized(false, None),
                texture_2d_array(TextureSampleType::Float { filterable: false }),
                uniform_buffer::<WaterLodSlices>(false),
            ),
        );
        let layout = BindGroupLayoutDescriptor::new("ocean_query_layout", &entries);
        let shader = asset_server.load(super::shaders::OCEAN_QUERY);
        let pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("ocean_query".into()),
            layout: vec![layout.clone()],
            immediate_size: 0,
            shader,
            shader_defs: vec![],
            entry_point: Some("ocean_query".into()),
            zero_initialize_workgroup_memory: false,
            constants: vec![],
        });
        Self { layout, pipeline }
    }
}

#[derive(Resource, Default)]
pub struct OceanQueryBindGroup(Option<BindGroup>);

#[allow(clippy::too_many_arguments)]
pub(super) fn prepare_ocean_query_bind_group(
    render_device: Res<RenderDevice>,
    pipeline_cache: Res<PipelineCache>,
    pipeline: Res<OceanQueryPipeline>,
    frame: Res<OceanQueryFrame>,
    buffers: Res<RenderAssets<GpuShaderBuffer>>,
    images: Res<RenderAssets<GpuImage>>,
    lod_gpu: Res<WaterLodGpu>,
    mut bind_group: ResMut<OceanQueryBindGroup>,
) {
    bind_group.0 = None;
    if frame.count == 0 {
        return;
    }
    let (Some(input), Some(output)) = (buffers.get(&frame.input), buffers.get(&frame.output))
    else {
        return;
    };
    let (Some(lod_view), Some(slices)) = (
        images.get(&frame.lod_displacement).map(|g| &g.texture_view),
        lod_gpu.slices.binding(),
    ) else {
        return;
    };
    let layout = pipeline_cache.get_bind_group_layout(&pipeline.layout);
    bind_group.0 = Some(render_device.create_bind_group(
        "ocean_query_bind_group",
        &layout,
        &BindGroupEntries::sequential((
            input.buffer.as_entire_binding(),
            output.buffer.as_entire_binding(),
            lod_view,
            slices,
        )),
    ));
}

pub fn run_ocean_query(
    mut ctx: RenderContext,
    pipeline: Res<OceanQueryPipeline>,
    cache: Res<PipelineCache>,
    frame: Res<OceanQueryFrame>,
    bind_group: Res<OceanQueryBindGroup>,
) {
    let Some(bind_group) = &bind_group.0 else {
        return;
    };
    let Some(compute) = cache.get_compute_pipeline(pipeline.pipeline) else {
        return;
    };
    let mut pass = ctx
        .command_encoder()
        .begin_compute_pass(&ComputePassDescriptor {
            label: Some("ocean_query_pass"),
            timestamp_writes: None,
        });
    pass.set_pipeline(compute);
    pass.set_bind_group(0, bind_group, &[]);
    pass.dispatch_workgroups(frame.count.div_ceil(64), 1, 1);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn queries_with_pending(frame: u32, points: Vec<Vec2>) -> OceanQueries {
        OceanQueries {
            points: points.clone(),
            frame,
            pending: VecDeque::from([(frame, frame as f32, points)]),
            results_frame: None,
            results_time: 0.0,
            results_points: Vec::new(),
            results: Vec::new(),
            input: Handle::default(),
            output: Handle::default(),
        }
    }

    fn readback_bytes(frame: u32, samples: &[[f32; 4]]) -> Vec<u8> {
        let mut bytes = Vec::new();
        for v in [frame, samples.len() as u32, 0, 0] {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        for s in samples {
            for v in s {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
        }
        bytes
    }

    #[test]
    fn readback_pairs_samples_with_submitted_points() {
        let points = vec![Vec2::new(1.0, 2.0), Vec2::new(-3.0, 4.0)];
        let mut q = queries_with_pending(7, points.clone());
        q.accept_readback(&readback_bytes(
            7,
            &[[0.5, 0.1, -0.2, 0.3], [-1.0, 0.0, 0.0, 0.0]],
        ));
        let r = q.results().expect("results");
        assert_eq!(r.frame, 7);
        assert_eq!(r.points, points.as_slice());
        assert_eq!(r.samples[0].height, 0.5);
        assert_eq!(r.samples[0].slope, Vec2::new(0.1, -0.2));
        assert_eq!(r.samples[1].height, -1.0);
    }

    #[test]
    fn readback_ignores_unknown_and_stale_frames() {
        let mut q = queries_with_pending(3, vec![Vec2::ZERO]);
        q.accept_readback(&readback_bytes(99, &[[1.0, 0.0, 0.0, 0.0]]));
        assert!(q.results().is_none());
        q.accept_readback(&readback_bytes(3, &[[2.0, 0.0, 0.0, 0.0]]));
        q.pending.push_back((2, 2.0, vec![Vec2::ONE]));
        q.accept_readback(&readback_bytes(2, &[[5.0, 0.0, 0.0, 0.0]]));
        assert_eq!(q.results().unwrap().samples[0].height, 2.0);
    }

    #[test]
    fn consecutive_results_give_vertical_velocity() {
        let p = vec![Vec2::ZERO];
        let mut q = queries_with_pending(4, p.clone());
        q.accept_readback(&readback_bytes(4, &[[1.0, 0.0, 0.0, 0.0]]));
        q.pending.push_back((6, 6.0, p));
        q.accept_readback(&readback_bytes(6, &[[2.0, 0.0, 0.0, 0.0]]));
        let s = q.results().unwrap().samples[0];
        assert_eq!(s.vertical_velocity, 0.5);
        assert_eq!(s.height_after(2.0), 3.0);
    }

    #[test]
    fn flat_sample_normal_points_up() {
        assert_eq!(OceanSample::default().normal(), Vec3::Y);
    }
}
