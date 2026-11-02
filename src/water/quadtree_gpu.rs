//! Compute selection with indirect draws through Bevy's material phases.

use std::{any::TypeId, collections::HashMap};

use bevy::{
    asset::AssetId,
    camera::primitives::Frustum,
    core_pipeline::{
        core_3d::*,
        deferred::{AlphaMask3dDeferred, Opaque3dDeferred},
        prepass::*,
        schedule::Core3d,
    },
    ecs::{
        query::ROQueryItem,
        system::{SystemParamItem, lifetimeless::SRes},
    },
    pbr::*,
    prelude::*,
    render::{
        RenderApp,
        batching::gpu_preprocessing::IndirectParametersBuffers,
        diagnostic::RecordDiagnostics,
        extract_resource::ExtractResource,
        render_asset::RenderAssets,
        render_phase::*,
        render_resource::{binding_types::*, *},
        renderer::{RenderContext, RenderDevice, RenderQueue, ViewQuery},
        storage::{GpuShaderBuffer, ShaderBuffer},
    },
};

use super::WaterQuadtree;

#[derive(Clone, ShaderType)]
struct Params {
    eye: Vec4,
    shape: Vec4,
    limits: UVec4,
    heights: Vec4,
    planes: [Vec4; 6],
}

#[derive(Clone)]
pub(super) struct FrameEntry {
    params: Params,
    tiles: Handle<ShaderBuffer>,
    mesh: AssetId<Mesh>,
}

impl FrameEntry {
    pub(super) fn new(
        q: WaterQuadtree,
        eye: Vec3,
        frustum: Option<&Frustum>,
        tiles: Handle<ShaderBuffer>,
        mesh: AssetId<Mesh>,
    ) -> Self {
        Self {
            params: Params {
                eye: eye.extend(0.0),
                shape: Vec4::new(
                    q.leaf_size,
                    q.lod_range.max(q.min_lod_range()),
                    q.morph_start_ratio(),
                    q.horizontal_margin.max(0.0),
                ),
                limits: UVec4::new(
                    q.levels.clamp(1, 24),
                    q.max_tiles,
                    q.cells(),
                    u32::from(q.frustum_cull && frustum.is_some()),
                ),
                heights: q.height_bounds.extend(0.0).extend(0.0),
                planes: frustum.map_or([Vec4::ZERO; 6], |f| f.half_spaces.map(|p| p.normal_d())),
            },
            tiles,
            mesh,
        }
    }
}

#[derive(Resource, Clone, Default, ExtractResource)]
#[extract_app(RenderApp)]
pub(super) struct WaterQuadtreeFrame(pub Vec<FrameEntry>);

#[derive(Resource)]
pub(super) struct QuadtreePipeline {
    selection_layout: BindGroupLayoutDescriptor,
    draw_layout: BindGroupLayoutDescriptor,
    selection: [CachedComputePipelineId; 3],
    draws: CachedComputePipelineId,
}

impl FromWorld for QuadtreePipeline {
    fn from_world(world: &mut World) -> Self {
        let cache = world.resource::<PipelineCache>();
        let shader = world
            .resource::<AssetServer>()
            .load("embedded://bevy_fft/water/quadtree.wesl");
        let selection_layout = BindGroupLayoutDescriptor::new(
            "water_quadtree_selection",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (
                    uniform_buffer::<Params>(false),
                    storage_buffer_sized(false, None),
                ),
            ),
        );
        let draw_layout = BindGroupLayoutDescriptor::new(
            "water_quadtree_indirect",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (
                    storage_buffer_read_only_sized(false, None),
                    storage_buffer_sized(false, None),
                ),
            ),
        );
        let selection = ["reset", "refine", "finish"].map(|entry| {
            cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some(format!("water_quadtree_{entry}").into()),
                layout: vec![selection_layout.clone()],
                shader: shader.clone(),
                entry_point: Some(entry.into()),
                ..default()
            })
        });
        let draws = cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("water_quadtree_indirect".into()),
            layout: vec![selection_layout.clone(), draw_layout.clone()],
            shader,
            entry_point: Some("prepare_draws".into()),
            ..default()
        });
        Self {
            selection_layout,
            draw_layout,
            selection,
            draws,
        }
    }
}

struct PreparedQuadtree {
    params: UniformBuffer<Params>,
    tiles: Buffer,
    group: BindGroup,
    draws: HashMap<TypeId, PreparedDraw>,
}

struct PreparedDraw {
    source: Buffer,
    output: Buffer,
    group: BindGroup,
}

#[derive(Resource, Default)]
pub(super) struct GpuQuadtrees(HashMap<AssetId<Mesh>, PreparedQuadtree>);

pub(super) fn prepare_quadtrees(
    frame: Res<WaterQuadtreeFrame>,
    buffers: Res<RenderAssets<GpuShaderBuffer>>,
    pipeline: Res<QuadtreePipeline>,
    cache: Res<PipelineCache>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    mut state: ResMut<GpuQuadtrees>,
) {
    state
        .0
        .retain(|mesh, _| frame.0.iter().any(|entry| entry.mesh == *mesh));
    for entry in &frame.0 {
        let Some(tiles) = buffers.get(&entry.tiles) else {
            state.0.remove(&entry.mesh);
            continue;
        };
        if let Some(tree) = state.0.get_mut(&entry.mesh)
            && tree.tiles.id() == tiles.buffer.id()
        {
            tree.params.set(entry.params.clone());
            tree.params.write_buffer(&device, &queue);
            continue;
        }
        let mut params = UniformBuffer::from(entry.params.clone());
        params.write_buffer(&device, &queue);
        let group = device.create_bind_group(
            "water_quadtree_selection",
            &cache.get_bind_group_layout(&pipeline.selection_layout),
            &BindGroupEntries::sequential((
                params.binding().unwrap(),
                tiles.buffer.as_entire_binding(),
            )),
        );
        let draws = state
            .0
            .remove(&entry.mesh)
            .map_or_else(HashMap::new, |old| old.draws);
        state.0.insert(
            entry.mesh,
            PreparedQuadtree {
                params,
                tiles: tiles.buffer.clone(),
                group,
                draws,
            },
        );
    }
}

pub(super) fn select_quadtrees(
    mut ctx: RenderContext,
    pipeline: Res<QuadtreePipeline>,
    cache: Res<PipelineCache>,
    state: Res<GpuQuadtrees>,
) {
    if state.0.is_empty() {
        return;
    }
    let Some(pipelines) = pipeline
        .selection
        .iter()
        .map(|id| cache.get_compute_pipeline(*id))
        .collect::<Option<Vec<_>>>()
    else {
        return;
    };
    let diagnostics = ctx.diagnostic_recorder();
    let diagnostics = diagnostics.as_deref();
    let span = diagnostics.time_span(ctx.command_encoder(), "water_quadtree_selection");
    for tree in state.0.values() {
        let mut pass = ctx
            .command_encoder()
            .begin_compute_pass(&ComputePassDescriptor {
                label: Some("water_quadtree_selection"),
                timestamp_writes: None,
            });
        pass.set_bind_group(0, &tree.group, &[]);
        for (pipeline, groups) in pipelines.iter().zip([1, 9, 1]) {
            pass.set_pipeline(pipeline);
            pass.dispatch_workgroups(groups, 1, 1);
        }
    }
    span.end(ctx.command_encoder());
}

// Bevy rebuilds its arguments separately for early depth, late depth, and the main pass.
// Each water draw reads a private copy with the selected index count and Bevy's instance offset.
pub(super) fn prepare_indirect_draws<const STAGE: u8>(
    view: ViewQuery<Has<bevy::render::occlusion_culling::OcclusionCulling>>,
    mut ctx: RenderContext,
    pipeline: Res<QuadtreePipeline>,
    cache: Res<PipelineCache>,
    device: Res<RenderDevice>,
    indirect: Option<Res<IndirectParametersBuffers>>,
    mut state: ResMut<GpuQuadtrees>,
) {
    if state.0.is_empty() {
        return;
    }
    if STAGE == 1 && !view.into_inner() {
        return;
    }
    let (Some(compute), Some(indirect)) = (cache.get_compute_pipeline(pipeline.draws), indirect)
    else {
        return;
    };
    for tree in state.0.values_mut() {
        for (phase, buffers) in &indirect.buffers {
            let prepass = [
                TypeId::of::<Opaque3dPrepass>(),
                TypeId::of::<AlphaMask3dPrepass>(),
                TypeId::of::<Opaque3dDeferred>(),
                TypeId::of::<AlphaMask3dDeferred>(),
                TypeId::of::<Shadow>(),
            ]
            .contains(phase);
            if prepass != (STAGE != 2) {
                continue;
            }
            let Some(source) = buffers.indexed.data_buffer() else {
                continue;
            };
            if tree
                .draws
                .get(phase)
                .is_none_or(|draw| draw.source.id() != source.id())
            {
                let output = indirect_buffer(&device, source.size());
                let group = device.create_bind_group(
                    "water_quadtree_indirect",
                    &cache.get_bind_group_layout(&pipeline.draw_layout),
                    &BindGroupEntries::sequential((
                        source.as_entire_binding(),
                        output.as_entire_binding(),
                    )),
                );
                tree.draws.insert(
                    *phase,
                    PreparedDraw {
                        source: source.clone(),
                        output,
                        group,
                    },
                );
            }
            let draw = &tree.draws[phase];
            let mut pass = ctx
                .command_encoder()
                .begin_compute_pass(&ComputePassDescriptor {
                    label: Some("water_quadtree_indirect"),
                    timestamp_writes: None,
                });
            pass.set_pipeline(compute);
            pass.set_bind_group(0, &tree.group, &[]);
            pass.set_bind_group(1, &draw.group, &[]);
            pass.dispatch_workgroups((source.size() / 20).div_ceil(64) as u32, 1, 1);
        }
    }
}

fn indirect_buffer(device: &RenderDevice, size: u64) -> Buffer {
    device.create_buffer(&BufferDescriptor {
        label: Some("water_quadtree_draw_args"),
        size,
        usage: BufferUsages::STORAGE | BufferUsages::INDIRECT,
        mapped_at_creation: false,
    })
}

fn preprocessing_ready(p: &PreprocessPipelines, cache: &PipelineCache) -> bool {
    [
        p.direct_preprocess.pipeline_id,
        p.gpu_frustum_culling_preprocess.pipeline_id,
        p.early_gpu_occlusion_culling_preprocess.pipeline_id,
        p.late_gpu_occlusion_culling_preprocess.pipeline_id,
        p.gpu_frustum_culling_build_indexed_indirect_params
            .pipeline_id,
        p.early_phase.reset_indirect_batch_sets.pipeline_id,
        p.early_phase
            .gpu_occlusion_culling_build_indexed_indirect_params
            .pipeline_id,
        p.late_phase.reset_indirect_batch_sets.pipeline_id,
        p.late_phase
            .gpu_occlusion_culling_build_indexed_indirect_params
            .pipeline_id,
        p.main_phase.reset_indirect_batch_sets.pipeline_id,
        p.main_phase
            .gpu_occlusion_culling_build_indexed_indirect_params
            .pipeline_id,
    ]
    .into_iter()
    .all(|id| id.is_some_and(|id| cache.get_compute_pipeline(id).is_some()))
}

struct DrawWaterMesh;
impl<P: PhaseItem + 'static> RenderCommand<P> for DrawWaterMesh {
    type Param = (<DrawMesh as RenderCommand<P>>::Param, SRes<GpuQuadtrees>);
    type ViewQuery = <DrawMesh as RenderCommand<P>>::ViewQuery;
    type ItemQuery = ();
    fn render<'w>(
        item: &P,
        view: ROQueryItem<Self::ViewQuery>,
        _: Option<()>,
        (params, trees): SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let trees = trees.into_inner();
        let mesh_id = params.1.mesh_asset_id(item.main_entity());
        let Some(tree) = mesh_id.and_then(|id| trees.0.get(&id)) else {
            return DrawMesh::render(item, view, Some(()), params, pass);
        };
        let PhaseItemExtraIndex::IndirectParametersIndex { range, .. } = item.extra_index() else {
            return DrawMesh::render(item, view, Some(()), params, pass);
        };
        if let Some(pipelines) = &params.5
            && (!view || !preprocessing_ready(pipelines, &params.3))
        {
            return RenderCommandResult::Skip;
        }
        let Some(args) = tree.draws.get(&TypeId::of::<P>()) else {
            return RenderCommandResult::Skip;
        };
        let mesh_id = mesh_id.unwrap();
        let meshes = params.0.into_inner();
        let allocator = params.4.into_inner();
        let (Some(mesh), Some(vertices), Some(indices)) = (
            meshes.get(mesh_id),
            allocator.mesh_vertex_slice(&mesh_id),
            allocator.mesh_index_slice(&mesh_id),
        ) else {
            return RenderCommandResult::Skip;
        };
        let bevy::render::mesh::RenderMeshBufferInfo::Indexed { index_format, .. } =
            mesh.buffer_info
        else {
            return RenderCommandResult::Skip;
        };
        pass.set_vertex_buffer(0, vertices.buffer.slice(..));
        pass.set_index_buffer(indices.buffer.slice(..), index_format);
        pass.draw_indexed_indirect(&args.output, range.start as u64 * 20);
        RenderCommandResult::Success
    }
}

type DrawWater = (
    SetItemPipeline,
    SetMeshViewBindGroup<0>,
    SetMeshViewBindingArrayBindGroup<1>,
    SetMeshBindGroup<2>,
    SetMaterialBindGroup<3>,
    DrawWaterMesh,
);
type DrawWaterPrepass = (
    SetItemPipeline,
    SetPrepassViewBindGroup<0>,
    SetPrepassViewEmptyBindGroup<1>,
    SetMeshBindGroup<2>,
    SetMaterialBindGroup<3>,
    DrawWaterMesh,
);
type DrawWaterDepth = (
    SetItemPipeline,
    SetPrepassViewBindGroup<0>,
    SetPrepassViewEmptyBindGroup<1>,
    SetMeshBindGroup<2>,
    SetPrepassEmptyMaterialBindGroup<3>,
    DrawWaterMesh,
);

fn install<P: PhaseItem + 'static, Original: 'static, C: RenderCommand<P> + Send + Sync + 'static>(
    world: &mut World,
) where
    C::Param: bevy::ecs::system::ReadOnlySystemParam,
{
    let draw = RenderCommandState::<P, C>::new(world);
    world
        .resource::<DrawFunctions<P>>()
        .write()
        .add_with::<Original, _>(draw);
}

pub(super) fn install_draws(app: &mut App) {
    let Some(render) = app.get_sub_app_mut(RenderApp) else {
        return;
    };
    let world = render.world_mut();
    install::<Opaque3d, DrawMaterial, DrawWater>(world);
    install::<AlphaMask3d, DrawMaterial, DrawWater>(world);
    install::<Transmissive3d, DrawMaterial, DrawWater>(world);
    install::<Transparent3d, DrawMaterial, DrawWater>(world);
    install::<Opaque3dPrepass, DrawPrepass, DrawWaterPrepass>(world);
    install::<AlphaMask3dPrepass, DrawPrepass, DrawWaterPrepass>(world);
    install::<Opaque3dDeferred, DrawPrepass, DrawWaterPrepass>(world);
    install::<AlphaMask3dDeferred, DrawPrepass, DrawWaterPrepass>(world);
    install::<Shadow, DrawPrepass, DrawWaterPrepass>(world);
    install::<Opaque3dPrepass, DrawDepthOnlyPrepass, DrawWaterDepth>(world);
    install::<AlphaMask3dPrepass, DrawDepthOnlyPrepass, DrawWaterDepth>(world);
    install::<Shadow, DrawDepthOnlyPrepass, DrawWaterDepth>(world);
}

pub(super) fn configure_draws(render: &mut bevy::app::SubApp) {
    render.add_systems(
        Core3d,
        (
            prepare_indirect_draws::<0>
                .after(early_prepass_build_indirect_parameters)
                .before(bevy::core_pipeline::prepass::node::early_prepass)
                .before(per_view_shadow_pass::<EARLY_SHADOW_PASS>)
                .before(shared_shadow_pass::<EARLY_SHADOW_PASS>),
            prepare_indirect_draws::<1>
                .after(late_prepass_build_indirect_parameters)
                .before(bevy::core_pipeline::prepass::node::late_prepass)
                .before(bevy::core_pipeline::deferred::node::late_deferred_prepass)
                .before(per_view_shadow_pass::<LATE_SHADOW_PASS>)
                .before(shared_shadow_pass::<LATE_SHADOW_PASS>),
            prepare_indirect_draws::<2>
                .after(main_build_indirect_parameters)
                .before(bevy::core_pipeline::schedule::Core3dSystems::MainPass),
        ),
    );
}
