//! Radial band-pass on spectrum buffer **C** between forward FFT and IFFT.

use bevy::{
    core_pipeline::core_2d::graph::Core2d,
    ecs::{
        query::{QueryItem, QueryState},
        schedule::IntoScheduleConfigs,
        system::lifetimeless::Read,
    },
    prelude::*,
    render::{
        Render, RenderApp,
        extract_component::{
            ComponentUniforms, ExtractComponent, ExtractComponentPlugin, UniformComponentPlugin,
        },
        render_graph::{Node, NodeRunError, RenderGraphContext, RenderGraphExt, RenderLabel},
        render_resource::{
            binding_types::uniform_buffer,
            BindGroup, BindGroupEntries, BindGroupLayoutDescriptor, BindGroupLayoutEntries,
            CachedComputePipelineId, ComputePassDescriptor, ComputePipelineDescriptor, PipelineCache,
            ShaderStages, ShaderType,
        },
        renderer::{RenderContext, RenderDevice},
    },
    shader::ShaderDefVal,
};
use bevy_fft::fft::{
    prepare_fft_bind_groups,
    resources::{FftBindGroupLayouts, FftBindGroups},
    splice_spectrum_pass,
    FftSettings,
};

/// Parameters for [`radial_band_pass`] in `assets/examples/band_pass.wgsl`.
///
/// Both values are **normalized** folded **|k|** radius from 0 to 1. The shader names this `r_norm`.
/// **`band_center`** sets the annulus center. Use values near 0 for DC and near 1 for Nyquist.
/// **`band_width`** is the full width of the passband. The shader builds an annulus clipped to the
/// 0 to 1 range.
#[derive(Component, Clone, Copy, Reflect, ShaderType)]
#[repr(C)]
pub struct BandPassParams {
    pub band_center: f32,
    pub band_width: f32,
}

impl Default for BandPassParams {
    fn default() -> Self {
        Self {
            band_center: 0.37,
            band_width: 0.5,
        }
    }
}

impl ExtractComponent for BandPassParams {
    type QueryData = Read<BandPassParams>;
    type QueryFilter = ();
    type Out = Self;

    fn extract_component(item: QueryItem<'_, '_, Self::QueryData>) -> Option<Self::Out> {
        Some(*item)
    }
}

#[derive(PartialEq, Eq, Debug, Copy, Clone, Hash, RenderLabel)]
pub struct BandPassLabel;

#[derive(Resource)]
pub struct BandPassPipelineRes {
    pub params_layout: BindGroupLayoutDescriptor,
    pub pipeline: CachedComputePipelineId,
}

impl FromWorld for BandPassPipelineRes {
    fn from_world(world: &mut World) -> Self {
        let pipeline_cache = world.resource::<PipelineCache>();
        let layouts = world.resource::<FftBindGroupLayouts>();
        let asset_server = world.resource::<AssetServer>();
        let shader = asset_server.load("examples/band_pass.wgsl");

        let entries = BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (uniform_buffer::<BandPassParams>(false),),
        );
        let params_layout = BindGroupLayoutDescriptor::new("fft_demo_band_pass_params", &entries);

        let pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("fft_demo_band_pass".into()),
            layout: vec![layouts.common.clone(), params_layout.clone()],
            push_constant_ranges: vec![],
            shader,
            shader_defs: vec![ShaderDefVal::UInt("CHANNELS".into(), 4)],
            entry_point: Some("radial_band_pass".into()),
            zero_initialize_workgroup_memory: false,
        });

        Self {
            params_layout,
            pipeline,
        }
    }
}

#[derive(Component)]
pub struct BandPassBindGroup {
    pub group: BindGroup,
}

pub struct BandPassPlugin;

impl Plugin for BandPassPlugin {
    fn build(&self, app: &mut App) {
        app.register_type::<BandPassParams>()
            .add_plugins(ExtractComponentPlugin::<BandPassParams>::default())
            .add_plugins(UniformComponentPlugin::<BandPassParams>::default());
    }

    fn finish(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .init_resource::<BandPassPipelineRes>()
            .add_render_graph_node::<BandPassNode>(Core2d, BandPassLabel)
            .add_systems(
                Render,
                prepare_band_pass_bind_group
                    .in_set(bevy::render::RenderSystems::PrepareBindGroups)
                    .after(prepare_fft_bind_groups),
            );

        splice_spectrum_pass(render_app.world_mut(), BandPassLabel);
    }
}

type PrepareBandPassBindGroupQuery<'w, 's> = Query<
    'w,
    's,
    (Entity, &'static FftBindGroups),
    (
        With<BandPassParams>,
        Without<BandPassBindGroup>,
        With<FftSettings>,
    ),
>;

fn prepare_band_pass_bind_group(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    pipeline_cache: Res<PipelineCache>,
    bp: Res<BandPassPipelineRes>,
    uniforms: Res<ComponentUniforms<BandPassParams>>,
    query: PrepareBandPassBindGroupQuery,
) {
    let Some(uniform_binding) = uniforms.binding() else {
        return;
    };

    let layout = pipeline_cache.get_bind_group_layout(&bp.params_layout);
    for (entity, _fft_bg) in &query {
        let group = render_device.create_bind_group(
            "fft_demo_band_pass_bind_group",
            &layout,
            &BindGroupEntries::sequential((uniform_binding.clone(),)),
        );
        commands.entity(entity).insert(BandPassBindGroup { group });
    }
}

struct BandPassNode {
    query: QueryState<(
        &'static FftBindGroups,
        &'static BandPassBindGroup,
        &'static FftSettings,
    )>,
}

impl FromWorld for BandPassNode {
    fn from_world(world: &mut World) -> Self {
        Self {
            query: world.query(),
        }
    }
}

impl Node for BandPassNode {
    fn update(&mut self, world: &mut World) {
        self.query.update_archetypes(world);
    }

    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), NodeRunError> {
        let bp = world.resource::<BandPassPipelineRes>();
        let pipeline_cache = world.resource::<PipelineCache>();
        let Some(pipeline) = pipeline_cache.get_compute_pipeline(bp.pipeline) else {
            return Ok(());
        };

        let encoder = render_context.command_encoder();
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("fft_demo_band_pass_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);

        let wg = 16u32;
        for (fft_bg, bp_bg, settings) in self.query.iter_manual(world) {
            pass.set_bind_group(0, &fft_bg.common, &[]);
            pass.set_bind_group(1, &bp_bg.group, &[]);
            let nx = settings.size.x.div_ceil(wg);
            let ny = settings.size.y.div_ceil(wg);
            pass.dispatch_workgroups(nx, ny, 1);
        }

        Ok(())
    }
}
