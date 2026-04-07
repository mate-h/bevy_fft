use bevy::{
    core_pipeline::core_2d::graph::Core2d,
    ecs::{
        query::QueryState,
        world::{FromWorld, World},
    },
    log::{error, info},
    render::{
        render_graph::{Node, NodeRunError, RenderGraph, RenderGraphContext, RenderLabel},
        render_resource::{ComputePassDescriptor, PipelineCache},
        renderer::RenderContext,
    },
    utils::once,
};

use super::{
    FftSchedule, FftSettings,
    resources::{FftBindGroups, FftPipelines, FftResolveBindGroups},
};

use bytemuck;

#[derive(PartialEq, Eq, Debug, Copy, Clone, Hash, RenderLabel)]
pub enum FftNode {
    ComputeFFT,
    /// After the forward FFT the spectrum lives in **C**. The stock implementation for this label
    /// does nothing on the GPU. Use [`splice_spectrum_pass`] to insert a real compute pass here.
    SpectrumPass,
    ComputeIFFT,
    /// Finishes the frame by writing user-facing views into `spatial_output` and `power_spectrum`.
    ResolveOutputs,
    /// Optional hook. Register a compute node with this label to run pattern generation before [`Self::ComputeFFT`].
    GeneratePattern,
}

/// Empty placeholder for [`FftNode::SpectrumPass`], replaced when splicing a custom node.
#[derive(Default)]
pub struct FftSpectrumPassthroughNode;

impl Node for FftSpectrumPassthroughNode {
    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        _render_context: &mut RenderContext,
        _world: &World,
    ) -> Result<(), NodeRunError> {
        Ok(())
    }
}

/// Drops the default spectrum pass and wires `user_pass` between [`FftNode::ComputeFFT`] and [`FftNode::ComputeIFFT`].
///
/// Register `user_pass` on the **Core2d** render graph before calling this function.
pub fn splice_spectrum_pass(world: &mut World, user_pass: impl RenderLabel) {
    let Some(mut render_graph) = world.get_resource_mut::<RenderGraph>() else {
        return;
    };
    let Some(graph) = render_graph.get_sub_graph_mut(Core2d) else {
        return;
    };
    if graph.get_node_state(FftNode::SpectrumPass).is_err() {
        return;
    }
    let _ = graph.remove_node(FftNode::SpectrumPass);
    let user = user_pass.intern();
    graph.add_node_edge(FftNode::ComputeFFT, user);
    graph.add_node_edge(user, FftNode::ComputeIFFT);
}

pub(super) struct FftComputeNode {
    query: QueryState<(&'static FftBindGroups, &'static FftSettings)>,
}

impl FromWorld for FftComputeNode {
    fn from_world(world: &mut World) -> Self {
        Self {
            query: world.query(),
        }
    }
}

impl Node for FftComputeNode {
    fn update(&mut self, world: &mut World) {
        self.query.update_archetypes(world);
    }

    fn run(
        &self,
        graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), NodeRunError> {
        let pipelines = world.resource::<FftPipelines>();
        let pipeline_cache = world.resource::<PipelineCache>();
        let node_label = graph.label();

        for (bind_groups, settings) in self.query.iter_manual(world) {
            let schedule = FftSchedule::try_from_bits(settings.schedule).unwrap_or(FftSchedule::Forward);

            if node_label == FftNode::ComputeFFT.intern() && matches!(schedule, FftSchedule::Inverse) {
                once!(info!(
                    "Skipping forward FFT because schedule is FftSchedule::Inverse"
                ));
                continue;
            }

            if node_label == FftNode::ComputeIFFT.intern()
                && matches!(schedule, FftSchedule::Forward)
            {
                continue;
            }

            let (horizontal_pipeline_id, vertical_pipeline_id, prefix) =
                if node_label == FftNode::ComputeFFT.intern() {
                    (pipelines.fft_horizontal, pipelines.fft_vertical, "fft")
                } else if node_label == FftNode::ComputeIFFT.intern() {
                    (pipelines.ifft_horizontal, pipelines.ifft_vertical, "ifft")
                } else {
                    once!(error!(
                        "FftComputeNode used with invalid label: {:?}",
                        node_label
                    ));
                    return Ok(());
                };

            let Some(horizontal_pipeline) =
                pipeline_cache.get_compute_pipeline(horizontal_pipeline_id)
            else {
                once!(error!("Failed to get {}_horizontal pipeline", prefix));
                return Ok(());
            };

            let Some(vertical_pipeline) = pipeline_cache.get_compute_pipeline(vertical_pipeline_id)
            else {
                once!(error!("Failed to get {}_vertical pipeline", prefix));
                return Ok(());
            };

            let command_encoder = render_context.command_encoder();

            let workgroup_size = 256;
            let num_workgroups_x = settings.size.x.div_ceil(workgroup_size);
            let num_workgroups_y = settings.size.y;

            {
                let mut compute_pass = command_encoder.begin_compute_pass(&ComputePassDescriptor {
                    label: Some(&*format!("{}_horizontal_pass", prefix)),
                    timestamp_writes: None,
                });

                compute_pass.set_pipeline(horizontal_pipeline);
                compute_pass.set_bind_group(0, &bind_groups.common, &[]);

                compute_pass.set_push_constants(0, bytemuck::cast_slice(&[0u32]));
                compute_pass.dispatch_workgroups(num_workgroups_x, num_workgroups_y, 1);
            }

            {
                let vertical_start = settings.orders - 8;

                let mut compute_pass = command_encoder.begin_compute_pass(&ComputePassDescriptor {
                    label: Some(&*format!("{}_vertical_pass", prefix)),
                    timestamp_writes: None,
                });

                compute_pass.set_pipeline(vertical_pipeline);
                compute_pass.set_bind_group(0, &bind_groups.common, &[]);

                compute_pass.set_push_constants(0, bytemuck::cast_slice(&[vertical_start]));
                compute_pass.dispatch_workgroups(num_workgroups_x, num_workgroups_y, 1);
            }
        }

        Ok(())
    }
}

pub(super) struct FftResolveOutputsNode {
    query: QueryState<(&'static FftResolveBindGroups, &'static FftSettings)>,
}

impl FromWorld for FftResolveOutputsNode {
    fn from_world(world: &mut World) -> Self {
        Self {
            query: world.query(),
        }
    }
}

impl Node for FftResolveOutputsNode {
    fn update(&mut self, world: &mut World) {
        self.query.update_archetypes(world);
    }

    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), NodeRunError> {
        let pipelines = world.resource::<FftPipelines>();
        let pipeline_cache = world.resource::<PipelineCache>();

        let Some(pipeline) = pipeline_cache.get_compute_pipeline(pipelines.resolve_outputs) else {
            return Ok(());
        };

        let command_encoder = render_context.command_encoder();
        let mut compute_pass = command_encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("fft_resolve_outputs_pass"),
            timestamp_writes: None,
        });

        compute_pass.set_pipeline(pipeline);

        let wg = 16u32;
        for (bind, settings) in self.query.iter_manual(world) {
            compute_pass.set_bind_group(0, &bind.group, &[]);
            let nx = settings.size.x.div_ceil(wg);
            let ny = settings.size.y.div_ceil(wg);
            compute_pass.dispatch_workgroups(nx, ny, 1);
        }

        Ok(())
    }
}
