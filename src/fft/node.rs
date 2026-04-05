// use bevy_ecs::{
//     label::DynEq,
//     query::QueryState,
//     world::{FromWorld, World},
// };
// use bevy_log::{error, info};
// use bevy_render::{
//     render_graph::{Node, NodeRunError, RenderGraphContext, RenderLabel},
//     render_resource::{ComputePassDescriptor, PipelineCache},
//     renderer::RenderContext,
// };
// use bevy_utils::once;

use bevy::{
    ecs::{
        query::QueryState,
        world::{FromWorld, World},
    },
    log::{error, info},
    render::{
        render_graph::{Node, NodeRunError, RenderGraphContext, RenderLabel},
        render_resource::{ComputePassDescriptor, PipelineCache},
        renderer::RenderContext,
    },
    utils::once,
};

use super::{
    FftSettings,
    resources::{FftBindGroups, FftPipelines, FftResolveBindGroups},
};

use bytemuck;

#[derive(PartialEq, Eq, Debug, Copy, Clone, Hash, RenderLabel)]
pub enum FftNode {
    ComputeFFT,
    ComputeIFFT,
    /// Fills `spatial_output` and `power_spectrum` after the pipeline.
    ResolveOutputs,
    /// Reserved hook for examples (pattern generation).
    GeneratePattern,
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
            let inverse = settings.inverse != 0;
            let roundtrip = settings.roundtrip != 0;

            if node_label == FftNode::ComputeFFT.intern() && inverse && !roundtrip {
                once!(info!(
                    "Skipping forward FFT because inverse mode is enabled"
                ));
                continue;
            }

            if node_label == FftNode::ComputeIFFT.intern() && !inverse && !roundtrip {
                continue;
            }

            // Choose pipelines based on node label
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

            // Get pipelines
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

            // First pass: Horizontal FFT
            {
                let mut compute_pass = command_encoder.begin_compute_pass(&ComputePassDescriptor {
                    label: Some(&*format!("{}_horizontal_pass", prefix)),
                    timestamp_writes: None,
                });

                compute_pass.set_pipeline(horizontal_pipeline);
                compute_pass.set_bind_group(0, &bind_groups.common, &[]);

                // Set initial iteration to 0 using push constants
                compute_pass.set_push_constants(0, bytemuck::cast_slice(&[0u32]));
                compute_pass.dispatch_workgroups(num_workgroups_x, num_workgroups_y, 1);
            } // compute_pass is dropped here, ensuring the first pass completes

            // Second pass: Vertical FFT
            {
                // This calculation needs to consider the full orders
                let vertical_start = settings.orders - 8; // Start from where horizontal left off

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
