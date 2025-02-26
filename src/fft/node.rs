use bevy_ecs::{
    query::QueryState,
    world::{FromWorld, World},
};
use bevy_log::{error, info};
use bevy_render::{
    render_graph::{Node, NodeRunError, RenderGraphContext, RenderLabel},
    render_resource::{ComputePassDescriptor, PipelineCache},
    renderer::RenderContext,
};
use bevy_utils::once;

use super::{
    resources::{FftBindGroups, FftPipelines},
    FftSettings,
};

#[derive(PartialEq, Eq, Debug, Copy, Clone, Hash, RenderLabel)]
pub enum FftNode {
    ComputeFFT,
    ComputeIFFT,
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

        once!(info!("Running FFT/IFFT"));
        for (bind_groups, settings) in self.query.iter_manual(world) {
            // Choose pipeline based on inverse flag
            let pipeline_id = if settings.inverse == 1 {
                pipelines.ifft
            } else {
                pipelines.fft
            };

            let label = if settings.inverse == 1 {
                "ifft_compute_pass"
            } else {
                "fft_compute_pass"
            };

            let Some(pipeline) = pipeline_cache.get_compute_pipeline(pipeline_id) else {
                once!(error!("Failed to get {} pipeline", label));
                return Ok(());
            };

            once!(info!("Processing {} for {:?}", label, settings.size));
            let command_encoder = render_context.command_encoder();

            let mut compute_pass = command_encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some(label.into()),
                timestamp_writes: None,
            });

            // Set up compute pass
            compute_pass.set_pipeline(pipeline);
            compute_pass.set_bind_group(0, &bind_groups.compute, &[]);

            // Dispatch workgroups based on texture size
            let workgroup_size = 256; // Must match shader workgroup size
            let num_workgroups_x = settings.size.x.div_ceil(workgroup_size);
            let num_workgroups_y = settings.size.y;

            once!(info!("Dispatching {} for {:?}", label, settings.size));

            compute_pass.dispatch_workgroups(num_workgroups_x, num_workgroups_y, 1);
        }

        Ok(())
    }
}
