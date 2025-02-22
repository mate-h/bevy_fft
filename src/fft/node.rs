use bevy::log;
use bevy_ecs::{
    query::{QueryItem, QueryState},
    system::lifetimeless::Read,
    world::{FromWorld, World},
};
use bevy_render::{
    extract_component::DynamicUniformIndex,
    render_graph::{
        Node, NodeRunError, RenderGraphContext, RenderLabel, SlotInfo, SlotLabel, SlotType,
        ViewNode,
    },
    render_resource::{ComputePassDescriptor, PipelineCache},
    renderer::RenderContext,
};

use super::{
    resources::{FftBindGroups, FftPipelines},
    FftSettings, FftTextures,
};

#[derive(PartialEq, Eq, Debug, Copy, Clone, Hash, RenderLabel)]
pub enum FftNode {
    ComputeFFT,
    // ComputeIFFT,
}

pub(super) struct FftComputeNode {
    query: QueryState<(Read<FftBindGroups>, Read<FftSettings>)>,
}

impl FromWorld for FftComputeNode {
    fn from_world(world: &mut World) -> Self {
        Self {
            query: QueryState::new(world),
        }
    }
}

impl Node for FftComputeNode {
    fn run(
        &self,
        graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), NodeRunError> {
        log::info!("Running FFT");
        let pipelines = world.resource::<FftPipelines>();
        let pipeline_cache = world.resource::<PipelineCache>();
        let Some(fft_pipeline) = pipeline_cache.get_compute_pipeline(pipelines.fft) else {
            log::error!("Failed to get FFT pipeline");
            return Ok(());
        };
        for (bind_groups, settings) in self.query.iter_manual(world) {
            log::info!("Processing FFT for {:?}", settings.size);
            let command_encoder = render_context.command_encoder();

            let mut compute_pass = command_encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("fft_compute_pass"),
                timestamp_writes: None,
            });

            // Set up FFT compute pass
            compute_pass.set_pipeline(fft_pipeline);
            compute_pass.set_bind_group(0, &bind_groups.compute, &[]);

            // Dispatch workgroups based on texture size
            let workgroup_size = 256; // Must match shader workgroup size
            let num_workgroups_x = settings.size.x.div_ceil(workgroup_size);
            let num_workgroups_y = settings.size.y;

            compute_pass.dispatch_workgroups(num_workgroups_x, num_workgroups_y, 1);
        }

        Ok(())
    }
}
