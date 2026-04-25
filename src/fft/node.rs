use bevy::{
    ecs::{
        query::QueryState,
        world::{FromWorld, World},
    },
    log::{error, info},
    render::{
        graph::CameraDriverLabel,
        render_graph::{Node, NodeRunError, RenderGraph, RenderGraphContext, RenderLabel},
        render_resource::{ComputePass, ComputePassDescriptor, PipelineCache},
        renderer::RenderContext,
    },
    utils::once,
};

use super::{
    FftSchedule, FftSettings,
    resources::{FftBindGroups, FftPipelines, FftResolveBindGroups},
};

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct FftPushConstants {
    stage: u32,
    axis: u32,
    src_buffer: u32,
    dst_buffer: u32,
    flags: u32,
}

const BUF_A: u32 = 0;
const BUF_B: u32 = 1;
const BUF_C: u32 = 2;

const FLAG_INVERSE_FINALIZE: u32 = 1;
const FLAG_FORWARD_ALPHA: u32 = 2;

#[derive(PartialEq, Eq, Debug, Copy, Clone, Hash, RenderLabel)]
pub enum FftNode {
    ComputeFFT,
    /// After the forward FFT the spectrum lives in **C**. The stock implementation for this label
    /// does nothing on the GPU. Use [`splice_spectrum_pass`] to insert a real compute pass here.
    SpectrumPass,
    /// Writes `power_spectrum` from **C** while it still holds the spectrum (before inverse FFT scratch).
    ResolveSpectrum,
    ComputeIFFT,
    /// Writes `spatial_output` from **B** after the inverse FFT.
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

/// Drops the default spectrum pass and wires `user_pass` between [`FftNode::ComputeFFT`] and [`FftNode::ResolveSpectrum`].
///
/// Call from `RenderApp` after registering `user_pass` on the **root** [`RenderGraph`].
pub fn splice_spectrum_pass(world: &mut World, user_pass: impl RenderLabel) {
    let Some(mut graph) = world.get_resource_mut::<RenderGraph>() else {
        return;
    };
    if graph.get_node_state(FftNode::SpectrumPass).is_err() {
        bevy::log::warn!(
            "splice_spectrum_pass: `FftNode::SpectrumPass` is missing. Register `FftPlugin` before splicing so the forward chain is set up."
        );
        return;
    }
    let _ = graph.remove_node(FftNode::SpectrumPass);
    let user = user_pass.intern();
    graph.add_node_edge(FftNode::ComputeFFT, user);
    graph.add_node_edge(user, FftNode::ResolveSpectrum);
}

/// Runs `user_pass` after [`FftNode::ResolveOutputs`] and before [`CameraDriverLabel`].
///
/// Call from `RenderApp` after [`FftPlugin`] registers the `ResolveOutputs` → `CameraDriver` edge,
/// and after registering `user_pass` on the root [`RenderGraph`].
pub fn splice_after_resolve_outputs(world: &mut World, user_pass: impl RenderLabel) {
    let Some(mut graph) = world.get_resource_mut::<RenderGraph>() else {
        return;
    };
    if graph
        .remove_node_edge(FftNode::ResolveOutputs, CameraDriverLabel)
        .is_err()
    {
        bevy::log::warn!(
            "splice_after_resolve_outputs: could not remove ResolveOutputs → CameraDriver edge. Register FftPlugin before OceanPlugin."
        );
        return;
    }
    let user = user_pass.intern();
    graph.add_node_edge(FftNode::ResolveOutputs, user);
    graph.add_node_edge(user, CameraDriverLabel);
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

fn fft_set_push_constants(pass: &mut ComputePass<'_>, pc: &FftPushConstants) {
    pass.set_push_constants(0, bytemuck::bytes_of(pc));
}

fn fft_dispatch_dit_chain(
    pass: &mut ComputePass<'_>,
    pipeline: &bevy::render::render_resource::ComputePipeline,
    bind: &bevy::render::render_resource::BindGroup,
    orders: u32,
    axis: u32,
    mut src: u32,
    mut dst: u32,
    n: u32,
    forward_alpha: bool,
    inverse_finalize_on_last: bool,
) {
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bind, &[]);
    let half_n = n / 2;
    let gx = half_n.div_ceil(256);
    for stage in 0..orders {
        let mut flags = 0u32;
        if forward_alpha {
            flags |= FLAG_FORWARD_ALPHA;
        }
        if inverse_finalize_on_last && stage + 1 == orders {
            flags |= FLAG_INVERSE_FINALIZE;
        }
        let pc = FftPushConstants {
            stage,
            axis,
            src_buffer: src,
            dst_buffer: dst,
            flags,
        };
        fft_set_push_constants(pass, &pc);
        pass.dispatch_workgroups(gx, n, 1);
        std::mem::swap(&mut src, &mut dst);
    }
}

fn fft_dispatch_copy(
    pass: &mut ComputePass<'_>,
    pipeline: &bevy::render::render_resource::ComputePipeline,
    bind: &bevy::render::render_resource::BindGroup,
    src: u32,
    dst: u32,
    n: u32,
) {
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bind, &[]);
    let pc = FftPushConstants {
        stage: 0,
        axis: 0,
        src_buffer: src,
        dst_buffer: dst,
        flags: 0,
    };
    fft_set_push_constants(pass, &pc);
    let gx = n.div_ceil(16);
    let gy = n.div_ceil(16);
    pass.dispatch_workgroups(gx, gy, 1);
}

/// Forward 2D FFT: data must be in buffer **A**; spectrum ends in **C** (for real-to-complex style packing, put signal in A_re channel 0, A_im 0).
pub fn run_forward_fft(
    pipelines: &FftPipelines,
    pipeline_cache: &PipelineCache,
    pass: &mut ComputePass<'_>,
    bind: &bevy::render::render_resource::BindGroup,
    settings: &FftSettings,
) {
    let n = settings.size.x;
    let orders = settings.orders;

    let Some(br_h) = pipeline_cache.get_compute_pipeline(pipelines.forward_br_horizontal) else {
        once!(error!("Missing forward_br_horizontal pipeline"));
        return;
    };
    let Some(br_v) = pipeline_cache.get_compute_pipeline(pipelines.forward_br_vertical) else {
        once!(error!("Missing forward_br_vertical pipeline"));
        return;
    };
    let Some(dit) = pipeline_cache.get_compute_pipeline(pipelines.radix2_dit) else {
        once!(error!("Missing radix2_dit pipeline"));
        return;
    };
    let Some(cpy) = pipeline_cache.get_compute_pipeline(pipelines.fft_copy) else {
        once!(error!("Missing fft_copy pipeline"));
        return;
    };

    {
        pass.set_pipeline(br_h);
        pass.set_bind_group(0, bind, &[]);
        let gx = n.div_ceil(8);
        let gy = n.div_ceil(8);
        pass.dispatch_workgroups(gx, gy, 1);
    }

    fft_dispatch_dit_chain(pass, dit, bind, orders, 0, BUF_B, BUF_A, n, true, false);

    if orders % 2 == 1 {
        fft_dispatch_copy(pass, cpy, bind, BUF_A, BUF_B, n);
    }

    {
        pass.set_pipeline(br_v);
        pass.set_bind_group(0, bind, &[]);
        let gx = n.div_ceil(8);
        let gy = n.div_ceil(8);
        pass.dispatch_workgroups(gx, gy, 1);
    }

    fft_dispatch_dit_chain(pass, dit, bind, orders, 1, BUF_C, BUF_B, n, true, false);

    if orders % 2 == 1 {
        fft_dispatch_copy(pass, cpy, bind, BUF_B, BUF_C, n);
    }
}

/// Inverse 2D FFT: spectrum in **C**; result real parts primarily in **B** after the pass.
pub fn run_inverse_fft(
    pipelines: &FftPipelines,
    pipeline_cache: &PipelineCache,
    pass: &mut ComputePass<'_>,
    bind: &bevy::render::render_resource::BindGroup,
    settings: &FftSettings,
) {
    let n = settings.size.x;
    let orders = settings.orders;

    let Some(br_h) = pipeline_cache.get_compute_pipeline(pipelines.inverse_br_horizontal) else {
        once!(error!("Missing inverse_br_horizontal pipeline"));
        return;
    };
    let Some(br_v) = pipeline_cache.get_compute_pipeline(pipelines.inverse_br_vertical) else {
        once!(error!("Missing inverse_br_vertical pipeline"));
        return;
    };
    let Some(dit) = pipeline_cache.get_compute_pipeline(pipelines.radix2_dit) else {
        once!(error!("Missing radix2_dit pipeline"));
        return;
    };
    let Some(cpy) = pipeline_cache.get_compute_pipeline(pipelines.fft_copy) else {
        once!(error!("Missing fft_copy pipeline"));
        return;
    };

    {
        pass.set_pipeline(br_h);
        pass.set_bind_group(0, bind, &[]);
        let gx = n.div_ceil(8);
        let gy = n.div_ceil(8);
        pass.dispatch_workgroups(gx, gy, 1);
    }

    fft_dispatch_dit_chain(pass, dit, bind, orders, 0, BUF_A, BUF_C, n, false, true);

    if orders % 2 == 1 {
        fft_dispatch_copy(pass, cpy, bind, BUF_C, BUF_A, n);
    }

    {
        pass.set_pipeline(br_v);
        pass.set_bind_group(0, bind, &[]);
        let gx = n.div_ceil(8);
        let gy = n.div_ceil(8);
        pass.dispatch_workgroups(gx, gy, 1);
    }

    fft_dispatch_dit_chain(pass, dit, bind, orders, 1, BUF_B, BUF_A, n, false, true);

    if orders % 2 == 1 {
        fft_dispatch_copy(pass, cpy, bind, BUF_A, BUF_B, n);
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
            let schedule =
                FftSchedule::try_from_bits(settings.schedule).unwrap_or(FftSchedule::Forward);

            if node_label == FftNode::ComputeFFT.intern()
                && matches!(schedule, FftSchedule::Inverse)
            {
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

            let command_encoder = render_context.command_encoder();
            let label = if node_label == FftNode::ComputeFFT.intern() {
                "fft_forward"
            } else {
                "fft_inverse"
            };

            let mut compute_pass = command_encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some(label.into()),
                timestamp_writes: None,
            });

            if node_label == FftNode::ComputeFFT.intern() {
                if !matches!(schedule, FftSchedule::Inverse) {
                    run_forward_fft(
                        pipelines,
                        pipeline_cache,
                        &mut compute_pass,
                        &bind_groups.common,
                        settings,
                    );
                }
            } else if node_label == FftNode::ComputeIFFT.intern() {
                if !matches!(schedule, FftSchedule::Forward) {
                    run_inverse_fft(
                        pipelines,
                        pipeline_cache,
                        &mut compute_pass,
                        &bind_groups.common,
                        settings,
                    );
                }
            } else {
                once!(error!(
                    "FftComputeNode used with invalid label: {:?}",
                    node_label
                ));
                return Ok(());
            }
        }

        Ok(())
    }
}

pub(super) struct FftResolveSpectrumNode {
    query: QueryState<(&'static FftResolveBindGroups, &'static FftSettings)>,
}

impl FromWorld for FftResolveSpectrumNode {
    fn from_world(world: &mut World) -> Self {
        Self {
            query: world.query(),
        }
    }
}

impl Node for FftResolveSpectrumNode {
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

        let Some(pipeline) = pipeline_cache.get_compute_pipeline(pipelines.resolve_spectrum) else {
            return Ok(());
        };

        let command_encoder = render_context.command_encoder();
        let mut compute_pass = command_encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("fft_resolve_spectrum_pass".into()),
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

        let Some(pipeline) = pipeline_cache.get_compute_pipeline(pipelines.resolve_spatial) else {
            return Ok(());
        };

        let command_encoder = render_context.command_encoder();
        let mut compute_pass = command_encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("fft_resolve_spatial_pass".into()),
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
