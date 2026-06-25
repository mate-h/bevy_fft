use bevy::{
    app::SubApp,
    ecs::{
        schedule::{IntoScheduleConfigs, SystemSet},
        system::Query,
    },
    log::{error, info},
    render::{
        render_resource::{ComputePass, ComputePassDescriptor, PipelineCache},
        renderer::{RenderContext, RenderGraph, RenderGraphSystems},
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

/// Labels for the stock FFT compute chain on the root [`RenderGraph`] schedule.
///
/// Custom passes run between [`run_fft_forward`] and [`run_fft_resolve_spectrum`] on the root
/// [`RenderGraph`] schedule. Call [`disable_spectrum_passthrough`] when replacing the stock pass.
#[derive(PartialEq, Eq, Debug, Copy, Clone, Hash, SystemSet)]
pub enum FftNode {
    ComputeFFT,
    /// After the forward FFT the spectrum lives in **C**. The stock implementation for this label
    /// does nothing on the GPU. Call [`disable_spectrum_passthrough`] and register your pass between
    /// [`run_fft_forward`] and [`run_fft_resolve_spectrum`].
    SpectrumPass,
    /// Writes `power_spectrum` from **C** while it still holds the spectrum (before inverse FFT scratch).
    ResolveSpectrum,
    ComputeIFFT,
    /// Writes `spatial_output` from **B** after the inverse FFT.
    ResolveOutputs,
    /// Optional hook. Register a compute system before [`Self::ComputeFFT`] to run pattern generation.
    GeneratePattern,
}

/// When `true`, the default no-op [`fft_spectrum_passthrough`] system is skipped.
#[derive(bevy::prelude::Resource, Default)]
pub struct FftSpectrumSpliced(pub bool);

fn fft_set_immediates(pass: &mut ComputePass<'_>, pc: &FftPushConstants) {
    pass.set_immediates(0, bytemuck::bytes_of(pc));
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
        fft_set_immediates(pass, &pc);
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
    fft_set_immediates(pass, &pc);
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

/// Registers the stock FFT compute chain on the root [`RenderGraph`] schedule, before the camera driver.
pub fn plug_fft_render_graph(render_app: &mut SubApp) {
    use bevy::core_pipeline::schedule::camera_driver;

    render_app
        .init_resource::<FftSpectrumSpliced>()
        .add_systems(
            RenderGraph,
            ((
                run_fft_forward.in_set(FftNode::ComputeFFT),
                fft_spectrum_passthrough
                    .run_if(|s: bevy::prelude::Res<FftSpectrumSpliced>| !s.0)
                    .in_set(FftNode::SpectrumPass),
                run_fft_resolve_spectrum.in_set(FftNode::ResolveSpectrum),
                run_fft_inverse.in_set(FftNode::ComputeIFFT),
                run_fft_resolve_outputs.in_set(FftNode::ResolveOutputs),
            )
                .chain(),)
                .before(camera_driver)
                .in_set(RenderGraphSystems::Render),
        );
}

/// Disables the stock no-op spectrum pass. Register your pass on [`RenderGraph`] between
/// [`run_fft_forward`] and [`run_fft_resolve_spectrum`].
pub fn disable_spectrum_passthrough(render_app: &mut SubApp) {
    render_app.insert_resource(FftSpectrumSpliced(true));
}

pub fn run_fft_forward(
    mut ctx: RenderContext,
    pipelines: bevy::prelude::Res<FftPipelines>,
    pipeline_cache: bevy::prelude::Res<PipelineCache>,
    query: Query<(&FftBindGroups, &FftSettings)>,
) {
    let command_encoder = ctx.command_encoder();
    let mut compute_pass = command_encoder.begin_compute_pass(&ComputePassDescriptor {
        label: Some("fft_forward".into()),
        timestamp_writes: None,
    });

    for (bind_groups, settings) in &query {
        let schedule =
            FftSchedule::try_from_bits(settings.schedule).unwrap_or(FftSchedule::Forward);
        if matches!(schedule, FftSchedule::Inverse) {
            once!(info!(
                "Skipping forward FFT because schedule is FftSchedule::Inverse"
            ));
            continue;
        }
        run_forward_fft(
            &pipelines,
            &pipeline_cache,
            &mut compute_pass,
            &bind_groups.common,
            settings,
        );
    }
}

pub fn run_fft_inverse(
    mut ctx: RenderContext,
    pipelines: bevy::prelude::Res<FftPipelines>,
    pipeline_cache: bevy::prelude::Res<PipelineCache>,
    query: Query<(&FftBindGroups, &FftSettings)>,
) {
    let command_encoder = ctx.command_encoder();
    let mut compute_pass = command_encoder.begin_compute_pass(&ComputePassDescriptor {
        label: Some("fft_inverse".into()),
        timestamp_writes: None,
    });

    for (bind_groups, settings) in &query {
        let schedule =
            FftSchedule::try_from_bits(settings.schedule).unwrap_or(FftSchedule::Forward);
        if matches!(schedule, FftSchedule::Forward) {
            continue;
        }
        run_inverse_fft(
            &pipelines,
            &pipeline_cache,
            &mut compute_pass,
            &bind_groups.common,
            settings,
        );
    }
}

fn fft_spectrum_passthrough() {}

pub fn run_fft_resolve_spectrum(
    mut ctx: RenderContext,
    pipelines: bevy::prelude::Res<FftPipelines>,
    pipeline_cache: bevy::prelude::Res<PipelineCache>,
    query: Query<(&FftResolveBindGroups, &FftSettings)>,
) {
    let Some(pipeline) = pipeline_cache.get_compute_pipeline(pipelines.resolve_spectrum) else {
        return;
    };

    let command_encoder = ctx.command_encoder();
    let mut compute_pass = command_encoder.begin_compute_pass(&ComputePassDescriptor {
        label: Some("fft_resolve_spectrum_pass".into()),
        timestamp_writes: None,
    });

    compute_pass.set_pipeline(pipeline);

    let wg = 16u32;
    for (bind, settings) in &query {
        compute_pass.set_bind_group(0, &bind.group, &[]);
        let nx = settings.size.x.div_ceil(wg);
        let ny = settings.size.y.div_ceil(wg);
        compute_pass.dispatch_workgroups(nx, ny, 1);
    }
}

pub fn run_fft_resolve_outputs(
    mut ctx: RenderContext,
    pipelines: bevy::prelude::Res<FftPipelines>,
    pipeline_cache: bevy::prelude::Res<PipelineCache>,
    query: Query<(&FftResolveBindGroups, &FftSettings)>,
) {
    let Some(pipeline) = pipeline_cache.get_compute_pipeline(pipelines.resolve_spatial) else {
        return;
    };

    let command_encoder = ctx.command_encoder();
    let mut compute_pass = command_encoder.begin_compute_pass(&ComputePassDescriptor {
        label: Some("fft_resolve_spatial_pass".into()),
        timestamp_writes: None,
    });

    compute_pass.set_pipeline(pipeline);

    let wg = 16u32;
    for (bind, settings) in &query {
        compute_pass.set_bind_group(0, &bind.group, &[]);
        let nx = settings.size.x.div_ceil(wg);
        let ny = settings.size.y.div_ceil(wg);
        compute_pass.dispatch_workgroups(nx, ny, 1);
    }
}
