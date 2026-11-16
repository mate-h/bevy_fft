//! Persistent foam on the [`WaterLodData`] slices.
//!
//! Writers store how strongly each texel is breaking this frame in the LOD foam lane, from 0 to 1,
//! and the foam sim turns it into foam that lasts. Full breaking produces active foam at
//! `1 / formation_time` coverage per second. The sim keeps active foam, which sits over breaking
//! crests, and the residual film it leaves behind in [`WaterLodData::foam_state`]. Each frame both drift with
//! [`WaterLodData::drift`], gather where the drift converges, and fade with the half-lives in
//! [`WaterFoamSettings`]. Fading active foam feeds residual foam, as Stage A whitecaps turn into
//! Stage B patches in Monahan and Woolf's model. Sources and decay use the exact solution over each
//! step, so foam does not depend on the frame rate.
//!
//! The same pass carries the foam texture coordinates in [`WaterLodData::foam_flow`] along the
//! drift, after Neyret (2003), so the rendered foam pattern moves with the foam instead of standing
//! still under it. `foam_flow.wesl` describes the layout.

use std::f32::consts::LN_2;

use bevy::{
    prelude::*,
    render::{
        RenderApp,
        diagnostic::RecordDiagnostics,
        extract_resource::ExtractResource,
        render_asset::RenderAssets,
        render_resource::{binding_types::*, *},
        renderer::{RenderContext, RenderDevice, RenderQueue},
        texture::GpuImage,
    },
};

use super::{
    WATER_FOAM_FLOW_FORMAT, WATER_LOD_DRIFT_FORMAT, WaterLodData, WaterLodGpu, WaterLodSlices,
};

/// Format of [`WaterLodData::foam_state`] and the sim's working arrays. Half floats would lose a
/// little foam to rounding on every store, so the two lanes stay at full precision.
pub const WATER_FOAM_STATE_FORMAT: TextureFormat = WATER_LOD_DRIFT_FORMAT;

/// Embedded path of the foam sim shader.
pub const WATER_FOAM_SHADER: &str = "embedded://bevy_fft/water/foam.wesl";

/// How long foam lasts and how much residual foam each breaker leaves.
///
/// The defaults follow field measurements of whitecaps. Monahan and Woolf (1989) put Stage A
/// whitecaps, the active foam, at under a second and Stage B patches at one to ten seconds.
/// Callaghan et al. (2012) measured an area-weighted e-folding time of Stage B foam between 1.4 s and
/// 4.8 s, about 3.5 s typical, which is a half-life near 2.4 s. Surf and river foam held together
/// by surfactants lasts much longer and suits larger values.
#[derive(Resource, Clone, Copy, Debug, PartialEq, Reflect)]
#[reflect(Resource)]
pub struct WaterFoamSettings {
    /// Half-life of active foam in seconds.
    pub active_half_life: f32,
    /// Half-life of residual foam in seconds.
    pub residual_half_life: f32,
    /// Residual foam produced per unit of active foam that fades. Above one, a patch covers more
    /// area as it spreads and thins, as Koepke (1984) observed. At steady state the residual to
    /// active ratio is `residual_yield * residual_half_life / active_half_life`, which Monahan and
    /// Woolf put near 15 at a wind of 10 m/s.
    pub residual_yield: f32,
    /// Seconds of full breaking that produce full active coverage, ignoring decay. Callaghan et al.
    /// (2012) saw whitecap area peak within about a second of breaking onset, and a single texel
    /// turns white faster than a whole patch grows.
    pub formation_time: f32,
}

impl Default for WaterFoamSettings {
    fn default() -> Self {
        Self {
            active_half_life: 0.5,
            residual_half_life: 2.4,
            residual_yield: 3.0,
            formation_time: 0.25,
        }
    }
}

/// Per-step decay factors for the foam sim. Layout matches `WaterFoamParams` in `foam.wesl`.
#[derive(Clone, Copy, Debug, Default, ShaderType)]
pub struct WaterFoamParams {
    /// Slice placement the previous foam was written on.
    pub previous: WaterLodSlices,
    /// Simulated seconds since the previous frame.
    pub dt: f32,
    /// Share of active foam left after `dt`.
    pub active_decay: f32,
    /// Share of residual foam left after `dt`.
    pub residual_decay: f32,
    /// Residual foam gained per unit of active foam at the start of the step.
    pub residual_gain: f32,
    /// Active foam gained per unit of breaking intensity held over the step.
    pub source_active: f32,
    /// Residual foam gained per unit of breaking intensity held over the step.
    pub source_residual: f32,
    /// Wrapped seconds since startup at the end of the step, the surface material's time.
    pub time: f32,
    /// One on the first step the sim runs, which gives every advected texture coordinate a fresh
    /// offset. The render world sets it.
    pub restart: f32,
}

impl WaterFoamParams {
    /// Exact solution over `dt` seconds of `da/dt = b / t_f - la a` and `dr/dt = k la a - lr r`,
    /// with breaking intensity `b` held for the step, formation time `t_f`, and each rate
    /// `ln 2 / half-life`. An infinite half-life keeps that foam forever. The time is left at zero
    /// for the caller to set.
    pub fn step(settings: &WaterFoamSettings, previous: WaterLodSlices, dt: f32) -> Self {
        let rate = |half_life: f32| {
            if half_life.is_finite() {
                LN_2 / half_life.max(1e-4)
            } else {
                0.0
            }
        };
        let la = rate(settings.active_half_life);
        let lr = rate(settings.residual_half_life);
        let active_decay = (-la * dt).exp();
        let residual_decay = (-lr * dt).exp();
        // (1 - e^(-l dt)) / l, the integral of e^(-l s) over the step, which is dt for l = 0.
        let held = |l: f32, decay: f32| if l * dt > 1e-6 { (1.0 - decay) / l } else { dt };
        // (e^(-la dt) - e^(-lr dt)) / (lr - la), with limit dt e^(-la dt) for equal rates.
        let gap = lr - la;
        let overlap = if gap.abs() > 1e-5 * la.max(lr) {
            (active_decay - residual_decay) / gap
        } else {
            dt * active_decay
        };
        let k = settings.residual_yield;
        let inv_formation = settings.formation_time.max(1e-4).recip();
        Self {
            previous,
            dt,
            active_decay,
            residual_decay,
            residual_gain: k * la * overlap,
            source_active: held(la, active_decay) * inv_formation,
            source_residual: k * (held(lr, residual_decay) - overlap) * inv_formation,
            time: 0.0,
            restart: 0.0,
        }
    }
}

/// What the foam sim needs from the main world this frame.
#[derive(Resource, Clone, Copy, Default, ExtractResource)]
#[extract_app(RenderApp)]
pub struct WaterFoamFrame {
    pub params: WaterFoamParams,
}

pub(super) fn gather_water_foam(
    time: Res<Time>,
    settings: Res<WaterFoamSettings>,
    lod: Option<Res<WaterLodData>>,
    mut frame: ResMut<WaterFoamFrame>,
    mut last: Local<Option<WaterLodSlices>>,
) {
    let Some(lod) = lod else {
        return;
    };
    let previous = last.unwrap_or(lod.slices);
    *last = Some(lod.slices);
    frame.params = WaterFoamParams {
        time: time.elapsed_secs_wrapped(),
        ..WaterFoamParams::step(&settings, previous, time.delta_secs())
    };
}

#[derive(Resource)]
pub struct WaterFoamPipeline {
    layout: BindGroupLayoutDescriptor,
    forward: CachedComputePipelineId,
    correct: CachedComputePipelineId,
    flow: CachedComputePipelineId,
}

impl FromWorld for WaterFoamPipeline {
    fn from_world(world: &mut World) -> Self {
        let pipeline_cache = world.resource::<PipelineCache>();
        let asset_server = world.resource::<AssetServer>();
        let unfiltered = TextureSampleType::Float { filterable: false };
        let entries = BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                uniform_buffer::<WaterFoamParams>(false),
                uniform_buffer::<WaterLodSlices>(false),
                texture_2d_array(unfiltered),
                texture_2d_array(unfiltered),
                texture_2d_array(unfiltered),
                texture_2d_array(unfiltered),
                texture_storage_2d_array(WATER_FOAM_STATE_FORMAT, StorageTextureAccess::WriteOnly),
                texture_2d_array(unfiltered),
                texture_storage_2d_array(WATER_FOAM_FLOW_FORMAT, StorageTextureAccess::WriteOnly),
            ),
        );
        let layout = BindGroupLayoutDescriptor::new("water_foam_layout", &entries);
        let shader = asset_server.load(WATER_FOAM_SHADER);
        let queue = |entry: &'static str| {
            pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some(entry.into()),
                layout: vec![layout.clone()],
                immediate_size: 0,
                shader: shader.clone(),
                shader_defs: vec![],
                entry_point: Some(entry.into()),
                zero_initialize_workgroup_memory: false,
                constants: vec![],
            })
        };
        Self {
            forward: queue("foam_forward"),
            correct: queue("foam_correct"),
            flow: queue("foam_flow"),
            layout,
        }
    }
}

struct FoamArray {
    texture: Texture,
    view: TextureView,
}

/// Render-world working arrays and bind groups. `forward` holds the semi-Lagrangian step and
/// `corrected` the finished foam, which is copied into [`WaterLodData::foam_state`] at the end.
/// `flow` holds the advected texture coordinates, copied into [`WaterLodData::foam_flow`].
#[derive(Resource, Default)]
pub struct WaterFoamSim {
    arrays: Option<[FoamArray; 2]>,
    flow: Option<FoamArray>,
    size: Extent3d,
    flow_size: Extent3d,
    /// Whether the flow pass has run on the current flow array.
    flow_seeded: bool,
    params: UniformBuffer<WaterFoamParams>,
    groups: Option<[BindGroup; 2]>,
    dispatch: UVec3,
}

fn foam_array(
    device: &RenderDevice,
    size: Extent3d,
    format: TextureFormat,
    label: &'static str,
) -> FoamArray {
    let texture = device.create_texture(&TextureDescriptor {
        label: Some(label),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format,
        usage: TextureUsages::STORAGE_BINDING
            | TextureUsages::TEXTURE_BINDING
            | TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = texture.create_view(&TextureViewDescriptor {
        dimension: Some(TextureViewDimension::D2Array),
        ..default()
    });
    FoamArray { texture, view }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn prepare_water_foam(
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    pipeline_cache: Res<PipelineCache>,
    pipeline: Res<WaterFoamPipeline>,
    frame: Res<WaterFoamFrame>,
    lod: Option<Res<WaterLodData>>,
    lod_gpu: Res<WaterLodGpu>,
    images: Res<RenderAssets<GpuImage>>,
    mut sim: ResMut<WaterFoamSim>,
) {
    sim.groups = None;
    let Some(lod) = lod else {
        return;
    };
    let (Some(state), Some(source), Some(drift), Some(flow), Some(slices)) = (
        images.get(&lod.foam_state),
        images.get(&lod.foam),
        images.get(&lod.drift),
        images.get(&lod.foam_flow),
        lod_gpu.slices.binding(),
    ) else {
        return;
    };
    let size = state.texture_descriptor.size;
    if sim.arrays.is_none() || sim.size != size {
        sim.arrays = Some([
            foam_array(&device, size, WATER_FOAM_STATE_FORMAT, "water_foam_forward"),
            foam_array(
                &device,
                size,
                WATER_FOAM_STATE_FORMAT,
                "water_foam_corrected",
            ),
        ]);
        sim.size = size;
    }
    let flow_size = flow.texture_descriptor.size;
    if sim.flow.is_none() || sim.flow_size != flow_size {
        sim.flow = Some(foam_array(
            &device,
            flow_size,
            WATER_FOAM_FLOW_FORMAT,
            "water_foam_flow",
        ));
        sim.flow_size = flow_size;
        sim.flow_seeded = false;
    }
    let ready = [pipeline.forward, pipeline.correct, pipeline.flow]
        .iter()
        .all(|id| pipeline_cache.get_compute_pipeline(*id).is_some());
    let restart = if sim.flow_seeded { 0.0 } else { 1.0 };
    sim.params.set(WaterFoamParams {
        restart,
        ..frame.params
    });
    sim.flow_seeded |= ready;
    sim.params.write_buffer(&device, &queue);
    let Some(params) = sim.params.binding() else {
        return;
    };
    let (Some([forward, corrected]), Some(flow_out)) = (&sim.arrays, &sim.flow) else {
        return;
    };
    let layout = pipeline_cache.get_bind_group_layout(&pipeline.layout);
    // The forward pass never reads `forward_foam`, so it binds the corrected array there to keep
    // the layout shared.
    let group = |forward_in: &TextureView, out: &TextureView| {
        device.create_bind_group(
            "water_foam_bind_group",
            &layout,
            &BindGroupEntries::sequential((
                params.clone(),
                slices.clone(),
                &state.texture_view,
                forward_in,
                &source.texture_view,
                &drift.texture_view,
                out,
                &flow.texture_view,
                &flow_out.view,
            )),
        )
    };
    let groups = [
        group(&corrected.view, &forward.view),
        group(&forward.view, &corrected.view),
    ];
    sim.groups = Some(groups);
    let res = lod.slices.resolution;
    sim.dispatch = UVec3::new(res.div_ceil(8), res.div_ceil(8), lod.slices.count);
}

pub(super) fn run_water_foam(
    mut ctx: RenderContext,
    pipeline: Res<WaterFoamPipeline>,
    cache: Res<PipelineCache>,
    sim: Res<WaterFoamSim>,
    lod: Option<Res<WaterLodData>>,
    images: Res<RenderAssets<GpuImage>>,
) {
    let (Some(groups), Some(arrays), Some(flow_array), Some(lod)) =
        (&sim.groups, &sim.arrays, &sim.flow, lod)
    else {
        return;
    };
    let (Some(forward), Some(correct), Some(flow)) = (
        cache.get_compute_pipeline(pipeline.forward),
        cache.get_compute_pipeline(pipeline.correct),
        cache.get_compute_pipeline(pipeline.flow),
    ) else {
        return;
    };
    let (Some(state), Some(flow_state)) = (images.get(&lod.foam_state), images.get(&lod.foam_flow))
    else {
        return;
    };
    let diagnostics = ctx.diagnostic_recorder();
    let diagnostics = diagnostics.as_deref();
    let encoder = ctx.command_encoder();
    let time_span = diagnostics.time_span(encoder, "water_foam");
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("water_foam_pass"),
            timestamp_writes: None,
        });
        let d = sim.dispatch;
        pass.set_pipeline(forward);
        pass.set_bind_group(0, &groups[0], &[]);
        pass.dispatch_workgroups(d.x, d.y, d.z);
        pass.set_pipeline(correct);
        pass.set_bind_group(0, &groups[1], &[]);
        pass.dispatch_workgroups(d.x, d.y, d.z);
        pass.set_pipeline(flow);
        pass.dispatch_workgroups(d.x, d.y, d.z);
    }
    encoder.copy_texture_to_texture(
        arrays[1].texture.as_image_copy(),
        state.texture.as_image_copy(),
        sim.size,
    );
    encoder.copy_texture_to_texture(
        flow_array.texture.as_image_copy(),
        flow_state.texture.as_image_copy(),
        sim.flow_size,
    );
    time_span.end(encoder);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Foam after `steps` steps, starting from full active foam, with breaking `source` held.
    fn foam_after(settings: &WaterFoamSettings, dt: f32, steps: u32, source: f32) -> (f32, f32) {
        let p = WaterFoamParams::step(settings, WaterLodSlices::default(), dt);
        let (mut a, mut r) = (1.0f32, 0.0f32);
        for _ in 0..steps {
            let next_a = a * p.active_decay + source * p.source_active;
            r = r * p.residual_decay + a * p.residual_gain + source * p.source_residual;
            a = next_a;
        }
        (a, r)
    }

    fn residual_after(settings: &WaterFoamSettings, dt: f32, steps: u32) -> (f32, f32) {
        foam_after(settings, dt, steps, 0.0)
    }

    #[test]
    fn sources_do_not_depend_on_step_size() {
        let s = WaterFoamSettings::default();
        let coarse = foam_after(&s, 1.0 / 30.0, 60, 0.3);
        let fine = foam_after(&s, 1.0 / 120.0, 240, 0.3);
        assert!((coarse.0 - fine.0).abs() < 1e-4, "{coarse:?} {fine:?}");
        assert!((coarse.1 - fine.1).abs() < 1e-4, "{coarse:?} {fine:?}");
    }

    #[test]
    fn held_breaking_settles_at_rate_over_decay() {
        // da/dt = b / t_f - la a settles at a = b / (t_f la), and r = k la a / lr.
        let s = WaterFoamSettings::default();
        let (a, r) = foam_after(&s, 0.05, 2000, 0.1);
        let la = LN_2 / s.active_half_life;
        let lr = LN_2 / s.residual_half_life;
        let a_ss = 0.1 / (s.formation_time * la);
        assert!((a / a_ss - 1.0).abs() < 1e-3, "{a} {a_ss}");
        assert!((r / (s.residual_yield * la * a_ss / lr) - 1.0).abs() < 1e-3);
    }

    #[test]
    fn formation_time_fills_a_texel_without_decay() {
        let s = WaterFoamSettings {
            active_half_life: f32::INFINITY,
            ..default()
        };
        let p = WaterFoamParams::step(&s, WaterLodSlices::default(), s.formation_time);
        assert!((p.source_active - 1.0).abs() < 1e-5);
        assert_eq!(p.source_residual, 0.0);
    }

    #[test]
    fn decay_does_not_depend_on_step_size() {
        let s = WaterFoamSettings::default();
        let coarse = residual_after(&s, 1.0 / 30.0, 60);
        let fine = residual_after(&s, 1.0 / 120.0, 240);
        assert!((coarse.0 - fine.0).abs() < 1e-5, "{coarse:?} {fine:?}");
        assert!((coarse.1 - fine.1).abs() < 1e-5, "{coarse:?} {fine:?}");
    }

    #[test]
    fn active_foam_halves_after_its_half_life() {
        let s = WaterFoamSettings::default();
        let (a, _) = residual_after(&s, s.active_half_life / 10.0, 10);
        assert!((a - 0.5).abs() < 1e-5);
    }

    #[test]
    fn steady_state_ratio_follows_the_half_lives() {
        // A constant source of active foam settles at r / a = yield * residual / active half-life.
        let s = WaterFoamSettings::default();
        let p = WaterFoamParams::step(&s, WaterLodSlices::default(), 0.01);
        let a = 1.0;
        let r = a * p.residual_gain / (1.0 - p.residual_decay);
        let expected = s.residual_yield * s.residual_half_life / s.active_half_life;
        assert!((r / a / expected - 1.0).abs() < 0.01, "{}", r / a);
    }

    #[test]
    fn equal_half_lives_use_the_limit() {
        let s = WaterFoamSettings {
            active_half_life: 1.0,
            residual_half_life: 1.0,
            residual_yield: 1.0,
            ..default()
        };
        let coarse = residual_after(&s, 0.1, 10);
        let fine = residual_after(&s, 0.01, 100);
        assert!((coarse.1 - fine.1).abs() < 1e-5);
        assert!(coarse.1.is_finite() && coarse.1 > 0.0);
    }

    #[test]
    fn infinite_half_lives_keep_foam() {
        let s = WaterFoamSettings {
            active_half_life: f32::INFINITY,
            residual_half_life: f32::INFINITY,
            residual_yield: 1.0,
            ..default()
        };
        let p = WaterFoamParams::step(&s, WaterLodSlices::default(), 0.1);
        assert_eq!(
            (p.active_decay, p.residual_decay, p.residual_gain),
            (1.0, 1.0, 0.0)
        );
    }
}
