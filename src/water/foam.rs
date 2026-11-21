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
//! The sim also keeps the air fraction of the water column in the blue lane of the foam state, for
//! white water. Writers store in [`WaterLodData::aeration`] the air fraction the flow would hold
//! where it loses energy fast enough to entrain air. The sim carries the air along the drift like
//! foam, without gathering it where the drift converges, raises it toward that value over
//! [`WaterAerationSettings::formation_time`], and lets it leave the column as
//! [`WaterAerationSettings`] describes.
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

use super::{WATER_FOAM_FLOW_FORMAT, WaterLodData, WaterLodGpu, WaterLodSlices};

/// Format of [`WaterLodData::foam_state`] and the sim's working arrays: active foam, residual
/// foam, air fraction, and a spare lane. Half floats would lose a little foam to rounding on every
/// store, so the lanes stay at full precision. Two-lane `Rg32Float` storage is unreliable on Metal.
pub const WATER_FOAM_STATE_FORMAT: TextureFormat = TextureFormat::Rgba32Float;

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

/// How fast white water takes up air and loses it again. The air fraction is the volume of air over
/// the volume of air and water, averaged over the water column.
///
/// The defaults follow the unsubmerged hydraulic jump that Vallé and Pasternack (2006) surveyed in
/// a bedrock step-pool channel: 0.27 m of water at Froude number 2 jumping to about 0.64 m. Air
/// peaked at 40 to 50 percent 0.25 m past the toe, which the inflow at about 3.3 m/s covers in
/// 0.08 s, and fell below 2 percent 2.8 m past it. The tailwater runs at about 1.4 m/s, so the
/// air fell by e about every 0.6 s, which over 0.64 m of water is a detrainment speed near 1 m/s.
/// That is faster than a millimetre bubble rises on its own, about 0.25 m/s, since the roller's
/// air packets are larger and the probe saw only the top 0.1 m.
#[derive(Resource, Clone, Copy, Debug, PartialEq, Reflect)]
#[reflect(Resource)]
pub struct WaterAerationSettings {
    /// Seconds over which entraining flow raises the air fraction to what it can hold.
    pub formation_time: f32,
    /// Meters per second at which air leaves the water column. The air fraction falls by e every
    /// `depth / detrainment_speed` seconds, so deep plunge pools stay white for longer.
    pub detrainment_speed: f32,
    /// Shallower water counts as this deep for the detrainment, in meters, so a thin film doesn't
    /// lose its air in a single frame.
    pub min_depth: f32,
    /// Residual foam coverage that each meter of air leaving the column leaves on the surface, per
    /// meter of air per unit area. The bubbles that leave rise and gather into the streaks of foam
    /// that trail rapids and jumps, which then fade over
    /// [`WaterFoamSettings::residual_half_life`]. A single layer of millimetre bubbles holds about a
    /// millimetre of air, and most bubbles burst as they surface, so the default of 20 lets about
    /// one in fifty of them survive into lasting foam. Water that keeps an air fraction `C` then
    /// holds residual foam of `foam_yield * C * detrainment_speed * residual_half_life / ln 2`,
    /// about `69 C` with the defaults, so it is fully covered above 1.5 percent of air. An
    /// appearance choice.
    pub foam_yield: f32,
}

impl Default for WaterAerationSettings {
    fn default() -> Self {
        Self {
            formation_time: 0.08,
            detrainment_speed: 1.0,
            min_depth: 0.1,
            foam_yield: 20.0,
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
    /// Share of the gap to the air fraction the flow can hold that is left after `dt`.
    pub air_formation: f32,
    /// Meters of water column the air leaves over `dt`.
    pub air_detrainment: f32,
    /// See [`WaterAerationSettings::min_depth`].
    pub air_min_depth: f32,
    /// See [`WaterAerationSettings::foam_yield`].
    pub air_foam: f32,
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
            air_formation: 0.0,
            air_detrainment: 0.0,
            air_min_depth: 1.0,
            air_foam: 0.0,
        }
    }

    /// Sets the air fraction's factors for this step from `settings`.
    pub fn with_aeration(mut self, settings: &WaterAerationSettings) -> Self {
        self.air_formation = (-self.dt / settings.formation_time.max(1e-4)).exp();
        self.air_detrainment = settings.detrainment_speed.max(0.0) * self.dt;
        self.air_min_depth = settings.min_depth.max(1e-3);
        self.air_foam = settings.foam_yield.max(0.0);
        self
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
    aeration: Res<WaterAerationSettings>,
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
        ..WaterFoamParams::step(&settings, previous, time.delta_secs()).with_aeration(&aeration)
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
                storage_buffer_read_only::<u32>(false),
                texture_2d_array(unfiltered),
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
    view: TextureView,
}

/// Views of one half of a [`WaterLodData`] array that holds two frames, for reading last frame's
/// half and writing this frame's.
struct HalfViews {
    texture: TextureId,
    /// Per half, the slices of that half as a 2D array.
    halves: [TextureView; 2],
}

impl HalfViews {
    fn new(texture: &Texture) -> Self {
        let half = texture.depth_or_array_layers() / 2;
        let view = |i: u32| {
            texture.create_view(&TextureViewDescriptor {
                dimension: Some(TextureViewDimension::D2Array),
                base_array_layer: i * half,
                array_layer_count: Some(half),
                ..default()
            })
        };
        Self {
            texture: texture.id(),
            halves: [view(0), view(1)],
        }
    }

    fn refresh(slot: &mut Option<Self>, texture: &Texture) -> bool {
        if slot.as_ref().is_some_and(|v| v.texture == texture.id()) {
            return false;
        }
        *slot = Some(Self::new(texture));
        true
    }
}

/// Render-world working array and bind groups. `forward` holds the semi-Lagrangian step. The
/// corrected foam and the advected texture coordinates go straight into this frame's half of
/// [`WaterLodData::foam_state`] and [`WaterLodData::foam_flow`], which hold two frames each, and
/// the sim reads last frame's half.
#[derive(Resource, Default)]
pub struct WaterFoamSim {
    forward: Option<FoamArray>,
    state_halves: Option<HalfViews>,
    flow_halves: Option<HalfViews>,
    size: Extent3d,
    /// Whether the flow pass has run on the current flow array.
    flow_seeded: bool,
    params: UniformBuffer<WaterFoamParams>,
    groups: Option<[BindGroup; 2]>,
    dispatch: UVec3,
    /// See [`WaterFoamDeposit`].
    pub deposit: WaterFoamDeposit,
}

/// Active foam that other passes add per LOD texel during
/// [`WaterFoamDepositSystems`](super::WaterFoamDepositSystems), as `atomic<u32>` in units of
/// `1 / 65536`. Slice `s` texel `(x, y)` is at `(s * resolution + y) * resolution + x`. The foam
/// sim adds it to the active foam and clears it after its pass.
#[derive(Default, Clone)]
pub struct WaterFoamDeposit {
    pub buffer: Option<Buffer>,
    /// Texels in the buffer.
    pub len: u64,
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
    FoamArray { view }
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
    let (Some(state), Some(source), Some(drift), Some(flow), Some(aeration), Some(slices)) = (
        images.get(&lod.foam_state),
        images.get(&lod.foam),
        images.get(&lod.drift),
        images.get(&lod.foam_flow),
        images.get(&lod.aeration),
        lod_gpu.slices.binding(),
    ) else {
        return;
    };
    // One half of the state array: this frame's slices.
    let size = Extent3d {
        depth_or_array_layers: state.texture_descriptor.size.depth_or_array_layers / 2,
        ..state.texture_descriptor.size
    };
    if sim.forward.is_none() || sim.size != size {
        sim.forward = Some(foam_array(
            &device,
            size,
            WATER_FOAM_STATE_FORMAT,
            "water_foam_forward",
        ));
        sim.size = size;
    }
    HalfViews::refresh(&mut sim.state_halves, &state.texture);
    if HalfViews::refresh(&mut sim.flow_halves, &flow.texture) {
        sim.flow_seeded = false;
    }
    let deposit_len =
        (size.width as u64) * (size.height as u64) * size.depth_or_array_layers as u64;
    if sim.deposit.buffer.is_none() || sim.deposit.len != deposit_len {
        sim.deposit = WaterFoamDeposit {
            buffer: Some(device.create_buffer(&BufferDescriptor {
                label: Some("water_foam_deposit"),
                size: deposit_len * 4,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })),
            len: deposit_len,
        };
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
    let (Some(forward), Some(state_halves), Some(flow_halves), Some(deposit)) = (
        &sim.forward,
        &sim.state_halves,
        &sim.flow_halves,
        &sim.deposit.buffer,
    ) else {
        return;
    };
    // This frame writes half `now` and reads last frame's, the other one.
    let now = lod.slices.foam_half as usize & 1;
    let (state_in, state_out) = (&state_halves.halves[1 - now], &state_halves.halves[now]);
    let (flow_in, flow_out) = (&flow_halves.halves[1 - now], &flow_halves.halves[now]);
    let layout = pipeline_cache.get_bind_group_layout(&pipeline.layout);
    // The forward pass never reads `forward_foam`, so it binds this frame's state half there to
    // keep the layout shared; the forward pass doesn't write it.
    let group = |forward_in: &TextureView, out: &TextureView| {
        device.create_bind_group(
            "water_foam_bind_group",
            &layout,
            &BindGroupEntries::sequential((
                params.clone(),
                slices.clone(),
                state_in,
                forward_in,
                &source.texture_view,
                &drift.texture_view,
                out,
                flow_in,
                flow_out,
                deposit.as_entire_buffer_binding(),
                &aeration.texture_view,
            )),
        )
    };
    let groups = [
        group(state_out, &forward.view),
        group(&forward.view, state_out),
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
) {
    let Some(groups) = &sim.groups else {
        return;
    };
    let (Some(forward), Some(correct), Some(flow)) = (
        cache.get_compute_pipeline(pipeline.forward),
        cache.get_compute_pipeline(pipeline.correct),
        cache.get_compute_pipeline(pipeline.flow),
    ) else {
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
    if let Some(deposit) = &sim.deposit.buffer {
        encoder.clear_buffer(deposit, 0, None);
    }
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
    fn air_factors_do_not_depend_on_step_size() {
        // Over one second the gap to the held air closes by e^(-1 / formation time), and air over
        // a meter of water falls by e^(-detrainment speed).
        let a = WaterAerationSettings::default();
        let s = WaterFoamSettings::default();
        let step =
            |dt: f32| WaterFoamParams::step(&s, WaterLodSlices::default(), dt).with_aeration(&a);
        let (coarse, fine) = (step(1.0 / 30.0), step(1.0 / 120.0));
        let formation = (coarse.air_formation.powi(30), fine.air_formation.powi(120));
        assert!((formation.0 - formation.1).abs() < 1e-6, "{formation:?}");
        assert!((formation.0 - (-1.0 / a.formation_time).exp()).abs() < 1e-6);
        let left = |p: WaterFoamParams, n: i32| (-p.air_detrainment / 1.0f32).exp().powi(n);
        assert!((left(coarse, 30) - (-a.detrainment_speed).exp()).abs() < 1e-5);
        assert!((left(fine, 120) - (-a.detrainment_speed).exp()).abs() < 1e-5);
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
