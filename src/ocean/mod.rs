//! FFT ocean: spectrum synthesis, cascades, and height queries.
//!
//! Register [`crate::fft::FftPlugin`] **before** [`OceanPlugin`]. `OceanPlugin::finish` expects FFT render
//! resources to exist; reversing the order will panic at startup. `OceanPlugin` also adds
//! [`crate::water::WaterPlugin`] when it is missing.
//!
//! Spawn one [`crate::fft::FftSource`] with [`OceanSimSettings`] per cascade, each with its own tile
//! size, wavenumber band, and [`OceanCascade`] slot. One inverse FFT per cascade writes height in B,
//! slopes in R and G, and wind-aligned horizontal chop (scalar) in W, and [`OceanFoamMask`] keeps a
//! decaying Jacobian foam history. Each frame the cascades are summed into the shared
//! [`crate::water::WaterLodData`] slices, shaped by the [`OceanSurfaceSettings`] resource.
//! [`crate::water::WaterSurfaceMaterial`] and [`OceanQueries`] read only those slices, so other
//! solvers that write there show up in the same surface. Put the material on an [`OceanClipmap`]
//! for a camera-following mesh, on a [`crate::water::WaterQuadtree`] for a frustum-culled mesh
//! whose tiles grow with distance, or on any other mesh.
//!
//! Treat Bevy X, Z, and up (Y) as meters. [`OceanSimSettings::tile_size`] is the periodic patch size in meters
//! on the horizontal plane (shaders use Δk = 2π / tile_size).
//!
//! The GPU inverse FFT applies a 1/N factor per axis so the 2D result matches the usual inverse of an
//! unnormalized DFT. Skipping that would break the frequency-to-spatial relationship. For a physically
//! meaningful sea state in meters, calibrate variance in `init_h0` (and any global gain) so heights are
//! correct after this normalization, rather than editing `inv_dim` in `ifft.wgsl`.

mod clipmap;
mod lod_write;
mod query;
mod render;

pub use clipmap::OceanClipmap;

pub use query::{
    OCEAN_QUERY_MAX_POINTS, OceanQueries, OceanQueryReadback, OceanQueryResults, OceanSample,
};

use bevy::{
    asset::embedded_asset,
    prelude::*,
    reflect::Reflect,
    render::{
        Render, RenderApp, RenderSystems,
        extract_component::{ExtractComponentPlugin, UniformComponentPlugin},
    },
};

pub use render::{
    OceanComputeBindGroups, OceanDynamicUniform, OceanFoamLabel, OceanFoamMask, OceanFoamPhase,
    OceanFoamUniform, OceanH0Image, OceanH0Uniform, OceanInitTracker, OceanSpectrumLabel,
};

use render::{
    OceanComputePipelines, OceanFoamPipelines, prepare_ocean_compute_bind_groups,
    prepare_ocean_foam_bind_groups, prepare_ocean_foam_mask_image, prepare_ocean_h0_image,
    run_ocean_foam, run_ocean_spectrum, sync_ocean_dynamic_uniform, sync_ocean_foam_display,
    sync_ocean_foam_uniform, sync_ocean_h0_uniform,
};

use crate::fft::{
    FftPlugin, FftSystemSet, disable_spectrum_passthrough, prepare_fft_bind_groups,
    run_fft_forward, run_fft_resolve_outputs, run_fft_resolve_spectrum,
};

/// Same factor as `PM_PEAK_COEFF` in `assets/ocean/init_h0.wgsl` (`ω_pm ≈ this * g / U` in rad/s).
pub const OCEAN_PM_PEAK_COEFF: f32 = 0.87;

/// User-facing simulation parameters on the main world FFT entity. Horizontal extents use meters.
#[derive(Component, Clone, Reflect)]
pub struct OceanSimSettings {
    pub texture_size: u32,
    /// Periodic ocean patch size in meters (edge length along X and Z in Bevy space).
    pub tile_size: f32,
    pub wind_direction: f32,
    /// Fetch / strength (m/s). Sets ω_pm in `init_h0` as [`OCEAN_PM_PEAK_COEFF`] * g / U.
    pub wind_speed: f32,
    pub peak_enhancement: f32,
    pub directional_spread: f32,
    pub small_wave_cutoff: f32,
    pub gravity: f32,
    /// Dimensionless multiplier on top of the Parseval-matched RMS in `init_h0` (1 ≈ nominal sea state).
    pub amplitude_scale: f32,
    /// Multiplier for [`Time::elapsed_secs`] inside [`OceanDynamicUniform::elapsed_seconds`].
    pub time_scale: f32,
    /// Mean water depth in meters. Sets the finite-depth dispersion ω² = gk·tanh(kh) and the TMA
    /// spectrum factor. Values above roughly half the longest wavelength behave like deep water.
    pub depth: f32,
    /// Horvath swell parameter ξ in `[0, 1]`. Zero keeps the wind-sea spreading. Larger values
    /// stretch long waves into parallel trains, as swell that has left its storm does.
    pub swell: f32,
    /// Lowest wavenumber (rad/m) this spectrum keeps. Waves below it belong to a larger cascade.
    pub band_k_min: f32,
    /// Highest wavenumber (rad/m) this spectrum keeps. Waves at or above it belong to a smaller
    /// cascade. Use `f32::INFINITY` for no upper limit.
    pub band_k_max: f32,
    /// Increment to force regeneration of the `H0` texture on the GPU.
    pub h0_serial: u32,
}

impl Default for OceanSimSettings {
    fn default() -> Self {
        Self {
            texture_size: 256,
            tile_size: 16.0,
            wind_direction: 0.0,
            wind_speed: 10.0,
            peak_enhancement: 3.3,
            directional_spread: 0.0,
            small_wave_cutoff: 0.012,
            gravity: 9.81,
            amplitude_scale: 1.0,
            time_scale: 1.0,
            depth: 1000.0,
            swell: 0.0,
            band_k_min: 0.0,
            band_k_max: f32::INFINITY,
            h0_serial: 1,
        }
    }
}

/// Maximum number of FFT cascades the ocean writes into the water LOD data.
pub const OCEAN_MAX_CASCADES: usize = 4;

/// Shaping applied when the ocean cascades are summed into the water LOD data and when foam is
/// accumulated. Shared by every cascade.
#[derive(Resource, Clone, Copy, Debug, PartialEq, Reflect)]
#[reflect(Resource)]
pub struct OceanSurfaceSettings {
    /// Multiplier on the summed cascade heights and slopes.
    pub amplitude: f32,
    /// Multiplier on horizontal wind-aligned displacement, same sense as amplitude on height.
    /// Typical range about 0 to 2.
    pub choppiness: f32,
    /// Jacobian below this value (wave folding) ramps up foam. Typical 0.6 to 1.0.
    pub foam_cutoff: f32,
    /// Width of the transition in Jacobian units. Larger values give softer edges.
    pub foam_falloff: f32,
    /// Per-frame retention of foam from the previous tick. Lower values give shorter trails.
    /// Typical 0.9 to 0.99.
    pub foam_trail_decay: f32,
}

impl Default for OceanSurfaceSettings {
    fn default() -> Self {
        Self {
            amplitude: 1.0,
            choppiness: 0.35,
            foam_cutoff: 0.88,
            foam_falloff: 0.22,
            foam_trail_decay: 0.94,
        }
    }
}

/// Cascade slot of an ocean FFT entity, from 0 to [`OCEAN_MAX_CASCADES`] - 1. Entities without it
/// use slot 0.
#[derive(Component, Clone, Copy, Default, Debug, PartialEq, Eq, Reflect)]
pub struct OceanCascade {
    pub index: u32,
}

/// Stable handles for WGSL registered by [`OceanPlugin`].
pub mod shaders {
    pub const OCEAN_QUERY: &str = "embedded://bevy_fft/ocean/query.wesl";
    pub const OCEAN_LOD_WRITE: &str = "embedded://bevy_fft/ocean/lod_write.wesl";
}

pub struct OceanPlugin;

impl Plugin for OceanPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "query.wesl");
        embedded_asset!(app, "lod_write.wesl");
        if !app.is_plugin_added::<crate::water::WaterPlugin>() {
            app.add_plugins(crate::water::WaterPlugin);
        }
        bevy::shader::load_shader_library!(app, "dispersion.wesl");
        bevy::shader::load_shader_library!(app, "sampling.wesl");

        app.register_type::<OceanSimSettings>()
            .register_type::<OceanH0Uniform>()
            .register_type::<OceanDynamicUniform>()
            .register_type::<OceanH0Image>()
            .register_type::<OceanSurfaceSettings>()
            .init_resource::<OceanSurfaceSettings>()
            .register_type::<OceanCascade>()
            .register_type::<OceanClipmap>()
            .add_systems(Update, clipmap::build_ocean_clipmaps)
            .add_systems(
                PostUpdate,
                clipmap::follow_ocean_clipmaps.before(bevy::transform::TransformSystems::Propagate),
            )
            .register_type::<OceanFoamUniform>()
            .register_type::<OceanFoamMask>()
            .register_type::<OceanFoamPhase>()
            .add_plugins(ExtractComponentPlugin::<OceanH0Image>::default())
            .add_plugins(ExtractComponentPlugin::<OceanH0Uniform>::default())
            .add_plugins(UniformComponentPlugin::<OceanH0Uniform>::default())
            .add_plugins(ExtractComponentPlugin::<OceanDynamicUniform>::default())
            .add_plugins(UniformComponentPlugin::<OceanDynamicUniform>::default())
            .add_plugins(ExtractComponentPlugin::<OceanFoamMask>::default())
            .add_plugins(ExtractComponentPlugin::<OceanFoamPhase>::default())
            .add_plugins(ExtractComponentPlugin::<OceanFoamUniform>::default())
            .add_plugins(UniformComponentPlugin::<OceanFoamUniform>::default())
            .add_systems(
                Update,
                prepare_ocean_h0_image.after(FftSystemSet::PrepareTextures),
            )
            .add_systems(
                Update,
                prepare_ocean_foam_mask_image.after(prepare_ocean_h0_image),
            )
            .add_systems(Update, sync_ocean_h0_uniform.after(prepare_ocean_h0_image))
            .add_systems(
                Update,
                sync_ocean_dynamic_uniform.after(sync_ocean_h0_uniform),
            )
            .add_systems(
                Update,
                sync_ocean_foam_uniform.after(sync_ocean_dynamic_uniform),
            )
            .add_systems(PostUpdate, sync_ocean_foam_display)
            .init_resource::<query::OceanQueryFrame>()
            .add_plugins(bevy::render::extract_resource::ExtractResourcePlugin::<
                query::OceanQueryFrame,
            >::default())
            .init_resource::<lod_write::OceanLodWriteFrame>()
            .add_plugins(bevy::render::extract_resource::ExtractResourcePlugin::<
                lod_write::OceanLodWriteFrame,
            >::default())
            .add_systems(
                PostUpdate,
                (
                    lod_write::gather_ocean_lod_write.after(sync_ocean_foam_display),
                    publish_ocean_chop_direction.before(crate::water::WaterLodSystems),
                ),
            )
            .add_systems(Startup, query::setup_ocean_queries)
            .add_systems(
                PostUpdate,
                query::submit_ocean_queries.after(sync_ocean_foam_display),
            );
    }

    fn finish(&self, app: &mut App) {
        assert!(
            app.is_plugin_added::<FftPlugin>(),
            "OceanPlugin requires FftPlugin to be registered first (e.g. add_plugins((FftPlugin, OceanPlugin)))."
        );
        if app.get_sub_app_mut(RenderApp).is_none() {
            return;
        }

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .init_resource::<OceanComputePipelines>()
            .init_resource::<OceanFoamPipelines>()
            .init_resource::<OceanInitTracker>()
            .init_resource::<lod_write::OceanLodWritePipeline>()
            .init_resource::<lod_write::OceanLodWriteBindGroup>()
            .add_systems(
                Render,
                lod_write::prepare_ocean_lod_write.in_set(RenderSystems::PrepareBindGroups),
            )
            .init_resource::<query::OceanQueryPipeline>()
            .init_resource::<query::OceanQueryBindGroup>()
            .add_systems(
                Render,
                query::prepare_ocean_query_bind_group.in_set(RenderSystems::PrepareBindGroups),
            )
            .add_systems(
                Render,
                prepare_ocean_compute_bind_groups
                    .in_set(RenderSystems::PrepareBindGroups)
                    .after(prepare_fft_bind_groups),
            )
            .add_systems(
                Render,
                prepare_ocean_foam_bind_groups
                    .in_set(RenderSystems::PrepareBindGroups)
                    .after(crate::fft::resources::prepare_fft_resolve_bind_groups),
            );

        disable_spectrum_passthrough(render_app);
        render_app.add_systems(
            bevy::render::renderer::RenderGraph,
            (
                run_ocean_spectrum
                    .after(run_fft_forward)
                    .before(run_fft_resolve_spectrum),
                run_ocean_foam
                    .after(run_fft_resolve_outputs)
                    .before(bevy::core_pipeline::schedule::camera_driver),
            )
                .in_set(bevy::render::renderer::RenderGraphSystems::Render),
        );
        render_app.add_systems(
            bevy::render::renderer::RenderGraph,
            (
                lod_write::run_ocean_lod_write
                    .after(run_ocean_foam)
                    .in_set(crate::water::WaterLodGraphSystems::Base),
                query::run_ocean_query.in_set(crate::water::WaterLodGraphSystems::Read),
            ),
        );
    }
}

/// Tells LOD readers which way the chop lane points. The cascades share one wind, so slot 0
/// decides.
fn publish_ocean_chop_direction(
    lod: Option<ResMut<crate::water::WaterLodData>>,
    cascades: Query<(&OceanSimSettings, Option<&OceanCascade>)>,
) {
    let Some(mut lod) = lod else {
        return;
    };
    let wind = cascades
        .iter()
        .min_by_key(|(_, c)| c.map_or(0, |c| c.index))
        .map(|(s, _)| s.wind_direction);
    if let Some(wind) = wind
        && lod.slices.chop_direction != wind
    {
        lod.slices.chop_direction = wind;
    }
}
