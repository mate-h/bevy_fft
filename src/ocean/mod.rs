//! Ocean-style mesh displacement driven by the FFT pipeline `spatial_output` texture.
//!
//! Register [`crate::fft::FftPlugin`] **before** [`OceanPlugin`]. `OceanPlugin::finish` expects FFT render
//! resources to exist; reversing the order will panic at startup. Use [`OceanSurfaceMaterial`] ([`ExtendedMaterial`]
//! of [`StandardMaterial`] and [`OceanSurfaceExtension`]) on a subdivided plane; drive FFT with
//! [`OceanSimSettings`] and [`crate::fft::FftSchedule::Inverse`]. One inverse FFT writes height in B,
//! slopes in R and G, and wind-aligned horizontal chop (scalar) in W.
//! [`OceanFoamMask`] ping-pong targets are updated after the FFT resolve from a Jacobian of that chop field, with temporal decay so foam leaves trails.
//! The vertex shader applies the chop scalar along [`OceanMaterialUniform::wind_direction`] on XZ.
//! A full 2D chop still needs another band or pass.
//!
//! Treat Bevy X, Z, and up (Y) as meters. [`OceanSimSettings::tile_size`] is the periodic patch size in meters
//! on the horizontal plane (shaders use Δk = 2π / tile_size).
//!
//! The GPU inverse FFT applies a 1/N factor per axis so the 2D result matches the usual inverse of an
//! unnormalized DFT. Skipping that would break the frequency-to-spatial relationship. For a physically
//! meaningful sea state in meters, calibrate variance in `init_h0` (and any global gain) so heights are
//! correct after this normalization, rather than editing `inv_dim` in `ifft.wgsl`.

mod clipmap;
mod query;
mod render;

pub use clipmap::{OceanClipmap, OceanClipmapTarget};

pub use query::{
    OCEAN_QUERY_MAX_POINTS, OceanQueries, OceanQueryReadback, OceanQueryResults, OceanSample,
};

use bevy::{
    asset::{Asset, Handle, embedded_asset},
    pbr::{ExtendedMaterial, MaterialExtension, StandardMaterial},
    prelude::*,
    reflect::{Reflect, TypePath},
    render::{
        Render, RenderApp, RenderSystems,
        extract_component::{ExtractComponentPlugin, UniformComponentPlugin},
        render_resource::{AsBindGroup, ShaderType},
    },
    shader::ShaderRef,
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

/// Maximum number of FFT cascades one [`OceanSurfaceMaterial`] can sum.
pub const OCEAN_MAX_CASCADES: usize = 4;

#[derive(Clone, Copy, ShaderType, Reflect)]
pub struct OceanMaterialUniform {
    /// World-space tile edge in meters for each cascade slot. Must match each cascade's
    /// [`OceanSimSettings::tile_size`].
    pub cascade_tile_sizes: Vec4,
    pub amplitude: f32,
    /// Multiplier on horizontal wind-aligned displacement (alpha channel), same sense as amplitude on height. Typical range about 0 to 2.
    pub choppiness: f32,
    /// Radians on XZ: horizontal unit vector `(cos θ, sin θ)` matches `OceanSimSettings::wind_direction` and the chop factor in `ocean_spectrum`.
    pub wind_direction: f32,
    /// Number of filled cascade slots, from 1 to [`OCEAN_MAX_CASCADES`].
    pub cascade_count: u32,
    /// Brighter foam in lit shading. Blends albedo toward white by `foam_coverage * foam_intensity`.
    pub foam_intensity: f32,
    /// Jacobian below this value (wave folding) ramps up foam; typical 0.6 to 1.0.
    pub foam_cutoff: f32,
    /// Width of the transition in Jacobian units (larger = softer edges).
    pub foam_falloff: f32,
    /// Per-frame retention of foam from the previous tick (lower = shorter trails). Typical 0.9 to 0.99.
    /// Read by the foam compute shader.
    pub foam_trail_decay: f32,
    /// Vertex spacing in meters of the innermost clipmap level. Zero turns off the distance fade
    /// of vertex displacement. [`OceanClipmap`] fills this in.
    pub lod_base_spacing: f32,
    /// Half-width of the innermost clipmap level in cells. The vertex shader estimates local
    /// spacing as `max(lod_base_spacing, distance / lod_cells_per_level)`.
    pub lod_cells_per_level: f32,
    pub _lod_pad0: f32,
    pub _lod_pad1: f32,
}

/// Sums up to four FFT cascades in world space. Bindings start at 100 so they stack with [`StandardMaterial`].
///
/// Each slot holds one cascade's `FftTextures::spatial_output` and foam mask. Slots at or past
/// [`OceanMaterialUniform::cascade_count`] are ignored. The surface samples by world position, so any
/// mesh layout works and the ocean stays anchored when the mesh moves.
#[derive(Asset, TypePath, AsBindGroup, Clone)]
pub struct OceanSurfaceExtension {
    #[uniform(100)]
    pub settings: OceanMaterialUniform,
    #[texture(101, filterable = false)]
    pub displacement_0: Handle<Image>,
    #[texture(102, filterable = false)]
    pub displacement_1: Handle<Image>,
    #[texture(103, filterable = false)]
    pub displacement_2: Handle<Image>,
    #[texture(104, filterable = false)]
    pub displacement_3: Handle<Image>,
    #[texture(105, filterable = false)]
    pub foam_0: Handle<Image>,
    #[texture(106, filterable = false)]
    pub foam_1: Handle<Image>,
    #[texture(107, filterable = false)]
    pub foam_2: Handle<Image>,
    #[texture(108, filterable = false)]
    pub foam_3: Handle<Image>,
}

impl OceanSurfaceExtension {
    /// Puts one cascade's textures into slot `index` (0 to 3) and records its tile size.
    pub fn set_cascade(
        &mut self,
        index: usize,
        tile_size: f32,
        displacement: Handle<Image>,
        foam: Handle<Image>,
    ) {
        assert!(
            index < OCEAN_MAX_CASCADES,
            "cascade index {index} out of range"
        );
        self.settings.cascade_tile_sizes[index] = tile_size;
        let (d, f) = match index {
            0 => (&mut self.displacement_0, &mut self.foam_0),
            1 => (&mut self.displacement_1, &mut self.foam_1),
            2 => (&mut self.displacement_2, &mut self.foam_2),
            _ => (&mut self.displacement_3, &mut self.foam_3),
        };
        *d = displacement;
        *f = foam;
    }

    /// Foam texture handle in slot `index`.
    pub fn foam(&self, index: usize) -> &Handle<Image> {
        match index {
            0 => &self.foam_0,
            1 => &self.foam_1,
            2 => &self.foam_2,
            _ => &self.foam_3,
        }
    }

    fn foam_mut(&mut self, index: usize) -> &mut Handle<Image> {
        match index {
            0 => &mut self.foam_0,
            1 => &mut self.foam_1,
            2 => &mut self.foam_2,
            _ => &mut self.foam_3,
        }
    }
}

impl Default for OceanSurfaceExtension {
    fn default() -> Self {
        Self {
            settings: OceanMaterialUniform {
                cascade_tile_sizes: Vec4::splat(16.0),
                amplitude: 1.0,
                choppiness: 0.35,
                wind_direction: 0.0,
                cascade_count: 1,
                foam_intensity: 0.65,
                foam_cutoff: 0.88,
                foam_falloff: 0.22,
                foam_trail_decay: 0.94,
                lod_base_spacing: 0.0,
                lod_cells_per_level: 1.0,
                _lod_pad0: 0.0,
                _lod_pad1: 0.0,
            },
            displacement_0: Handle::default(),
            displacement_1: Handle::default(),
            displacement_2: Handle::default(),
            displacement_3: Handle::default(),
            foam_0: Handle::default(),
            foam_1: Handle::default(),
            foam_2: Handle::default(),
            foam_3: Handle::default(),
        }
    }
}

/// Slot of an ocean FFT entity inside [`OceanSurfaceExtension`]. Entities without it use slot 0.
#[derive(Component, Clone, Copy, Default, Debug, PartialEq, Eq, Reflect)]
pub struct OceanCascade {
    pub index: u32,
}

impl MaterialExtension for OceanSurfaceExtension {
    fn vertex_shader() -> ShaderRef {
        shaders::OCEAN_SURFACE.into()
    }

    fn fragment_shader() -> ShaderRef {
        shaders::OCEAN_SURFACE.into()
    }

    fn deferred_vertex_shader() -> ShaderRef {
        shaders::OCEAN_SURFACE.into()
    }

    fn deferred_fragment_shader() -> ShaderRef {
        shaders::OCEAN_SURFACE.into()
    }
}

/// Tessendorf surface: [`StandardMaterial`] shading with FFT displacement in the extension.
pub type OceanSurfaceMaterial = ExtendedMaterial<StandardMaterial, OceanSurfaceExtension>;

/// Stable handles for WGSL registered by [`OceanPlugin`].
pub mod shaders {
    pub const OCEAN_SURFACE: &str = "embedded://bevy_fft/ocean/ocean.wesl";
    pub const OCEAN_QUERY: &str = "embedded://bevy_fft/ocean/query.wesl";
}

/// Marks the entity that carries [`OceanSurfaceMaterial`] for the sim-driven ocean mesh.
#[derive(Component)]
pub struct OceanSurfaceTag;

pub struct OceanPlugin;

impl Plugin for OceanPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "ocean.wesl");
        embedded_asset!(app, "query.wesl");
        bevy::shader::load_shader_library!(app, "dispersion.wesl");
        bevy::shader::load_shader_library!(app, "sampling.wesl");

        app.register_type::<OceanSimSettings>()
            .register_type::<OceanH0Uniform>()
            .register_type::<OceanDynamicUniform>()
            .register_type::<OceanH0Image>()
            .register_type::<OceanMaterialUniform>()
            .register_type::<OceanCascade>()
            .register_type::<OceanClipmap>()
            .register_type::<OceanClipmapTarget>()
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
            .add_plugins(MaterialPlugin::<OceanSurfaceMaterial>::default())
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
                query::run_ocean_query
                    .after(run_fft_resolve_outputs)
                    .before(bevy::core_pipeline::schedule::camera_driver),
            )
                .in_set(bevy::render::renderer::RenderGraphSystems::Render),
        );
    }
}
