//! Ocean-style mesh displacement driven by the FFT pipeline `spatial_output` texture.
//!
//! Add [`OceanPlugin`] after [`crate::fft::FftPlugin`]. Use [`OceanSurfaceMaterial`] ([`ExtendedMaterial`]
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

mod render;

use bevy::{
    asset::{Asset, Handle, load_internal_asset},
    color::LinearRgba,
    ecs::change_detection::Mut,
    pbr::{ExtendedMaterial, MaterialExtension, MeshMaterial3d, StandardMaterial},
    prelude::*,
    reflect::{Reflect, TypePath},
    render::{
        Render, RenderApp, RenderSystems,
        extract_component::{ExtractComponentPlugin, UniformComponentPlugin},
        render_graph::RenderGraph,
        render_resource::{AsBindGroup, ShaderType},
    },
    shader::{Shader, ShaderRef},
    transform::TransformSystems,
};

pub use render::{
    OceanComputeBindGroups, OceanDynamicUniform, OceanFoamLabel, OceanFoamMask, OceanFoamPhase,
    OceanFoamUniform, OceanH0Image, OceanH0Uniform, OceanInitTracker, OceanSpectrumLabel,
};

use render::{
    OceanComputePipelines, OceanFoamNode, OceanFoamPipelines, OceanSpectrumNode,
    prepare_ocean_compute_bind_groups, prepare_ocean_foam_bind_groups,
    prepare_ocean_foam_mask_image, prepare_ocean_h0_image, sync_ocean_dynamic_uniform,
    sync_ocean_foam_display, sync_ocean_foam_uniform, sync_ocean_h0_uniform,
};

use crate::fft::{
    FftSystemSet, prepare_fft_bind_groups, splice_after_resolve_outputs, splice_spectrum_pass,
};

/// Same factor as `PM_PEAK_COEFF` in `assets/ocean/init_h0.wgsl` (`ω_pm ≈ this * g / U` in rad/s).
pub const OCEAN_PM_PEAK_COEFF: f32 = 0.87;

/// Bright daylight reference in lux (`lux::RAW_SUNLIGHT` scale). Crest scatter uses
/// `linear_rgb * (illuminance / this)` so emissive matches artist-friendly [`OceanMaterialUniform::crest_scatter_intensity`]
/// instead of raw lux.
pub const CREST_SCATTER_REFERENCE_ILLUMINANCE_LUX: f32 = 130_000.0;

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
            h0_serial: 1,
        }
    }
}

#[derive(Clone, Copy, ShaderType, Reflect)]
pub struct OceanMaterialUniform {
    pub amplitude: f32,
    /// Multiplier on horizontal wind-aligned displacement (alpha channel), same sense as amplitude on height. Typical range about 0 to 2.
    pub choppiness: f32,
    pub ocean_tile_world_size: f32,
    pub grid_size: f32,
    /// Radians on XZ: horizontal unit vector `(cos θ, sin θ)` matches `OceanSimSettings::wind_direction` and the chop factor in `ocean_spectrum`.
    pub wind_direction: f32,
    /// Brighter foam in lit shading. Blends albedo toward [`Self::foam_color`] by `foam_coverage * foam_intensity`.
    pub foam_intensity: f32,
    /// Jacobian below this value (wave folding) ramps up foam; typical 0.6 to 1.0.
    pub foam_cutoff: f32,
    /// Width of the transition in Jacobian units (larger = softer edges).
    pub foam_falloff: f32,
    /// Linear RGB target color for foam when mixing into base albedo (`.w` unused).
    pub foam_color: Vec4,
    /// Additive emissive on steep, back-lit wave fronts so directional light reads through tall peaks (scaled by light color).
    /// Zero disables the effect.
    pub crest_scatter_intensity: f32,
    /// Exponent on `dot(view, light)`; higher concentrates glow when looking toward the light through the crest.
    pub crest_scatter_view_power: f32,
    /// Exponent on `(1 - N·V)` for a thin-film style rim around silhouettes.
    pub crest_scatter_rim_power: f32,
    /// Scales slope magnitude from the displacement map before saturating; larger weights sharper crests.
    pub crest_scatter_slope_scale: f32,
    /// Linear RGB multiplier on crest scatter after the key light color. Stronger green and blue than red reads as turquoise water instead of white sun.
    pub crest_scatter_tint: Vec4,
    /// World-space direction toward the first [`DirectionalLight`] in the scene (Bevy `dir_to_light`). Filled by [`sync_ocean_crest_key_light`].
    pub crest_light_dir_to_light_ws: Vec4,
    /// Linear RGB × (`illuminance` / [`CREST_SCATTER_REFERENCE_ILLUMINANCE_LUX`]) for the key light. Filled by [`sync_ocean_crest_key_light`].
    pub crest_light_radiance: Vec4,
    /// Per-frame retention of foam from the previous tick (lower = shorter trails). Typical 0.9 to 0.99.
    /// Passed to the foam compute shader only; kept here so the material uniform matches the GPU struct.
    pub foam_trail_decay: f32,
}

/// GPU displacement uniforms and FFT height map. Bindings start at 100 so they stack with [`StandardMaterial`].
#[derive(Asset, TypePath, AsBindGroup, Clone)]
pub struct OceanSurfaceExtension {
    #[uniform(100)]
    pub settings: OceanMaterialUniform,
    #[texture(101)]
    #[sampler(102)]
    pub displacement: Handle<Image>,
    #[texture(103)]
    #[sampler(104)]
    pub foam_mask: Handle<Image>,
}

/// Default linear RGB tint for [`OceanMaterialUniform::crest_scatter_tint`] (turquoise bias).
pub fn default_crest_scatter_tint() -> Vec4 {
    let c: LinearRgba = Color::srgb(0.12, 0.66, 0.72).into();
    let a = c.to_f32_array();
    Vec4::new(a[0], a[1], a[2], 0.0)
}

impl Default for OceanSurfaceExtension {
    fn default() -> Self {
        Self {
            settings: OceanMaterialUniform {
                amplitude: 1.0,
                choppiness: 0.35,
                ocean_tile_world_size: 16.0,
                grid_size: 256.0,
                wind_direction: 0.0,
                foam_intensity: 0.65,
                foam_cutoff: 0.88,
                foam_falloff: 0.22,
                foam_color: Vec4::new(1.0, 1.0, 1.0, 0.0),
                crest_scatter_intensity: 0.25,
                crest_scatter_view_power: 2.5,
                crest_scatter_rim_power: 3.0,
                crest_scatter_slope_scale: 1.0,
                crest_scatter_tint: default_crest_scatter_tint(),
                crest_light_dir_to_light_ws: Vec4::ZERO,
                crest_light_radiance: Vec4::ZERO,
                foam_trail_decay: 0.94,
            },
            displacement: Handle::default(),
            foam_mask: Handle::default(),
        }
    }
}

impl MaterialExtension for OceanSurfaceExtension {
    fn vertex_shader() -> ShaderRef {
        ShaderRef::Handle(shaders::OCEAN_SURFACE.clone())
    }

    fn fragment_shader() -> ShaderRef {
        ShaderRef::Handle(shaders::OCEAN_SURFACE.clone())
    }

    fn prepass_vertex_shader() -> ShaderRef {
        ShaderRef::Handle(shaders::OCEAN_SURFACE.clone())
    }

    fn deferred_vertex_shader() -> ShaderRef {
        ShaderRef::Handle(shaders::OCEAN_SURFACE.clone())
    }

    fn deferred_fragment_shader() -> ShaderRef {
        ShaderRef::Handle(shaders::OCEAN_SURFACE.clone())
    }
}

/// Tessendorf surface: [`StandardMaterial`] shading with FFT displacement in the extension.
pub type OceanSurfaceMaterial = ExtendedMaterial<StandardMaterial, OceanSurfaceExtension>;

/// Stable handles for WGSL registered by [`OceanPlugin`].
pub mod shaders {
    use bevy::asset::{Handle, uuid_handle};
    use bevy::shader::Shader;

    pub const OCEAN_SURFACE: Handle<Shader> = uuid_handle!("b7c8e9f0-1a2b-4c3d-9e8f-7a6b5c4d3e2f");
}

/// Marks the entity that carries [`OceanSurfaceMaterial`] for the sim-driven ocean mesh.
#[derive(Component)]
pub struct OceanSurfaceTag;

/// Copies the first directional light into [`OceanMaterialUniform`] so the ocean shader does not bind the full view lights buffer (prepass layouts omit it).
fn sync_ocean_crest_key_light(
    mut materials: ResMut<Assets<OceanSurfaceMaterial>>,
    lights: Query<(Entity, &DirectionalLight, &GlobalTransform)>,
    ocean_surfaces: Query<&MeshMaterial3d<OceanSurfaceMaterial>, With<OceanSurfaceTag>>,
) {
    let (dir_w, rad) = lights
        .iter()
        .max_by(|(ea, la, _), (eb, lb, _)| {
            la.illuminance
                .total_cmp(&lb.illuminance)
                .then(ea.index().cmp(&eb.index()))
        })
        .map(|(_, light, gt)| {
            let d = Vec3::from(gt.back()).normalize_or_zero();
            let c = LinearRgba::from(light.color);
            let a = c.to_f32_array();
            let scale = light.illuminance / CREST_SCATTER_REFERENCE_ILLUMINANCE_LUX;
            let rgb = Vec3::new(a[0], a[1], a[2]) * scale;
            (d.extend(0.0), rgb.extend(0.0))
        })
        .unwrap_or((Vec4::ZERO, Vec4::ZERO));

    for h in &ocean_surfaces {
        if let Some(mat) = materials.get_mut(&h.0) {
            mat.extension.settings.crest_light_dir_to_light_ws = dir_w;
            mat.extension.settings.crest_light_radiance = rad;
        }
    }
}

pub struct OceanPlugin;

impl Plugin for OceanPlugin {
    fn build(&self, app: &mut App) {
        load_internal_asset!(app, shaders::OCEAN_SURFACE, "ocean.wgsl", Shader::from_wgsl);

        app.register_type::<OceanSimSettings>()
            .register_type::<OceanH0Uniform>()
            .register_type::<OceanDynamicUniform>()
            .register_type::<OceanH0Image>()
            .register_type::<OceanMaterialUniform>()
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
            .add_systems(
                PostUpdate,
                sync_ocean_crest_key_light.after(TransformSystems::Propagate),
            )
            .add_systems(PostUpdate, sync_ocean_foam_display);
    }

    fn finish(&self, app: &mut App) {
        if app.get_sub_app_mut(RenderApp).is_none() {
            return;
        }

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        if render_app
            .world()
            .get_resource::<crate::fft::resources::FftBindGroupLayouts>()
            .is_none()
        {
            panic!("OceanPlugin requires FftPlugin (add it before OceanPlugin)");
        }

        render_app
            .init_resource::<OceanComputePipelines>()
            .init_resource::<OceanFoamPipelines>()
            .init_resource::<OceanInitTracker>()
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

        render_app
            .world_mut()
            .resource_scope(|world, mut graph: Mut<RenderGraph>| {
                graph.add_node(OceanSpectrumLabel, OceanSpectrumNode::from_world(world));
                graph.add_node(OceanFoamLabel, OceanFoamNode::from_world(world));
            });
        splice_spectrum_pass(render_app.world_mut(), OceanSpectrumLabel);
        // `FftPlugin::finish` must run first so `ResolveOutputs` → `CameraDriver` exists for `splice_after_resolve_outputs`.
        splice_after_resolve_outputs(render_app.world_mut(), OceanFoamLabel);
    }
}
