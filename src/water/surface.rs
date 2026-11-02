use bevy::{
    asset::{Asset, Handle},
    pbr::{ExtendedMaterial, MaterialExtension, StandardMaterial},
    prelude::*,
    reflect::{Reflect, TypePath},
    render::{
        render_resource::{AsBindGroup, ShaderType},
        storage::ShaderBuffer,
    },
    shader::ShaderRef,
};

use super::{WaterLodData, WaterLodSlices, WaterQuadtreeBuffer, WaterQuadtreeFallback};

pub const WATER_SURFACE_SHADER: &str = "embedded://bevy_fft/water/surface.wesl";

#[derive(Clone, Copy, ShaderType, Reflect)]
pub struct WaterSurfaceUniform {
    /// Scales foam opacity. At one, dense foam reflects as Koepke (1984) measured.
    pub foam_intensity: f32,
    /// Extra slices added when vertices sample the LOD data, so vertex spacing never undersamples
    /// the data. Pixels sample at the natural level.
    pub vertex_lod_bias: f32,
    /// Water depth in meters below which pixels are discarded, so a terrain under a shallow water
    /// grid shows through on dry ground. Zero disables this solver-depth cutoff.
    pub dry_depth: f32,
    /// Seconds that drive the foam texture's flow phases. [`super::WaterPlugin`] sets it each
    /// frame.
    pub time: f32,
}

/// Optical settings for a transmissive water material. Distances are in meters.
/// Add [`super::WaterTransmission`] to the viewer and use a forward `StandardMaterial`
/// with positive `specular_transmission` and a water IOR near 1.333.
#[derive(Clone, Copy, ShaderType, Reflect)]
pub struct WaterOpticsUniform {
    /// RGB extinction coefficients in inverse meters. [`JerlovWaterType::extinction`] gives
    /// measured values for natural waters.
    pub extinction: Vec3,
    /// Weight of validated screen-space refraction, from 0 to 1.
    pub refraction_strength: f32,
    /// Linear diffuse tint approximating light scattered inside the water.
    pub scattering: Vec3,
    /// Minimum contact transition along the viewing ray, in meters.
    pub contact_fade: f32,
    /// Maximum distance traced through the opaque scene, in meters.
    pub max_distance: f32,
    /// Screen-space reflection weight. Environment lighting covers missing hits.
    pub ssr_strength: f32,
    /// Zero shades water. One shows distance, two transmittance, three coverage, four refraction hits.
    pub debug_view: u32,
    /// Nonzero composites water over the captured opaque scene.
    pub enabled: u32,
}

impl Default for WaterOpticsUniform {
    fn default() -> Self {
        Self {
            extinction: JerlovWaterType::OceanIB.extinction(),
            refraction_strength: 0.8,
            scattering: Vec3::new(0.008, 0.035, 0.065),
            contact_fade: 0.02,
            max_distance: 128.0,
            ssr_strength: 1.0,
            debug_view: 0,
            enabled: 0,
        }
    }
}

impl WaterOpticsUniform {
    /// Default optics with extinction taken from a Jerlov water type.
    pub fn jerlov(water: JerlovWaterType) -> Self {
        Self {
            extinction: water.extinction(),
            ..default()
        }
    }
}

/// Jerlov's classification of natural waters by how fast daylight fades with depth. The five
/// open ocean types run from the clearest, I, to III. The five coastal types run from 1 to 9 as
/// suspended sediment and dissolved organic matter grow.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Reflect)]
pub enum JerlovWaterType {
    OceanI,
    OceanIA,
    #[default]
    OceanIB,
    OceanII,
    OceanIII,
    Coastal1,
    Coastal3,
    Coastal5,
    Coastal7,
    Coastal9,
}

impl JerlovWaterType {
    /// Diffuse downwelling attenuation K_d in inverse meters at 625, 550, and 450 nm, read from
    /// Jerlov's 1976 table for the upper 10 m. Used as RGB Beer-Lambert extinction, it slightly
    /// underestimates beam attenuation, since K_d leaves out light scattered forward.
    pub fn extinction(self) -> Vec3 {
        match self {
            Self::OceanI => Vec3::new(0.305, 0.063, 0.019),
            Self::OceanIA => Vec3::new(0.310, 0.067, 0.026),
            Self::OceanIB => Vec3::new(0.315, 0.072, 0.036),
            Self::OceanII => Vec3::new(0.335, 0.080, 0.053),
            Self::OceanIII => Vec3::new(0.346, 0.104, 0.094),
            Self::Coastal1 => Vec3::new(0.37, 0.12, 0.17),
            Self::Coastal3 => Vec3::new(0.40, 0.17, 0.29),
            Self::Coastal5 => Vec3::new(0.45, 0.25, 0.43),
            Self::Coastal7 => Vec3::new(0.50, 0.36, 0.70),
            Self::Coastal9 => Vec3::new(0.64, 0.52, 1.1),
        }
    }

    /// Absorption coefficient σ_a in inverse meters at 625, 550, and 450 nm, from the `a` columns
    /// of Solonenko and Mobley (2015, Tables 4 to 8). They fitted these inherent optical
    /// properties to the Jerlov types so that Hydrolight reproduces Jerlov's K_d. Table 7 prints
    /// the 300 to 400 nm rows again for type 5C from 600 nm on, so the 5C value at 625 nm is read
    /// from the curve of Monzon et al. (2024, Fig. 2) instead.
    pub fn absorption(self) -> Vec3 {
        match self {
            Self::OceanI => Vec3::new(0.295, 0.0620, 0.0180),
            Self::OceanIA => Vec3::new(0.295, 0.0622, 0.0221),
            Self::OceanIB => Vec3::new(0.296, 0.0623, 0.0235),
            Self::OceanII => Vec3::new(0.296, 0.0622, 0.0241),
            Self::OceanIII => Vec3::new(0.297, 0.0646, 0.0388),
            Self::Coastal1 => Vec3::new(0.304, 0.0760, 0.105),
            Self::Coastal3 => Vec3::new(0.307, 0.0820, 0.154),
            Self::Coastal5 => Vec3::new(0.327, 0.117, 0.297),
            Self::Coastal7 => Vec3::new(0.367, 0.195, 0.542),
            Self::Coastal9 => Vec3::new(0.436, 0.348, 0.943),
        }
    }

    /// Scattering coefficient σ_s in inverse meters at 625, 550, and 450 nm, from the `b` columns
    /// of the same tables as [`Self::absorption`], with the same exception for type 5C.
    pub fn scattering(self) -> Vec3 {
        match self {
            Self::OceanI => Vec3::new(0.00104, 0.00170, 0.00381),
            Self::OceanIA => Vec3::new(0.00256, 0.00354, 0.00631),
            Self::OceanIB => Vec3::new(0.0468, 0.0536, 0.0680),
            Self::OceanII => Vec3::new(0.288, 0.358, 0.504),
            Self::OceanIII => Vec3::new(0.788, 0.980, 1.38),
            Self::Coastal1 => Vec3::new(0.293, 0.365, 0.514),
            Self::Coastal3 => Vec3::new(0.855, 1.06, 1.50),
            Self::Coastal5 => Vec3::new(1.07, 1.33, 1.87),
            Self::Coastal7 => Vec3::new(1.89, 2.35, 3.30),
            Self::Coastal9 => Vec3::new(2.51, 3.12, 4.39),
        }
    }
}

/// The water body as a participating medium, seen when the camera looks out from under the
/// surface. RGB lanes hold 625, 550, and 450 nm. Monzon et al. (2024, §4.2) light the medium with
/// a closed-form multiple-scattering term that takes the downwelling irradiance to fall as
/// `exp(-K_d y)` with depth `y`.
#[derive(Clone, Copy, Debug, PartialEq, ShaderType, Reflect)]
pub struct WaterMediumUniform {
    /// Absorption coefficient σ_a in inverse meters.
    pub absorption: Vec3,
    /// Lambertian reflectance of the bed under the water, from 0 to 1. Zero leaves out the light
    /// the bed sends back up.
    pub floor_albedo: f32,
    /// Scattering coefficient σ_s in inverse meters.
    pub scattering: Vec3,
    /// Index of refraction of the water.
    pub ior: f32,
    /// Diffuse downwelling attenuation K_d in inverse meters.
    pub diffuse_attenuation: Vec3,
    pub _pad: f32,
}

impl Default for WaterMediumUniform {
    fn default() -> Self {
        Self::jerlov(JerlovWaterType::OceanIB)
    }
}

impl WaterMediumUniform {
    /// The medium of a Jerlov water type, with no light from the bed.
    pub fn jerlov(water: JerlovWaterType) -> Self {
        Self {
            absorption: water.absorption(),
            floor_albedo: 0.0,
            scattering: water.scattering(),
            ior: 1.333,
            diffuse_attenuation: water.extinction(),
            _pad: 0.0,
        }
    }

    /// Extinction coefficient σ_t = σ_a + σ_s in inverse meters.
    pub fn extinction(&self) -> Vec3 {
        self.absorption + self.scattering
    }
}

/// Water surface that reads the shared [`WaterLodData`]. Bindings start at 100 so they stack with
/// [`StandardMaterial`].
///
/// Solvers reach the material only through the LOD data, so one material shows the FFT ocean,
/// shallow water, and dispersive patches together. The surface samples by world position, so any
/// mesh layout works and the water stays anchored when the mesh moves. [`super::WaterPlugin`]
/// keeps `lod`, `lod_displacement`, `lod_foam`, `lod_foam_state`, and `lod_foam_flow` in sync with the
/// LOD data each frame, sets `foam_texture`, the time, and the medium, and points `quadtree` at the entity's [`WaterQuadtreeBuffer`] or at
/// the shared fallback.
#[derive(Asset, TypePath, AsBindGroup, Clone)]
pub struct WaterSurfaceExtension {
    #[uniform(100)]
    pub settings: WaterSurfaceUniform,
    #[uniform(101)]
    pub lod: WaterLodSlices,
    #[texture(102, dimension = "2d_array")]
    #[sampler(105)]
    pub lod_displacement: Handle<Image>,
    #[texture(103, dimension = "2d_array")]
    pub lod_foam: Handle<Image>,
    /// Tile list that places a [`super::WaterQuadtree`] pool mesh.
    #[storage(104, read_only)]
    pub quadtree: Handle<ShaderBuffer>,
    /// Opaque-only depth belonging to the water viewer. The plugin keeps it current.
    #[texture(106, sample_type = "depth")]
    pub opaque_depth: Option<Handle<Image>>,
    #[uniform(107)]
    pub optics: WaterOpticsUniform,
    /// Active and residual foam from the foam sim. Full precision, so the shader blends it with loads.
    #[texture(108, dimension = "2d_array", sample_type = "float", filterable = false)]
    pub lod_foam_state: Handle<Image>,
    /// Texture coordinates of the foam pattern, carried along the drift by the foam sim.
    #[texture(111, dimension = "2d_array", sample_type = "float", filterable = false)]
    pub lod_foam_flow: Handle<Image>,
    /// Tiling foam pattern thresholded by coverage, from [`super::WaterFoamTexture`].
    #[texture(109)]
    #[sampler(110)]
    pub foam_texture: Handle<Image>,
    /// The water body seen from below the surface. [`super::WaterPlugin`] copies it from
    /// [`super::WaterMedium`] each frame. The underside draws only when the material leaves back
    /// faces unculled and renders in the forward pass, since the deferred G-buffer pass has no
    /// lights or sky to shade it with.
    #[uniform(112)]
    pub medium: WaterMediumUniform,
    /// Caustic map from [`super::WaterCaustics`], which the single-scattered sunlight under the
    /// underside follows. The plugin keeps it and its placement current.
    #[texture(113, sample_type = "float", filterable = false)]
    pub caustic_map: Handle<Image>,
    #[uniform(114)]
    pub caustics: super::caustics::CausticsLookup,
}

impl Default for WaterSurfaceExtension {
    fn default() -> Self {
        Self {
            settings: WaterSurfaceUniform {
                foam_intensity: 1.0,
                vertex_lod_bias: 1.0,
                dry_depth: 0.0,
                time: 0.0,
            },
            lod: WaterLodSlices::default(),
            lod_displacement: Handle::default(),
            lod_foam: Handle::default(),
            quadtree: Handle::default(),
            opaque_depth: None,
            optics: default(),
            lod_foam_state: Handle::default(),
            lod_foam_flow: Handle::default(),
            foam_texture: Handle::default(),
            medium: default(),
            caustic_map: Handle::default(),
            caustics: default(),
        }
    }
}

impl MaterialExtension for WaterSurfaceExtension {
    fn vertex_shader() -> ShaderRef {
        WATER_SURFACE_SHADER.into()
    }

    fn fragment_shader() -> ShaderRef {
        WATER_SURFACE_SHADER.into()
    }

    fn prepass_vertex_shader() -> ShaderRef {
        WATER_SURFACE_SHADER.into()
    }

    fn prepass_fragment_shader() -> ShaderRef {
        WATER_SURFACE_SHADER.into()
    }

    fn deferred_vertex_shader() -> ShaderRef {
        WATER_SURFACE_SHADER.into()
    }

    fn deferred_fragment_shader() -> ShaderRef {
        WATER_SURFACE_SHADER.into()
    }
}

/// [`StandardMaterial`] shading with the water LOD data displacing the mesh.
pub type WaterSurfaceMaterial = ExtendedMaterial<StandardMaterial, WaterSurfaceExtension>;

/// Marks the entity that carries the [`WaterSurfaceMaterial`] mesh.
#[derive(Component)]
pub struct WaterSurfaceTag;

/// Copies the current slice placement and texture handles from the [`WaterLodData`] into every
/// water surface material, along with the tile list it draws.
#[allow(clippy::too_many_arguments)]
pub(super) fn sync_water_surface_lod(
    lod: Option<Res<WaterLodData>>,
    fallback: Option<Res<WaterQuadtreeFallback>>,
    scene_depth: Option<Res<super::scene_depth::WaterSceneDepth>>,
    foam_texture: Option<Res<super::WaterFoamTexture>>,
    medium: Option<Res<super::WaterMedium>>,
    caustics: Option<Res<super::caustics::WaterCausticsFrame>>,
    time: Res<Time>,
    surface: Query<
        (
            &MeshMaterial3d<WaterSurfaceMaterial>,
            Option<&WaterQuadtreeBuffer>,
        ),
        With<WaterSurfaceTag>,
    >,
    mut materials: ResMut<Assets<WaterSurfaceMaterial>>,
) {
    let (Some(lod), Some(fallback)) = (lod, fallback) else {
        return;
    };
    for (handle, quadtree) in &surface {
        if let Some(mut mat) = materials.get_mut(handle) {
            let tiles = quadtree.map_or(&fallback.0, |q| &q.0);
            if mat.extension.quadtree != *tiles {
                mat.extension.quadtree = tiles.clone();
            }
            mat.extension.lod = lod.slices;
            mat.extension.settings.time = time.elapsed_secs_wrapped();
            if let Some(medium) = &medium {
                mat.extension.medium = medium.0;
            }
            if let Some(caustics) = &caustics {
                mat.extension.caustics = caustics.lookup();
                if mat.extension.caustic_map != *caustics.map() {
                    mat.extension.caustic_map = caustics.map().clone();
                }
            }
            if let Some(scene_depth) = &scene_depth
                && mat.extension.opaque_depth.as_ref() != Some(&scene_depth.image)
            {
                mat.extension.opaque_depth = Some(scene_depth.image.clone());
            }
            if mat.extension.lod_displacement != lod.displacement {
                mat.extension.lod_displacement = lod.displacement.clone();
            }
            if mat.extension.lod_foam != lod.foam {
                mat.extension.lod_foam = lod.foam.clone();
            }
            if mat.extension.lod_foam_state != lod.foam_state {
                mat.extension.lod_foam_state = lod.foam_state.clone();
            }
            if mat.extension.lod_foam_flow != lod.foam_flow {
                mat.extension.lod_foam_flow = lod.foam_flow.clone();
            }
            if let Some(foam_texture) = &foam_texture
                && mat.extension.foam_texture != foam_texture.0
            {
                mat.extension.foam_texture = foam_texture.0.clone();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn open_ocean_scattering_grows_from_type_i_to_iii() {
        let types = [
            JerlovWaterType::OceanI,
            JerlovWaterType::OceanIA,
            JerlovWaterType::OceanIB,
            JerlovWaterType::OceanII,
            JerlovWaterType::OceanIII,
        ];
        for pair in types.windows(2) {
            let (a, b) = (pair[0].scattering(), pair[1].scattering());
            assert!(b.cmpgt(a).all(), "{:?} against {:?}", pair[0], pair[1]);
        }
    }

    #[test]
    fn medium_extinction_sums_absorption_and_scattering() {
        let m = WaterMediumUniform::jerlov(JerlovWaterType::Coastal5);
        let expected = Vec3::new(1.397, 1.447, 2.167);
        assert!((m.extinction() - expected).abs().max_element() < 1e-6);
    }
}
