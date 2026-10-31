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
    /// Brighter foam in lit shading. Blends albedo toward white by `foam_coverage * foam_intensity`.
    pub foam_intensity: f32,
    /// Extra slices added when vertices sample the LOD data, so vertex spacing never undersamples
    /// the data. Pixels sample at the natural level.
    pub vertex_lod_bias: f32,
    /// Water depth in meters below which pixels are discarded, so a terrain under a shallow water
    /// grid shows through on dry ground. Zero disables this solver-depth cutoff.
    pub dry_depth: f32,
    pub _pad1: f32,
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
}

/// Water surface that reads the shared [`WaterLodData`]. Bindings start at 100 so they stack with
/// [`StandardMaterial`].
///
/// Solvers reach the material only through the LOD data, so one material shows the FFT ocean,
/// shallow water, and dispersive patches together. The surface samples by world position, so any
/// mesh layout works and the water stays anchored when the mesh moves. [`super::WaterPlugin`]
/// keeps `lod`, `lod_displacement`, and `lod_foam` in sync with the LOD data each frame, and
/// points `quadtree` at the entity's [`WaterQuadtreeBuffer`] or at the shared fallback.
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
}

impl Default for WaterSurfaceExtension {
    fn default() -> Self {
        Self {
            settings: WaterSurfaceUniform {
                foam_intensity: 0.65,
                vertex_lod_bias: 1.0,
                dry_depth: 0.0,
                _pad1: 0.0,
            },
            lod: WaterLodSlices::default(),
            lod_displacement: Handle::default(),
            lod_foam: Handle::default(),
            quadtree: Handle::default(),
            opaque_depth: None,
            optics: default(),
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
pub(super) fn sync_water_surface_lod(
    lod: Option<Res<WaterLodData>>,
    fallback: Option<Res<WaterQuadtreeFallback>>,
    scene_depth: Option<Res<super::scene_depth::WaterSceneDepth>>,
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
        }
    }
}
