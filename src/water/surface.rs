use bevy::{
    asset::{Asset, Handle},
    pbr::{ExtendedMaterial, MaterialExtension, StandardMaterial},
    prelude::*,
    reflect::{Reflect, TypePath},
    render::render_resource::{AsBindGroup, ShaderType},
    shader::ShaderRef,
};

use super::{WaterLodData, WaterLodSlices};

pub const WATER_SURFACE_SHADER: &str = "embedded://bevy_fft/water/surface.wesl";

#[derive(Clone, Copy, ShaderType, Reflect)]
pub struct WaterSurfaceUniform {
    /// Brighter foam in lit shading. Blends albedo toward white by `foam_coverage * foam_intensity`.
    pub foam_intensity: f32,
    /// Extra slices added when vertices sample the LOD data, so vertex spacing never undersamples
    /// the data. Pixels sample at the natural level.
    pub vertex_lod_bias: f32,
    pub _pad0: f32,
    pub _pad1: f32,
}

/// Water surface that reads the shared [`WaterLodData`]. Bindings start at 100 so they stack with
/// [`StandardMaterial`].
///
/// Solvers reach the material only through the LOD data, so one material shows the FFT ocean,
/// shallow water, and dispersive patches together. The surface samples by world position, so any
/// mesh layout works and the water stays anchored when the mesh moves. [`super::WaterPlugin`]
/// keeps `lod`, `lod_displacement`, and `lod_foam` in sync with the LOD data each frame.
#[derive(Asset, TypePath, AsBindGroup, Clone)]
pub struct WaterSurfaceExtension {
    #[uniform(100)]
    pub settings: WaterSurfaceUniform,
    #[uniform(101)]
    pub lod: WaterLodSlices,
    #[texture(102, dimension = "2d_array", filterable = false)]
    pub lod_displacement: Handle<Image>,
    #[texture(103, dimension = "2d_array", filterable = false)]
    pub lod_foam: Handle<Image>,
}

impl Default for WaterSurfaceExtension {
    fn default() -> Self {
        Self {
            settings: WaterSurfaceUniform {
                foam_intensity: 0.65,
                vertex_lod_bias: 1.0,
                _pad0: 0.0,
                _pad1: 0.0,
            },
            lod: WaterLodSlices::default(),
            lod_displacement: Handle::default(),
            lod_foam: Handle::default(),
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
/// water surface material.
pub(super) fn sync_water_surface_lod(
    lod: Option<Res<WaterLodData>>,
    surface: Query<&MeshMaterial3d<WaterSurfaceMaterial>, With<WaterSurfaceTag>>,
    mut materials: ResMut<Assets<WaterSurfaceMaterial>>,
) {
    let Some(lod) = lod else {
        return;
    };
    for handle in &surface {
        if let Some(mut mat) = materials.get_mut(handle) {
            mat.extension.lod = lod.slices;
            if mat.extension.lod_displacement != lod.displacement {
                mat.extension.lod_displacement = lod.displacement.clone();
            }
            if mat.extension.lod_foam != lod.foam {
                mat.extension.lod_foam = lod.foam.clone();
            }
        }
    }
}
