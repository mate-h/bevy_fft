//! Shared water data that every solver writes and every reader samples.
//!
//! [`WaterLodData`] holds camera-centred texture arrays in the style of Crest's LOD data. Each
//! slice covers twice the width of the previous one at the same resolution, so detail is high near
//! the viewer and memory stays bounded however large the level is. Solvers such as the FFT ocean
//! write their contribution into the slices, and the surface material, physics queries, and
//! later caustics read only the slices. Mark the camera that should own the detail with
//! [`WaterViewer`]; without it the first [`Camera3d`] is used.
//!
//! Sampling helpers for WESL live in `bevy_fft::water::lod`.

mod lod_data;

pub use lod_data::{
    WATER_LOD_MAX_SLICES, WaterLodData, WaterLodGpu, WaterLodSettings, WaterLodSlices, WaterViewer,
};

use bevy::{
    prelude::*,
    render::{Render, RenderApp, RenderSystems, extract_resource::ExtractResourcePlugin},
};

/// System set that moves the [`WaterLodData`] slices under the viewer. Writers and readers that
/// need this frame's slice placement run after it.
#[derive(SystemSet, Clone, Debug, PartialEq, Eq, Hash)]
pub struct WaterLodSystems;

/// Registers [`WaterLodData`] and keeps its slices centred on the [`WaterViewer`].
pub struct WaterPlugin;

impl Plugin for WaterPlugin {
    fn build(&self, app: &mut App) {
        bevy::shader::load_shader_library!(app, "lod.wesl");
        app.register_type::<WaterLodSettings>()
            .register_type::<WaterViewer>()
            .init_resource::<WaterLodSettings>()
            .add_plugins(ExtractResourcePlugin::<WaterLodData>::default())
            .add_systems(PreStartup, lod_data::create_water_lod_data)
            .add_systems(
                PostUpdate,
                lod_data::update_water_lod_slices
                    .in_set(WaterLodSystems)
                    .after(bevy::transform::TransformSystems::Propagate),
            );
        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app.init_resource::<WaterLodGpu>().add_systems(
                Render,
                lod_data::prepare_water_lod_gpu.in_set(RenderSystems::PrepareResources),
            );
        }
    }
}
