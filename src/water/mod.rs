//! Shared water data that every solver writes and every reader samples.
//!
//! [`WaterLodData`] holds camera-centred texture arrays in the style of Crest's LOD data. Each
//! slice covers twice the width of the previous one at the same resolution, so detail is high near
//! the viewer and memory stays bounded however large the level is. Solvers such as the FFT ocean
//! write their contribution into the slices, and the surface material, physics queries, and
//! later caustics read only the slices. Mark the camera that should own the detail with
//! [`WaterViewer`]; without it the first [`Camera3d`] is used.
//!
//! [`WaterSurfaceMaterial`] draws the data on any mesh tagged with [`WaterSurfaceTag`].
//!
//! Writers run in the render graph in the order of [`WaterLodGraphSystems`]. A base writer such
//! as the ocean covers every texel. Regional solvers such as shallow water then blend over it
//! through [`WaterLodScratch`], and readers such as height queries run last.
//!
//! Sampling helpers for WESL live in `bevy_fft::water::lod`.

mod blend;
mod lod_data;
mod surface;

pub use blend::WaterLodScratch;
pub use lod_data::{
    WATER_LOD_MAX_SLICES, WaterLodData, WaterLodGpu, WaterLodSettings, WaterLodSlices, WaterViewer,
};
pub use surface::{
    WATER_SURFACE_SHADER, WaterSurfaceExtension, WaterSurfaceMaterial, WaterSurfaceTag,
    WaterSurfaceUniform,
};

use bevy::{
    asset::embedded_asset,
    prelude::*,
    render::{
        Render, RenderApp, RenderSystems,
        extract_resource::ExtractResourcePlugin,
        renderer::{RenderGraph, RenderGraphSystems},
    },
};

/// System set that moves the [`WaterLodData`] slices under the viewer. Writers and readers that
/// need this frame's slice placement run after it.
#[derive(SystemSet, Clone, Debug, PartialEq, Eq, Hash)]
pub struct WaterLodSystems;

/// Render graph stages for passes that touch [`WaterLodData`]. They run in this order, before the
/// cameras draw.
#[derive(SystemSet, Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum WaterLodGraphSystems {
    /// Writers that cover every texel, such as the FFT ocean.
    Base,
    /// Regional writers that fade into the base data at their edges.
    Blend,
    /// Passes that read the finished data, such as height queries.
    Read,
}

/// Registers [`WaterLodData`] and keeps its slices centred on the [`WaterViewer`].
pub struct WaterPlugin;

impl Plugin for WaterPlugin {
    fn build(&self, app: &mut App) {
        bevy::shader::load_shader_library!(app, "lod.wesl");
        embedded_asset!(app, "surface.wesl");
        app.register_type::<WaterLodSettings>()
            .register_type::<WaterViewer>()
            .register_type::<WaterSurfaceUniform>()
            .init_resource::<WaterLodSettings>()
            .add_plugins(ExtractResourcePlugin::<WaterLodData>::default())
            .add_plugins(MaterialPlugin::<WaterSurfaceMaterial>::default())
            .add_systems(PreStartup, lod_data::create_water_lod_data)
            .add_systems(
                PostUpdate,
                (
                    lod_data::update_water_lod_slices
                        .in_set(WaterLodSystems)
                        .after(bevy::transform::TransformSystems::Propagate),
                    surface::sync_water_surface_lod.after(WaterLodSystems),
                ),
            );
        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .init_resource::<WaterLodGpu>()
                .init_resource::<WaterLodScratch>()
                .add_systems(
                    Render,
                    (
                        lod_data::prepare_water_lod_gpu,
                        blend::prepare_water_lod_scratch,
                    )
                        .in_set(RenderSystems::PrepareResources),
                )
                .configure_sets(
                    RenderGraph,
                    (
                        WaterLodGraphSystems::Base,
                        WaterLodGraphSystems::Blend,
                        WaterLodGraphSystems::Read,
                    )
                        .chain()
                        .in_set(RenderGraphSystems::Render)
                        .before(bevy::core_pipeline::schedule::camera_driver),
                );
        }
    }
}
