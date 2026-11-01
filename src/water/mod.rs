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
//! [`WaterQuadtree`] provides a view-dependent mesh for it, with tiles that grow with distance and
//! are culled to the camera frustum.
//!
//! Writers run in the render graph in the order of [`WaterLodGraphSystems`]. A base writer such
//! as the ocean covers every texel. Regional solvers such as shallow water then blend over it
//! through [`WaterLodScratch`]. The foam sim then carries foam over from the previous frame, and
//! readers such as height queries run last.
//!
//! [`WaterUnderwater`] on the viewer adds a waterline mask, [`WaterWaterline`], that tells which
//! pixels look out from under the water, and fills them with the medium of [`WaterMedium`].
//!
//! Sampling helpers for WESL live in `bevy_fft::water::lod`.

mod blend;
mod caustics;
mod foam;
mod foam_texture;
mod lod_data;
mod medium;
mod quadtree;
mod quadtree_gpu;
mod scene_depth;
mod surface;
mod underwater;

pub use blend::WaterLodScratch;
pub use caustics::{
    WATER_CAUSTICS_FORMAT, WATER_CAUSTICS_SHADER, WaterCaustics, WaterCausticsSettings,
};
pub use foam::{
    WATER_FOAM_SHADER, WATER_FOAM_STATE_FORMAT, WaterFoamFrame, WaterFoamParams, WaterFoamSettings,
};
pub use foam_texture::{WATER_FOAM_TEXTURE_SIZE, WaterFoamTexture, foam_texture_image};
pub use lod_data::{
    WATER_LOD_DRIFT_FORMAT, WATER_LOD_MAX_SLICES, WATER_LOD_SURFACE_FORMAT, WaterLodData,
    WaterLodGpu, WaterLodSettings, WaterLodSlices, WaterViewer,
};
pub use medium::{WATER_MEDIUM_SHADER, WaterMedium};
pub use quadtree::{
    WaterQuadtree, WaterQuadtreeBuffer, WaterQuadtreeFallback, WaterQuadtreeStats,
    WaterQuadtreeTile,
};
pub use scene_depth::WaterTransmission;
pub use surface::{
    JerlovWaterType, WATER_SURFACE_SHADER, WaterMediumUniform, WaterOpticsUniform,
    WaterSurfaceExtension, WaterSurfaceMaterial, WaterSurfaceTag, WaterSurfaceUniform,
};
pub use underwater::{
    WATER_WATERLINE_FORMAT, WATER_WATERLINE_SHADER, WaterUnderwater, WaterWaterline,
};

use bevy::{
    asset::embedded_asset,
    prelude::*,
    render::{
        GpuResourceAppExt, Render, RenderApp, RenderSystems,
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
    /// Passes that evolve state on the finished data, such as the foam sim.
    Simulate,
    /// Passes that read the finished data, such as height queries.
    Read,
}

/// Registers [`WaterLodData`] and keeps its slices centred on the [`WaterViewer`].
pub struct WaterPlugin;

impl Plugin for WaterPlugin {
    fn build(&self, app: &mut App) {
        scene_depth::configure(app);
        underwater::configure(app);
        medium::configure(app);
        caustics::configure(app);
        bevy::shader::load_shader_library!(app, "lod.wesl");
        bevy::shader::load_shader_library!(app, "optics.wesl");
        bevy::shader::load_shader_library!(app, "breaking.wesl");
        bevy::shader::load_shader_library!(app, "underside.wesl");
        embedded_asset!(app, "surface.wesl");
        embedded_asset!(app, "quadtree.wesl");
        embedded_asset!(app, "foam.wesl");
        app.register_type::<WaterLodSettings>()
            .register_type::<WaterViewer>()
            .register_type::<WaterSurfaceUniform>()
            .register_type::<WaterOpticsUniform>()
            .register_type::<WaterMediumUniform>()
            .register_type::<JerlovWaterType>()
            .register_type::<WaterFoamSettings>()
            .init_resource::<WaterFoamSettings>()
            .init_resource::<WaterFoamFrame>()
            .add_plugins(ExtractResourcePlugin::<WaterFoamFrame>::default())
            .register_type::<WaterQuadtree>()
            .register_type::<WaterQuadtreeStats>()
            .init_resource::<WaterLodSettings>()
            .init_resource::<quadtree_gpu::WaterQuadtreeFrame>()
            .add_plugins(ExtractResourcePlugin::<quadtree_gpu::WaterQuadtreeFrame>::default())
            .add_plugins(ExtractResourcePlugin::<WaterLodData>::default())
            .add_plugins(MaterialPlugin::<WaterSurfaceMaterial>::default())
            .add_observer(quadtree::read_quadtree_stats)
            .add_systems(
                PreStartup,
                (
                    lod_data::create_water_lod_data,
                    quadtree::create_water_quadtree_fallback,
                    foam_texture::create_water_foam_texture,
                ),
            )
            .add_systems(Update, quadtree::build_water_quadtrees)
            .add_systems(
                PostUpdate,
                (
                    lod_data::update_water_lod_slices
                        .in_set(WaterLodSystems)
                        .after(bevy::transform::TransformSystems::Propagate),
                    quadtree::select_water_quadtrees
                        .after(bevy::camera::visibility::VisibilitySystems::UpdateFrusta),
                    surface::sync_water_surface_lod.after(WaterLodSystems),
                    foam::gather_water_foam.after(WaterLodSystems),
                ),
            );
        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .init_resource::<quadtree_gpu::GpuQuadtrees>()
                .init_gpu_resource::<quadtree_gpu::QuadtreePipeline>()
                .init_resource::<WaterLodGpu>()
                .init_resource::<WaterLodScratch>()
                .init_resource::<foam::WaterFoamSim>()
                .add_systems(
                    Render,
                    (
                        lod_data::prepare_water_lod_gpu,
                        blend::prepare_water_lod_scratch,
                        quadtree_gpu::prepare_quadtrees,
                    )
                        .in_set(RenderSystems::PrepareResources),
                )
                .add_systems(
                    Render,
                    foam::prepare_water_foam.in_set(RenderSystems::PrepareBindGroups),
                )
                .configure_sets(
                    RenderGraph,
                    (
                        WaterLodGraphSystems::Base,
                        WaterLodGraphSystems::Blend,
                        WaterLodGraphSystems::Simulate,
                        WaterLodGraphSystems::Read,
                    )
                        .chain()
                        .in_set(RenderGraphSystems::Render)
                        .before(bevy::core_pipeline::schedule::camera_driver),
                )
                .add_systems(
                    RenderGraph,
                    foam::run_water_foam.in_set(WaterLodGraphSystems::Simulate),
                )
                .add_systems(
                    RenderGraph,
                    quadtree_gpu::select_quadtrees
                        .in_set(RenderGraphSystems::Render)
                        .before(bevy::core_pipeline::schedule::camera_driver),
                );
            quadtree_gpu::configure_draws(render_app);
        }
    }

    fn finish(&self, app: &mut App) {
        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app.init_resource::<foam::WaterFoamPipeline>();
        }
        underwater::finish(app);
        caustics::finish(app);
        quadtree_gpu::install_draws(app);
    }
}
