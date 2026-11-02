//! The beach sun never moves, so the sky's environment light only needs filtering once.
//!
//! `AtmosphereEnvironmentMapLight` renders the sky into a cubemap and filters it into the
//! diffuse and specular maps every frame. Once the filter has run for a few frames, this plugin
//! keeps the filtered `EnvironmentMapLight` on the camera and removes the atmosphere component,
//! which stops the per-frame sky render and filtering.

use std::sync::{
    Arc,
    atomic::{AtomicU32, Ordering},
};

use bevy::{
    light::{AtmosphereEnvironmentMapLight, EnvironmentMapLight},
    pbr::generate::{
        GeneratorBindGroups, GeneratorPipelines, prepare_generated_environment_map_bind_groups,
    },
    prelude::*,
    render::{Render, RenderApp, RenderSystems, render_resource::PipelineCache},
};

/// Frames the filter must have run before its maps are kept.
const FILTERED_FRAMES: u32 = 30;

/// Frames in which the render world could filter the environment map.
#[derive(Resource, Clone, Default)]
struct FilteredFrames(Arc<AtomicU32>);

pub struct StaticSkyLightPlugin;

impl Plugin for StaticSkyLightPlugin {
    fn build(&self, app: &mut App) {
        let frames = FilteredFrames::default();
        app.insert_resource(frames.clone())
            .add_systems(Update, keep_sky_light);
        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app.insert_resource(frames).add_systems(
                Render,
                count_filtered_frames
                    .in_set(RenderSystems::PrepareBindGroups)
                    .after(prepare_generated_environment_map_bind_groups),
            );
        }
    }
}

fn count_filtered_frames(
    frames: Res<FilteredFrames>,
    cache: Res<PipelineCache>,
    pipelines: Option<Res<GeneratorPipelines>>,
    maps: Query<(), With<GeneratorBindGroups>>,
) {
    let Some(pipelines) = pipelines else {
        return;
    };
    let ready = [pipelines.copy, pipelines.radiance, pipelines.irradiance]
        .into_iter()
        .all(|id| cache.get_compute_pipeline(id).is_some());
    if ready && !maps.is_empty() {
        frames.0.fetch_add(1, Ordering::Relaxed);
    }
}

fn keep_sky_light(
    mut commands: Commands,
    frames: Res<FilteredFrames>,
    cameras: Query<(Entity, &EnvironmentMapLight), With<AtmosphereEnvironmentMapLight>>,
) {
    if frames.0.load(Ordering::Relaxed) < FILTERED_FRAMES {
        return;
    }
    for (entity, light) in &cameras {
        info!("Keeping the filtered sky light and stopping its regeneration");
        // Removing the atmosphere component also removes the light it created, so the kept
        // light goes back on afterwards.
        let light = light.clone();
        commands
            .entity(entity)
            .remove::<AtmosphereEnvironmentMapLight>();
        commands.queue(move |world: &mut World| {
            if let Ok(mut camera) = world.get_entity_mut(entity) {
                camera.insert(light);
            }
        });
    }
}
