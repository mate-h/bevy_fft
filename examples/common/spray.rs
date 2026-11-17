//! Spray from breaking crests for the water examples: adds the water particle pool with the crest
//! emitter on, and a panel to switch it and set its rate. `BEVY_FFT_SPRAY=0` starts it off.

use bevy::prelude::*;
use bevy_egui::{EguiContexts, EguiPrimaryContextPass, egui};
use bevy_fft::particles::{
    WaterCrestSpray, WaterParticleSettings, WaterParticleStats, WaterParticlesPlugin,
};

/// Particles per m² per second at full breaking.
const DEFAULT_RATE: f32 = 150.0;

pub struct CrestSprayPlugin;

impl Plugin for CrestSprayPlugin {
    fn build(&self, app: &mut App) {
        let on = std::env::var("BEVY_FFT_SPRAY").map_or(true, |v| v != "0");
        app.insert_resource(WaterParticleSettings {
            crest_spray: WaterCrestSpray {
                rate: if on { DEFAULT_RATE } else { 0.0 },
                ..default()
            },
            ..default()
        })
        .insert_resource(SprayRate(DEFAULT_RATE))
        .add_plugins(WaterParticlesPlugin)
        .add_systems(EguiPrimaryContextPass, spray_panel);
    }
}

/// The rate the panel switches back to.
#[derive(Resource)]
struct SprayRate(f32);

fn spray_panel(
    mut contexts: EguiContexts,
    mut settings: ResMut<WaterParticleSettings>,
    mut rate: ResMut<SprayRate>,
    stats: Res<WaterParticleStats>,
) {
    let Ok(ctx) = contexts.ctx_mut() else {
        return;
    };
    egui::Window::new("Spray")
        .default_open(false)
        .default_pos([320.0, 12.0])
        .show(ctx, |ui| {
            let mut on = settings.crest_spray.rate > 0.0;
            if ui.checkbox(&mut on, "Spray from breaking crests").changed() {
                settings.crest_spray.rate = if on { rate.0 } else { 0.0 };
            }
            if ui
                .add(
                    egui::Slider::new(&mut rate.0, 10.0..=2000.0)
                        .logarithmic(true)
                        .text("per m² s"),
                )
                .changed()
                && on
            {
                settings.crest_spray.rate = rate.0;
            }
            ui.checkbox(
                &mut settings.crest_spray.debug_potential,
                "Show where spray spawns",
            );
            ui.checkbox(&mut settings.read_counters, "Read counters");
            if settings.read_counters {
                ui.label(format!(
                    "{} alive, {} from crests last frame",
                    stats.alive, stats.crest_spray
                ));
            }
        });
}
