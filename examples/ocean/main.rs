//! Tessendorf-style ocean: GPU spectrum into FFT buffer **C**, inverse FFT, mesh displacement.
//!
//! The egui window shows [`FftTextures::power_spectrum`] and [`FftTextures::spatial_output`]. `spatial_output`
//! uses R, G, B for slopes and height and W for wind-along chop after the ocean pack.
//!
//! **Plugin order:** add [`FftPlugin`](bevy_fft::fft::FftPlugin) before [`OceanPlugin`](bevy_fft::ocean::OceanPlugin).
//! `OceanPlugin::finish` needs FFT render resources; reversing the order panics at startup.

use bevy::{
    anti_alias::fxaa::Fxaa,
    camera::{Exposure, primitives::Aabb, visibility::NoFrustumCulling},
    camera_controller::free_camera::{FreeCamera, FreeCameraPlugin, FreeCameraState},
    core_pipeline::tonemapping::Tonemapping,
    light::{
        Atmosphere, AtmosphereEnvironmentMapLight, CascadeShadowConfigBuilder, GlobalAmbientLight,
        VolumetricLight, atmosphere::ScatteringMedium, light_consts::lux,
    },
    material::OpaqueRendererMethod,
    mesh::Meshable,
    pbr::{
        AtmosphereSettings, DefaultOpaqueRendererMethod, ScreenSpaceReflections, StandardMaterial,
    },
    post_process::bloom::Bloom,
    prelude::*,
};
use bevy_egui::{
    EguiContexts, EguiPlugin, EguiPrimaryContextPass, EguiTextureHandle, egui,
    input::{EguiWantsInput, write_egui_wants_input_system},
};
use bevy_fft::prelude::*;

/// Grid edge lengths for the ocean FFT (powers of two).
const OCEAN_GRID_SIZES: [u32; 4] = [128, 256, 512, 1024];

/// World-space tile edge in meters for each FFT cascade, largest first.
const CASCADE_TILES: [f32; 3] = [256.0, 48.0, 9.0];

/// A cascade hands waves to the next smaller one at this many wavelengths per smaller tile.
/// Keeping a few periods inside the smaller tile avoids its coarse lowest bins.
const CASCADE_HANDOFF_WAVES: f32 = 6.0;

#[path = "../common/screenshot.rs"]
mod screenshot;

fn main() {
    App::new()
        // Same as `examples/3d/atmosphere.rs`: deferred opaques (water there uses `ExtendedMaterial` + deferred fragment).
        .insert_resource(DefaultOpaqueRendererMethod::deferred())
        .insert_resource(ClearColor(Color::BLACK))
        .insert_resource(GlobalAmbientLight::NONE)
        .init_resource::<SunLightSettings>()
        .add_plugins((
            screenshot::default_plugins(),
            FreeCameraPlugin,
            FftPlugin,
            OceanPlugin,
            EguiPlugin::default(),
            screenshot::ScreenshotOnEnvPlugin,
        ))
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            spawn_ocean_when_ready.after(FftSystemSet::PrepareTextures),
        )
        .add_systems(Update, float_buoys)
        .add_systems(EguiPrimaryContextPass, (ocean_egui_panel, water_mesh_panel))
        .add_systems(
            PostUpdate,
            sync_free_camera_with_egui_focus.after(write_egui_wants_input_system),
        )
        .run();
}

/// A marker that rides the surface through [`OceanQueries`].
#[derive(Component)]
struct Buoy {
    xz: Vec2,
}

/// World `xz` positions of the demo buoys, in meters.
const BUOY_POSITIONS: [Vec2; 5] = [
    Vec2::new(0.0, 0.0),
    Vec2::new(6.0, -4.0),
    Vec2::new(-7.0, -2.0),
    Vec2::new(3.0, 8.0),
    Vec2::new(-4.0, 10.0),
];

fn spawn_buoys(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
) {
    let mesh = meshes.add(Sphere::new(0.4).mesh().uv(24, 12));
    let material = materials.add(StandardMaterial {
        base_color: Color::srgb(0.95, 0.35, 0.1),
        perceptual_roughness: 0.5,
        ..default()
    });
    for xz in BUOY_POSITIONS {
        commands.spawn((
            Mesh3d(mesh.clone()),
            MeshMaterial3d(material.clone()),
            Transform::from_xyz(xz.x, 0.0, xz.y),
            Buoy { xz },
        ));
    }
}

/// Places each buoy on the queried surface height, extrapolated to now, and tilts it to the
/// surface normal.
fn float_buoys(
    mut queries: ResMut<OceanQueries>,
    mut buoys: Query<(&Buoy, &mut Transform)>,
    time: Res<Time>,
) {
    if queries.points().is_empty() {
        queries.set_points(BUOY_POSITIONS);
    }
    let Some(results) = queries.results() else {
        return;
    };
    for (buoy, mut transform) in &mut buoys {
        let Some(i) = results.points.iter().position(|p| *p == buoy.xz) else {
            continue;
        };
        let sample = results.samples[i];
        // Extrapolate over the readback latency so the buoy keeps up with the waves.
        let age = time.elapsed_secs() - results.time;
        transform.translation = Vec3::new(buoy.xz.x, sample.height_after(age), buoy.xz.y);
        transform.rotation = Quat::from_rotation_arc(Vec3::Y, sample.normal());
    }
}

fn sync_free_camera_with_egui_focus(
    mut state: Query<&mut FreeCameraState, With<FreeCamera>>,
    egui_wants: Res<EguiWantsInput>,
) {
    let Ok(mut camera_state) = state.single_mut() else {
        return;
    };
    camera_state.enabled = !egui_wants.wants_any_input();
}

/// Y component of [`SunLight`] position before `looking_at` (0 is horizon along +X, 1 raises the light).
#[derive(Resource)]
struct SunLightSettings {
    elevation: f32,
}

impl Default for SunLightSettings {
    fn default() -> Self {
        Self { elevation: 0.1 }
    }
}

#[derive(Component)]
struct SunLight;

/// Wavenumber band in rad/m kept by cascade `index`. Bands touch but never overlap, so each
/// wave is counted once across the cascades.
fn cascade_band(index: usize) -> (f32, f32) {
    let handoff = |i: usize| CASCADE_HANDOFF_WAVES * std::f32::consts::TAU / CASCADE_TILES[i];
    let k_min = if index == 0 { 0.0 } else { handoff(index) };
    let k_max = if index + 1 < CASCADE_TILES.len() {
        handoff(index + 1)
    } else {
        f32::INFINITY
    };
    (k_min, k_max)
}

fn setup(
    mut commands: Commands,
    mut scattering_mediums: ResMut<Assets<ScatteringMedium>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut standard_materials: ResMut<Assets<StandardMaterial>>,
) {
    spawn_buoys(&mut commands, &mut meshes, &mut standard_materials);
    let n = 256u32;
    for (index, &tile) in CASCADE_TILES.iter().enumerate() {
        let (band_k_min, band_k_max) = cascade_band(index);
        commands.spawn((
            FftSource::square_inverse_only(n),
            OceanSimSettings {
                texture_size: n,
                tile_size: tile,
                band_k_min,
                band_k_max,
                ..default()
            },
            OceanCascade {
                index: index as u32,
            },
            OceanH0Uniform::default(),
            OceanDynamicUniform::default(),
            OceanFoamUniform::default(),
            OceanFoamPhase(0),
        ));
    }

    // Match `examples/3d/atmosphere.rs`: no transform on the atmosphere entity; the on_add hook
    // places the planet center below the origin so the scene sits near the surface.
    let earth_medium = scattering_mediums.add(ScatteringMedium::earth(256, 256));
    commands.spawn(Atmosphere::earth(earth_medium));
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 8.0, 24.0).looking_at(Vec3::ZERO, Vec3::Y),
        FreeCamera::default(),
        AtmosphereSettings::default(),
        Exposure { ev100: 12.0 },
        Tonemapping::AcesFitted,
        Bloom::NATURAL,
        AtmosphereEnvironmentMapLight::default(),
        Msaa::Off,
        Fxaa::default(),
        ScreenSpaceReflections::default(),
    ));

    let cascade_shadow_config = CascadeShadowConfigBuilder {
        first_cascade_far_bound: 0.3,
        maximum_distance: 15.0,
        ..default()
    }
    .build();

    commands.spawn((
        DirectionalLight {
            shadow_maps_enabled: true,
            illuminance: lux::RAW_SUNLIGHT,
            ..default()
        },
        SunLight,
        VolumetricLight,
        Transform::from_xyz(0.0, 0.1, -1.0).looking_at(Vec3::ZERO, Vec3::Y),
        cascade_shadow_config,
    ));
}

#[derive(Default)]
struct OceanFftPreviewEgui {
    spectrum: Option<egui::TextureId>,
    displacement: Option<egui::TextureId>,
    registered_sp: Option<AssetId<Image>>,
    registered_dp: Option<AssetId<Image>>,
}

#[allow(clippy::type_complexity)]
fn spawn_ocean_when_ready(
    mut commands: Commands,
    mut materials: ResMut<Assets<WaterSurfaceMaterial>>,
    q: Query<
        (
            &FftTextures,
            &OceanFoamMask,
            &OceanSimSettings,
            &OceanCascade,
        ),
        (
            With<FftSource>,
            With<OceanH0Uniform>,
            With<OceanDynamicUniform>,
            With<OceanFoamUniform>,
            With<OceanFoamPhase>,
        ),
    >,
    existing: Query<Entity, With<WaterSurfaceTag>>,
) {
    if !existing.is_empty() || q.iter().count() < CASCADE_TILES.len() {
        return;
    }

    let material = WaterSurfaceMaterial {
        base: StandardMaterial {
            base_color: Color::srgb(0.04, 0.12, 0.22),
            perceptual_roughness: 0.08,
            metallic: 0.0,
            opaque_render_method: OpaqueRendererMethod::Auto,
            ..default()
        },
        extension: WaterSurfaceExtension::default(),
    };

    let mut surface = commands.spawn((MeshMaterial3d(materials.add(material)), WaterSurfaceTag));
    if std::env::var("BEVY_FFT_OCEAN_MESH").as_deref() == Ok("quadtree") {
        surface.insert(WaterQuadtree::default());
    } else {
        surface.insert(OceanClipmap::default());
    }
}

/// Switches the surface between the clipmap and the quadtree and shows how many vertices each draws.
#[allow(clippy::type_complexity)]
fn water_mesh_panel(
    mut contexts: EguiContexts,
    mut commands: Commands,
    mut surfaces: Query<
        (
            Entity,
            Option<&mut OceanClipmap>,
            Option<&WaterQuadtreeStats>,
            Option<&Mesh3d>,
        ),
        With<WaterSurfaceTag>,
    >,
    meshes: Res<Assets<Mesh>>,
) {
    let Ok((entity, clipmap, stats, mesh)) = surfaces.single_mut() else {
        return;
    };
    let Ok(ctx) = contexts.ctx_mut() else {
        return;
    };
    egui::Window::new("Water mesh")
        .default_pos([12.0, 520.0])
        .show(ctx, |ui| {
            let quadtree = stats.is_some();
            let mut pick_quadtree = quadtree;
            ui.horizontal(|ui| {
                ui.selectable_value(&mut pick_quadtree, false, "Clipmap");
                ui.selectable_value(&mut pick_quadtree, true, "Quadtree");
            });
            if let Some(mut clipmap) = clipmap {
                let vertices = mesh
                    .and_then(|m| meshes.get(&m.0))
                    .map_or(0, |m| m.count_vertices());
                ui.label(format!("{vertices} vertices, no culling"));
                let mut levels = clipmap.levels;
                ui.add(egui::Slider::new(&mut levels, 3..=12).text("Clipmap levels"));
                if levels != clipmap.levels {
                    clipmap.levels = levels;
                }
            }
            if let Some(stats) = stats {
                ui.label(format!(
                    "{} vertices in {} tiles",
                    stats.vertices, stats.tiles
                ));
                if stats.dropped > 0 {
                    ui.label(format!("{} tiles did not fit the pool", stats.dropped));
                }
            }
            if pick_quadtree != quadtree {
                let mut e = commands.entity(entity);
                e.remove::<(
                    Mesh3d,
                    WaterQuadtree,
                    WaterQuadtreeBuffer,
                    WaterQuadtreeStats,
                    OceanClipmap,
                    NoFrustumCulling,
                    Aabb,
                )>()
                .insert(Transform::IDENTITY);
                if pick_quadtree {
                    e.insert(WaterQuadtree::default());
                } else {
                    e.insert(OceanClipmap::default());
                }
            }
        });
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn ocean_egui_panel(
    fft_textures: Query<(&FftTextures, &OceanCascade), With<FftSource>>,
    mut contexts: EguiContexts,
    mut commands: Commands,
    mut cache: Local<OceanFftPreviewEgui>,
    mut smoothed_frame_ms: Local<Option<f32>>,
    time: Res<Time>,
    mut sim: Query<
        (Entity, &mut FftSource, &mut OceanSimSettings, &OceanCascade),
        (
            With<OceanH0Uniform>,
            With<OceanDynamicUniform>,
            With<OceanFoamUniform>,
            With<OceanFoamPhase>,
        ),
    >,
    mut materials: ResMut<Assets<WaterSurfaceMaterial>>,
    mut ocean_surface: ResMut<OceanSurfaceSettings>,
    surface: Query<Entity, With<WaterSurfaceTag>>,
    surface_mat: Query<&MeshMaterial3d<WaterSurfaceMaterial>, With<WaterSurfaceTag>>,
    mut sun_settings: ResMut<SunLightSettings>,
    mut sun_transform: Query<&mut Transform, With<SunLight>>,
    queries: Res<OceanQueries>,
) {
    // The panel edits cascade 0. Shared fields are copied to the other cascades afterwards, while
    // each cascade keeps its own tile size and wavenumber band.
    let mut cascades: Vec<_> = sim.iter_mut().collect();
    cascades.sort_by_key(|c| c.3.index);
    let Some((first, rest)) = cascades.split_first_mut() else {
        return;
    };
    let sim_entity = first.0;
    let fft_source = &mut *first.1;
    let s = &mut *first.2;

    if let Some((tex, _)) = fft_textures.iter().find(|(_, c)| c.index == 0) {
        let sp_id = tex.power_spectrum.id();
        let dp_id = tex.spatial_output.id();
        if cache.registered_sp != Some(sp_id) {
            cache.spectrum = Some(contexts.add_image(EguiTextureHandle::Weak(sp_id)));
            cache.registered_sp = Some(sp_id);
        }
        if cache.registered_dp != Some(dp_id) {
            cache.displacement = Some(contexts.add_image(EguiTextureHandle::Weak(dp_id)));
            cache.registered_dp = Some(dp_id);
        }
    }
    let surface_ready = surface_mat.single().ok().and_then(|m| materials.get(&m.0));
    let mut shaping = *ocean_surface;
    let mut foam_intensity = surface_ready
        .map(|m| m.extension.settings.foam_intensity)
        .unwrap_or(0.65);

    let sp_tex = cache.spectrum;
    let dp_tex = cache.displacement;

    let Ok(ctx) = contexts.ctx_mut() else {
        return;
    };

    let size_idx = OCEAN_GRID_SIZES
        .iter()
        .position(|&n| n == s.texture_size)
        .unwrap_or(2);
    let mut new_idx = size_idx;

    let dt = time.delta_secs();
    let inst_ms = dt * 1000.0;
    let tau = 0.25_f32;
    let alpha = 1.0 - (-dt / tau).exp();
    let smooth_ms = smoothed_frame_ms.map_or(inst_ms, |s| s + (inst_ms - s) * alpha);
    *smoothed_frame_ms = Some(smooth_ms);

    egui::Window::new("Ocean").show(ctx, |ui| {
        if surface_ready.is_none() {
            ui.label("Spawning ocean mesh after GPU init…");
        }
        ui.label(egui::RichText::new(format!("{smooth_ms:.2} ms/frame")).weak());
        if let Some(results) = queries.results() {
            let latency = queries.frame().wrapping_sub(results.frame);
            ui.label(
                egui::RichText::new(format!(
                    "{} buoy queries, {latency} frames behind",
                    results.samples.len()
                ))
                .weak(),
            );
        }
        ui.add_space(2.0);
        ui.label("FFT resolution");
        egui::ComboBox::from_id_salt("ocean_fft_resolution")
            .selected_text(format!(
                "{} × {}",
                OCEAN_GRID_SIZES[new_idx], OCEAN_GRID_SIZES[new_idx]
            ))
            .show_ui(ui, |ui| {
                for (i, n) in OCEAN_GRID_SIZES.iter().enumerate() {
                    ui.selectable_value(&mut new_idx, i, format!("{n} × {n}"));
                }
            });

        let new_n = OCEAN_GRID_SIZES[new_idx];
        if new_n != s.texture_size {
            commands
                .entity(sim_entity)
                .remove::<FftTextures>()
                .remove::<OceanH0Image>()
                .remove::<OceanFoamMask>()
                .insert(OceanFoamPhase(0));
            *fft_source = FftSource::square_inverse_only(new_n);
            s.texture_size = new_n;
            s.h0_serial = s.h0_serial.wrapping_add(1);
            for (entity, other_source, other, _) in rest.iter_mut() {
                commands
                    .entity(*entity)
                    .remove::<FftTextures>()
                    .remove::<OceanH0Image>()
                    .remove::<OceanFoamMask>()
                    .insert(OceanFoamPhase(0));
                **other_source = FftSource::square_inverse_only(new_n);
                other.texture_size = new_n;
            }
            for entity in &surface {
                commands.entity(entity).despawn();
            }
            *cache = OceanFftPreviewEgui::default();
        }

        ui.add(egui::Slider::new(&mut s.wind_speed, 0.0..=20.0).text("Wind speed"));
        ui.add(
            egui::Slider::new(
                &mut s.wind_direction,
                -std::f32::consts::PI..=std::f32::consts::PI,
            )
            .text("Wind direction"),
        );
        ui.add(egui::Slider::new(&mut s.peak_enhancement, 1.0..=8.0).text("Peak enhancement"));
        ui.add(egui::Slider::new(&mut s.directional_spread, 0.0..=16.0).text("Spread"));
        ui.add(egui::Slider::new(&mut s.swell, 0.0..=1.0).text("Swell"));
        ui.add(
            egui::Slider::new(&mut s.depth, 0.5..=1000.0)
                .logarithmic(true)
                .text("Depth (m)"),
        );
        ui.add(egui::Slider::new(&mut s.small_wave_cutoff, 0.000..=0.1).text("Small wave cutoff"));
        ui.add(egui::Slider::new(&mut s.amplitude_scale, 0.0..=4.0).text("H0 amplitude scale"));
        ui.add(egui::Slider::new(&mut s.time_scale, 0.0..=3.0).text("Time scale"));
        ui.add(egui::Slider::new(&mut shaping.amplitude, 0.0..=4.0).text("Mesh amplitude"));
        ui.add(egui::Slider::new(&mut shaping.choppiness, 0.0..=2.0).text("Choppiness"));
        ui.add(egui::Slider::new(&mut foam_intensity, 0.0..=1.5).text("Foam intensity"));
        ui.add(egui::Slider::new(&mut shaping.foam_cutoff, 0.2..=1.2).text("Foam cutoff"));
        ui.add(egui::Slider::new(&mut shaping.foam_falloff, 0.02..=0.8).text("Foam falloff"));
        ui.add(
            egui::Slider::new(&mut shaping.foam_trail_decay, 0.85..=0.999).text("Foam trail decay"),
        );
        ui.add(egui::Slider::new(&mut sun_settings.elevation, 0.0..=1.0).text("Sun light height"));
        if ui.button("Regenerate spectrum").clicked() {
            s.h0_serial = s.h0_serial.wrapping_add(1);
        }

        ui.separator();
        let size = egui::vec2(64.0, 64.0);
        ui.horizontal(|ui| {
            if let Some(id) = sp_tex {
                ui.vertical(|ui| {
                    ui.label("power spectrum");
                    ui.add(egui::Image::new(egui::load::SizedTexture::new(id, size)));
                });
            }
            if let Some(id) = dp_tex {
                ui.vertical(|ui| {
                    ui.label("spatial output");
                    ui.add(egui::Image::new(egui::load::SizedTexture::new(id, size)));
                });
            }
        });
    });

    for (_, _, other, _) in rest.iter_mut() {
        other.wind_speed = s.wind_speed;
        other.wind_direction = s.wind_direction;
        other.peak_enhancement = s.peak_enhancement;
        other.directional_spread = s.directional_spread;
        other.swell = s.swell;
        other.depth = s.depth;
        other.small_wave_cutoff = s.small_wave_cutoff;
        other.amplitude_scale = s.amplitude_scale;
        other.time_scale = s.time_scale;
        other.gravity = s.gravity;
        other.h0_serial = s.h0_serial;
    }

    ocean_surface.set_if_neq(shaping);
    let Ok(handle) = surface_mat.single() else {
        return;
    };
    if surface_ready.is_some_and(|m| m.extension.settings.foam_intensity != foam_intensity)
        && let Some(mut mat) = materials.get_mut(handle)
    {
        mat.extension.settings.foam_intensity = foam_intensity;
    }

    let Ok(mut tf) = sun_transform.single_mut() else {
        return;
    };
    *tf = Transform::from_xyz(0.0, sun_settings.elevation, -1.0).looking_at(Vec3::ZERO, Vec3::Y);
}
