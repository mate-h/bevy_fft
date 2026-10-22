//! Tessendorf-style ocean: GPU spectrum into FFT buffer **C**, inverse FFT, mesh displacement.
//!
//! The egui window shows [`FftTextures::power_spectrum`] and [`FftTextures::spatial_output`]. `spatial_output`
//! uses R, G, B for slopes and height and W for wind-along chop after the ocean pack.
//!
//! **Plugin order:** add [`FftPlugin`](bevy_fft::fft::FftPlugin) before [`OceanPlugin`](bevy_fft::ocean::OceanPlugin).
//! `OceanPlugin::finish` needs FFT render resources; reversing the order panics at startup.

use bevy::{
    anti_alias::fxaa::Fxaa,
    camera::Exposure,
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

/// Edge length in meters of the ocean plane mesh.
const OCEAN_PLANE_SIZE: f32 = 256.0;

/// Vertices per axis on the ocean plane (`Plane3d` uses `subdivisions + 2` vertices).
const OCEAN_MESH_VERTEX_OPTIONS: [u32; 6] = [16, 32, 64, 128, 256, 512];

#[derive(Resource, Clone, Copy)]
struct OceanMeshSelection {
    vertices_per_edge: u32,
}

impl Default for OceanMeshSelection {
    fn default() -> Self {
        Self {
            vertices_per_edge: 256,
        }
    }
}

impl OceanMeshSelection {
    fn plane_subdivisions(self) -> u32 {
        self.vertices_per_edge.saturating_sub(2)
    }
}

#[path = "../common/screenshot.rs"]
mod screenshot;

fn main() {
    App::new()
        // Same as `examples/3d/atmosphere.rs`: deferred opaques (water there uses `ExtendedMaterial` + deferred fragment).
        .insert_resource(DefaultOpaqueRendererMethod::deferred())
        .insert_resource(ClearColor(Color::BLACK))
        .insert_resource(GlobalAmbientLight::NONE)
        .init_resource::<SunLightSettings>()
        .init_resource::<OceanMeshSelection>()
        .add_plugins((
            DefaultPlugins,
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
        .add_systems(EguiPrimaryContextPass, ocean_egui_panel)
        .add_systems(
            PostUpdate,
            sync_free_camera_with_egui_focus.after(write_egui_wants_input_system),
        )
        .run();
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

fn setup(mut commands: Commands, mut scattering_mediums: ResMut<Assets<ScatteringMedium>>) {
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
    foam: Option<egui::TextureId>,
    registered_sp: Option<AssetId<Image>>,
    registered_dp: Option<AssetId<Image>>,
    registered_foam: Option<AssetId<Image>>,
}

#[allow(clippy::type_complexity)]
fn spawn_ocean_when_ready(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<OceanSurfaceMaterial>>,
    mesh_sel: Res<OceanMeshSelection>,
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
    existing: Query<Entity, With<OceanSurfaceTag>>,
) {
    if !existing.is_empty() || q.iter().count() < CASCADE_TILES.len() {
        return;
    }

    let mesh = Plane3d::new(Vec3::Y, Vec2::splat(OCEAN_PLANE_SIZE * 0.5))
        .mesh()
        .subdivisions(mesh_sel.plane_subdivisions())
        .build();

    let mut extension = OceanSurfaceExtension::default();
    extension.settings.cascade_count = CASCADE_TILES.len() as u32;
    for (tex, foam, sim, cascade) in &q {
        extension.settings.wind_direction = sim.wind_direction;
        extension.set_cascade(
            cascade.index as usize,
            sim.tile_size,
            tex.spatial_output.clone(),
            foam.texture_a.clone(),
        );
    }

    let material = OceanSurfaceMaterial {
        base: StandardMaterial {
            base_color: Color::srgb(0.04, 0.12, 0.22),
            perceptual_roughness: 0.08,
            metallic: 0.0,
            opaque_render_method: OpaqueRendererMethod::Auto,
            ..default()
        },
        extension,
    };

    commands.spawn((
        Mesh3d(meshes.add(mesh)),
        MeshMaterial3d(materials.add(material)),
        Transform::from_xyz(0.0, 0.0, 0.0),
        OceanSurfaceTag,
    ));
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn ocean_egui_panel(
    fft_textures: Query<(&FftTextures, &OceanCascade), With<FftSource>>,
    mut contexts: EguiContexts,
    mut commands: Commands,
    mut mesh_sel: ResMut<OceanMeshSelection>,
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
    mut materials: ResMut<Assets<OceanSurfaceMaterial>>,
    surface: Query<Entity, With<OceanSurfaceTag>>,
    surface_mat: Query<&MeshMaterial3d<OceanSurfaceMaterial>, With<OceanSurfaceTag>>,
    mut sun_settings: ResMut<SunLightSettings>,
    mut sun_transform: Query<&mut Transform, With<SunLight>>,
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
    let mut mesh_amplitude = surface_ready
        .map(|m| m.extension.settings.amplitude)
        .unwrap_or(1.0);
    let mut choppiness = surface_ready
        .map(|m| m.extension.settings.choppiness)
        .unwrap_or(0.35);
    let mut foam_intensity = surface_ready
        .map(|m| m.extension.settings.foam_intensity)
        .unwrap_or(0.65);
    let mut foam_cutoff = surface_ready
        .map(|m| m.extension.settings.foam_cutoff)
        .unwrap_or(0.88);
    let mut foam_falloff = surface_ready
        .map(|m| m.extension.settings.foam_falloff)
        .unwrap_or(0.22);
    let mut foam_trail_decay = surface_ready
        .map(|m| m.extension.settings.foam_trail_decay)
        .unwrap_or(0.94);

    let sp_tex = cache.spectrum;
    let dp_tex = cache.displacement;

    if let Some(m) = surface_ready {
        let fm_id = m.extension.foam(0).id();
        if cache.registered_foam != Some(fm_id) {
            cache.foam = Some(contexts.add_image(EguiTextureHandle::Weak(fm_id)));
            cache.registered_foam = Some(fm_id);
        }
    }
    let fm_tex = cache.foam;

    let Ok(ctx) = contexts.ctx_mut() else {
        return;
    };

    let size_idx = OCEAN_GRID_SIZES
        .iter()
        .position(|&n| n == s.texture_size)
        .unwrap_or(2);
    let mut new_idx = size_idx;

    let mesh_idx = OCEAN_MESH_VERTEX_OPTIONS
        .iter()
        .position(|&n| n == mesh_sel.vertices_per_edge)
        .unwrap_or(3);
    let mut new_mesh_idx = mesh_idx;

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

        ui.add_space(4.0);
        ui.label("Mesh vertices");
        egui::ComboBox::from_id_salt("ocean_mesh_subdivisions")
            .selected_text(format!(
                "{} × {}",
                OCEAN_MESH_VERTEX_OPTIONS[new_mesh_idx], OCEAN_MESH_VERTEX_OPTIONS[new_mesh_idx]
            ))
            .show_ui(ui, |ui| {
                for (i, &v) in OCEAN_MESH_VERTEX_OPTIONS.iter().enumerate() {
                    ui.selectable_value(&mut new_mesh_idx, i, format!("{v} × {v}"));
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

        let new_mesh_vertices = OCEAN_MESH_VERTEX_OPTIONS[new_mesh_idx];
        if new_mesh_vertices != mesh_sel.vertices_per_edge {
            mesh_sel.vertices_per_edge = new_mesh_vertices;
            for entity in &surface {
                commands.entity(entity).despawn();
            }
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
        ui.add(egui::Slider::new(&mut mesh_amplitude, 0.0..=4.0).text("Mesh amplitude"));
        ui.add(egui::Slider::new(&mut choppiness, 0.0..=2.0).text("Choppiness"));
        ui.add(egui::Slider::new(&mut foam_intensity, 0.0..=1.5).text("Foam intensity"));
        ui.add(egui::Slider::new(&mut foam_cutoff, 0.2..=1.2).text("Foam cutoff"));
        ui.add(egui::Slider::new(&mut foam_falloff, 0.02..=0.8).text("Foam falloff"));
        ui.add(egui::Slider::new(&mut foam_trail_decay, 0.85..=0.999).text("Foam trail decay"));
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
            if let Some(id) = fm_tex {
                ui.vertical(|ui| {
                    ui.label("foam mask");
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

    let Ok(handle) = surface_mat.single() else {
        return;
    };
    let Some(mut mat) = materials.get_mut(handle) else {
        return;
    };
    mat.extension.settings.amplitude = mesh_amplitude;
    mat.extension.settings.choppiness = choppiness;
    mat.extension.settings.wind_direction = s.wind_direction;
    mat.extension.settings.foam_intensity = foam_intensity;
    mat.extension.settings.foam_cutoff = foam_cutoff;
    mat.extension.settings.foam_falloff = foam_falloff;
    mat.extension.settings.foam_trail_decay = foam_trail_decay;

    let Ok(mut tf) = sun_transform.single_mut() else {
        return;
    };
    *tf = Transform::from_xyz(0.0, sun_settings.elevation, -1.0).looking_at(Vec3::ZERO, Vec3::Y);
}
