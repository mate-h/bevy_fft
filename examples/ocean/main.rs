//! Tessendorf-style ocean: GPU spectrum into FFT buffer **C**, inverse FFT, mesh displacement.
//!
//! The egui window shows [`FftTextures::power_spectrum`] and [`FftTextures::spatial_output`]. `spatial_output`
//! uses R, G, B for slopes and height and A for wind-along chop after the ocean pack.

use bevy::{
    anti_alias::fxaa::Fxaa,
    camera::Exposure,
    camera_controller::free_camera::{FreeCamera, FreeCameraPlugin, FreeCameraState},
    core_pipeline::tonemapping::Tonemapping,
    light::{
        AtmosphereEnvironmentMapLight, CascadeShadowConfigBuilder, GlobalAmbientLight,
        light_consts::lux,
    },
    mesh::Meshable,
    pbr::{
        Atmosphere, AtmosphereSettings, DefaultOpaqueRendererMethod, OpaqueRendererMethod,
        ScatteringMedium, ScreenSpaceReflections, StandardMaterial,
    },
    post_process::bloom::Bloom,
    prelude::*,
};
use bevy_egui::{
    EguiContexts, EguiPlugin, EguiPrimaryContextPass, EguiTextureHandle, egui,
    input::{EguiWantsInput, write_egui_wants_input_system},
};
use bevy_fft::fft::prelude::*;
use bevy_fft::ocean::{
    OceanDynamicUniform, OceanH0Image, OceanH0Uniform, OceanMaterialUniform, OceanPlugin,
    OceanSimSettings, OceanSurfaceExtension, OceanSurfaceMaterial,
};

/// Grid edge lengths for the ocean FFT (powers of two).
const OCEAN_GRID_SIZES: [u32; 4] = [128, 256, 512, 1024];

/// Vertices per axis on the ocean plane (`Plane3d` uses `subdivisions + 2` vertices).
const OCEAN_MESH_VERTEX_OPTIONS: [u32; 6] = [16, 32, 64, 128, 256, 512];

#[derive(Resource, Clone, Copy)]
struct OceanMeshSelection {
    vertices_per_edge: u32,
}

impl Default for OceanMeshSelection {
    fn default() -> Self {
        Self {
            vertices_per_edge: 128,
        }
    }
}

impl OceanMeshSelection {
    fn plane_subdivisions(self) -> u32 {
        self.vertices_per_edge.saturating_sub(2)
    }
}

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
        ))
        .add_systems(Startup, setup)
        .add_systems(Update, spawn_ocean_when_ready.after(prepare_fft_textures))
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

#[derive(Component)]
struct OceanSurfaceTag;

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

fn setup(mut commands: Commands, mut scattering_mediums: ResMut<Assets<ScatteringMedium>>) {
    let n = 512u32;
    commands.spawn((
        FftSource::square_inverse_only(n),
        OceanSimSettings {
            texture_size: n,
            ..default()
        },
        OceanH0Uniform::default(),
        OceanDynamicUniform::default(),
    ));

    // Match `examples/3d/atmosphere.rs` camera stack (HDR path is implicit; atmosphere adds SSR + FXAA there).
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 8.0, 24.0).looking_at(Vec3::ZERO, Vec3::Y),
        FreeCamera::default(),
        Atmosphere::earthlike(scattering_mediums.add(ScatteringMedium::default())),
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
            shadows_enabled: true,
            illuminance: lux::RAW_SUNLIGHT,
            ..default()
        },
        SunLight,
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

fn spawn_ocean_when_ready(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<OceanSurfaceMaterial>>,
    mesh_sel: Res<OceanMeshSelection>,
    q: Query<
        (&FftTextures, &OceanSimSettings),
        (
            With<FftSource>,
            With<OceanH0Uniform>,
            With<OceanDynamicUniform>,
        ),
    >,
    existing: Query<Entity, With<OceanSurfaceTag>>,
) {
    if !existing.is_empty() {
        return;
    }

    let Ok((tex, sim)) = q.single() else {
        return;
    };

    let tile = sim.tile_size;
    let mesh = Plane3d::new(Vec3::Y, Vec2::splat(tile * 0.5))
        .mesh()
        .subdivisions(mesh_sel.plane_subdivisions())
        .build();

    let material = OceanSurfaceMaterial {
        base: StandardMaterial {
            base_color: Color::srgb(0.04, 0.12, 0.22).into(),
            perceptual_roughness: 0.08,
            metallic: 0.0,
            opaque_render_method: OpaqueRendererMethod::Auto,
            ..default()
        },
        extension: OceanSurfaceExtension {
            settings: OceanMaterialUniform {
                amplitude: 1.0,
                choppiness: 0.35,
                ocean_tile_world_size: tile,
                grid_size: sim.texture_size as f32,
                wind_direction: sim.wind_direction,
            },
            displacement: tex.spatial_output.clone(),
        },
    };

    commands.spawn((
        Mesh3d(meshes.add(mesh)),
        MeshMaterial3d(materials.add(material)),
        Transform::from_xyz(0.0, 0.0, 0.0),
        OceanSurfaceTag,
    ));
}

fn ocean_egui_panel(
    fft_textures: Query<&FftTextures, With<FftSource>>,
    mut contexts: EguiContexts,
    mut commands: Commands,
    mut mesh_sel: ResMut<OceanMeshSelection>,
    mut cache: Local<OceanFftPreviewEgui>,
    mut smoothed_frame_ms: Local<Option<f32>>,
    time: Res<Time>,
    mut sim: Query<
        (Entity, &mut FftSource, &mut OceanSimSettings),
        (With<OceanH0Uniform>, With<OceanDynamicUniform>),
    >,
    mut materials: ResMut<Assets<OceanSurfaceMaterial>>,
    surface: Query<Entity, With<OceanSurfaceTag>>,
    surface_mat: Query<&MeshMaterial3d<OceanSurfaceMaterial>, With<OceanSurfaceTag>>,
    mut sun_settings: ResMut<SunLightSettings>,
    mut sun_transform: Query<&mut Transform, With<SunLight>>,
) {
    let Ok((sim_entity, mut fft_source, mut s)) = sim.single_mut() else {
        return;
    };

    if let Ok(tex) = fft_textures.single() {
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

    let surface_ready = surface_mat
        .single()
        .ok()
        .and_then(|m| materials.get(&m.0));
    let mut mesh_amplitude = surface_ready
        .map(|m| m.extension.settings.amplitude)
        .unwrap_or(1.0);
    let mut choppiness = surface_ready
        .map(|m| m.extension.settings.choppiness)
        .unwrap_or(0.35);

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
        ui.label(
            egui::RichText::new(format!("{smooth_ms:.2} ms/frame")).weak(),
        );
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
                .remove::<OceanH0Image>();
            *fft_source = FftSource::square_inverse_only(new_n);
            s.texture_size = new_n;
            s.h0_serial = s.h0_serial.wrapping_add(1);
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

        ui.add(egui::Slider::new(&mut s.wind_speed, 0.0..=6.0).text("Wind speed"));
        ui.add(
            egui::Slider::new(
                &mut s.wind_direction,
                -std::f32::consts::PI..=std::f32::consts::PI,
            )
            .text("Wind direction"),
        );
        ui.add(egui::Slider::new(&mut s.peak_enhancement, 1.0..=8.0).text("Peak enhancement"));
        ui.add(egui::Slider::new(&mut s.directional_spread, 0.0..=16.0).text("Spread"));
        ui.add(egui::Slider::new(&mut s.small_wave_cutoff, 0.000..=0.1).text("Small wave cutoff"));
        ui.add(egui::Slider::new(&mut s.amplitude_scale, 0.0..=4.0).text("H0 amplitude scale"));
        ui.add(egui::Slider::new(&mut s.time_scale, 0.0..=3.0).text("Time scale"));
        ui.add(egui::Slider::new(&mut mesh_amplitude, 0.0..=4.0).text("Mesh amplitude"));
        ui.add(egui::Slider::new(&mut choppiness, 0.0..=2.0).text("Choppiness"));
        ui.add(
            egui::Slider::new(&mut sun_settings.elevation, 0.0..=1.0).text("Sun light height"),
        );
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

    let Ok(handle) = surface_mat.single() else {
        return;
    };
    let Some(mat) = materials.get_mut(handle) else {
        return;
    };
    mat.extension.settings.amplitude = mesh_amplitude;
    mat.extension.settings.choppiness = choppiness;
    mat.extension.settings.ocean_tile_world_size = s.tile_size;
    mat.extension.settings.grid_size = s.texture_size as f32;
    mat.extension.settings.wind_direction = s.wind_direction;

    let Ok(mut tf) = sun_transform.single_mut() else {
        return;
    };
    *tf = Transform::from_xyz(0.0, sun_settings.elevation, -1.0).looking_at(Vec3::ZERO, Vec3::Y);
}
