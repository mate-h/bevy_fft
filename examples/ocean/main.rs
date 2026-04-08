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
    OceanDynamicUniform, OceanH0Uniform, OceanMaterialUniform, OceanPlugin, OceanSimSettings,
    OceanSurfaceExtension, OceanSurfaceMaterial,
};

fn main() {
    App::new()
        // Same as `examples/3d/atmosphere.rs`: deferred opaques (water there uses `ExtendedMaterial` + deferred fragment).
        .insert_resource(DefaultOpaqueRendererMethod::deferred())
        .insert_resource(ClearColor(Color::BLACK))
        .insert_resource(GlobalAmbientLight::NONE)
        .init_resource::<SunLightSettings>()
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
    commands.spawn((
        FftSource::grid_256_inverse_only(),
        OceanSimSettings::default(),
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
    q: Query<
        (&FftTextures, &OceanSimSettings),
        (
            With<FftSource>,
            With<OceanH0Uniform>,
            With<OceanDynamicUniform>,
            Without<OceanSurfaceTag>,
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
        .subdivisions(255)
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
    mut cache: Local<OceanFftPreviewEgui>,
    mut sim: Query<&mut OceanSimSettings>,
    mut materials: ResMut<Assets<OceanSurfaceMaterial>>,
    surface: Query<&MeshMaterial3d<OceanSurfaceMaterial>, With<OceanSurfaceTag>>,
    mut sun_settings: ResMut<SunLightSettings>,
    mut sun_transform: Query<&mut Transform, With<SunLight>>,
) {
    let Ok(mut s) = sim.single_mut() else {
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

    let surface_ready = surface.single().ok().and_then(|m| materials.get(&m.0));
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

    egui::Window::new("Ocean").show(ctx, |ui| {
        if surface_ready.is_none() {
            ui.label("Spawning ocean mesh after GPU init…");
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

    let Ok(handle) = surface.single() else {
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
