//! Periodic eWave surface: deep-water-style linear evolution in Fourier space with GPU FFT.
//!
//! **Plugin order:** add [`FftPlugin`](bevy_fft::fft::FftPlugin) before [`EwavePlugin`](bevy_fft::ewave::EwavePlugin).
//! `EwavePlugin::finish` needs FFT render resources; reversing the order panics at startup.

use bevy::{
    camera::Exposure,
    camera_controller::free_camera::{FreeCamera, FreeCameraPlugin, FreeCameraState},
    core_pipeline::tonemapping::Tonemapping,
    light::{
        AtmosphereEnvironmentMapLight, CascadeShadowConfigBuilder, GlobalAmbientLight,
        light_consts::lux,
    },
    math::primitives::InfinitePlane3d,
    mesh::Meshable,
    pbr::{
        Atmosphere, AtmosphereSettings, DefaultOpaqueRendererMethod, OpaqueRendererMethod,
        ScatteringMedium, StandardMaterial,
    },
    post_process::bloom::Bloom,
    prelude::*,
    render::{
        RenderPlugin,
        render_resource::WgpuFeatures,
        settings::{RenderCreation, WgpuSettings},
    },
    transform::TransformSystems,
};
use bevy_egui::{
    EguiContexts, EguiPlugin, EguiPrimaryContextPass, egui,
    input::{EguiWantsInput, write_egui_wants_input_system},
};
use bevy_fft::prelude::{
    EwaveController, EwaveMaterialUniform, EwavePlugin, EwaveSurfaceExtension,
    EwaveSurfaceMaterial, EwaveSurfaceTag, FftPlugin,
};

const GRID: u32 = 256;
const PATCH_HALF_EXTENT: f32 = 16.0;
const PATCH_TILE_WORLD: f32 = PATCH_HALF_EXTENT * 2.0;

const CAMERA_OFFSET: Vec3 = Vec3::new(0.0, 12.0, 32.0);

fn main() {
    let mut wgpu = WgpuSettings::default();
    wgpu.features |= WgpuFeatures::FLOAT32_FILTERABLE;

    App::new()
        .insert_resource(DefaultOpaqueRendererMethod::forward())
        .insert_resource(ClearColor(Color::BLACK))
        .insert_resource(GlobalAmbientLight::NONE)
        .init_resource::<SunLightSettings>()
        .add_plugins((
            DefaultPlugins.set(RenderPlugin {
                render_creation: RenderCreation::Automatic(wgpu),
                ..default()
            }),
            FreeCameraPlugin,
            FftPlugin,
            EwavePlugin,
            EguiPlugin::default(),
        ))
        .add_systems(Startup, setup)
        .add_systems(EguiPrimaryContextPass, ewave_egui_panel)
        .add_systems(
            PostUpdate,
            (
                pointer_to_grid
                    .after(TransformSystems::Propagate)
                    .after(write_egui_wants_input_system),
                sync_free_camera_with_egui_focus.after(write_egui_wants_input_system),
            ),
        )
        .run();
}

/// Y component of the directional light’s position before `looking_at` (see `examples/ocean/main.rs`).
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

fn sync_free_camera_with_egui_focus(
    mut state: Query<&mut FreeCameraState, With<FreeCamera>>,
    egui_wants: Res<EguiWantsInput>,
) {
    let Ok(mut camera_state) = state.single_mut() else {
        return;
    };
    camera_state.enabled = !egui_wants.wants_any_input();
}

fn setup(
    mut commands: Commands,
    sun: Res<SunLightSettings>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<EwaveSurfaceMaterial>>,
    mut images: ResMut<Assets<Image>>,
    mut scattering_mediums: ResMut<Assets<ScatteringMedium>>,
) {
    let mut controller = EwaveController::spawn(&mut commands, &mut images, GRID);
    controller.tile_world = PATCH_TILE_WORLD;
    controller.height_scale = 0.12;
    let h_phi = controller.h_phi().clone();
    commands.insert_resource(controller);

    let plane = Plane3d::new(Vec3::Y, Vec2::splat(PATCH_HALF_EXTENT))
        .mesh()
        .subdivisions(127)
        .build();
    let material = EwaveSurfaceMaterial {
        base: StandardMaterial {
            base_color: Color::srgb(0.05, 0.15, 0.28).into(),
            perceptual_roughness: 0.06,
            metallic: 0.0,
            opaque_render_method: OpaqueRendererMethod::Auto,
            ..default()
        },
        extension: EwaveSurfaceExtension {
            settings: EwaveMaterialUniform {
                height_scale: 0.12,
                tile_world_size: PATCH_TILE_WORLD,
                grid_size: GRID as f32,
                _pad0: 0.0,
            },
            h_phi,
        },
    };

    let patch_origin = Vec3::new(PATCH_HALF_EXTENT, 0.0, PATCH_HALF_EXTENT);
    commands.spawn((
        Mesh3d(meshes.add(plane)),
        MeshMaterial3d(materials.add(material)),
        Transform::from_translation(patch_origin),
        EwaveSurfaceTag,
    ));

    let patch_center = patch_origin;
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
        Transform::from_xyz(0.0, sun.elevation, -1.0).looking_at(Vec3::ZERO, Vec3::Y),
        cascade_shadow_config,
    ));

    commands.spawn((
        Camera3d::default(),
        Transform::from_translation(patch_center + CAMERA_OFFSET).looking_at(patch_center, Vec3::Y),
        FreeCamera::default(),
        Atmosphere::earthlike(scattering_mediums.add(ScatteringMedium::default())),
        AtmosphereSettings::default(),
        Exposure { ev100: 12.0 },
        Tonemapping::AcesFitted,
        Bloom::NATURAL,
        AtmosphereEnvironmentMapLight::default(),
    ));
}

fn pointer_to_grid(
    mouse: Res<ButtonInput<MouseButton>>,
    egui_wants: Res<EguiWantsInput>,
    window: Query<&Window, With<bevy::window::PrimaryWindow>>,
    camera: Query<(&Camera, &GlobalTransform), With<FreeCamera>>,
    surface: Query<&GlobalTransform, With<EwaveSurfaceTag>>,
    mut controller: ResMut<EwaveController>,
) {
    let Ok(window) = window.single() else {
        return;
    };
    let Ok((camera, cam_transform)) = camera.single() else {
        return;
    };
    let Ok(surface_transform) = surface.single() else {
        return;
    };
    let Some(cursor) = window.cursor_position() else {
        return;
    };

    let Ok(ray) = camera.viewport_to_world(cam_transform, cursor) else {
        return;
    };

    let plane_origin = surface_transform.translation();
    let plane = InfinitePlane3d::new(surface_transform.up());
    let Some(world_hit) = ray.plane_intersection_point(plane_origin, plane) else {
        return;
    };

    let local_hit = surface_transform
        .affine()
        .inverse()
        .transform_point3(world_hit);
    let u = (local_hit.x / PATCH_TILE_WORLD + 0.5).clamp(0.0, 1.0);
    let v = (local_hit.z / PATCH_TILE_WORLD + 0.5).clamp(0.0, 1.0);
    let sx = u * controller.n as f32;
    let sy = v * controller.n as f32;

    let new_sim = Vec2::new(sx, sy);
    // Paint on the 3D patch only when the pointer is not over an egui window. `wants_pointer_input`
    // is often stricter; `is_pointer_over_area` matches “cursor over UI chrome”.
    let paint = mouse.pressed(MouseButton::Right) && !egui_wants.is_pointer_over_area();

    if mouse.just_pressed(MouseButton::Right) && paint {
        controller.pointer_prev = new_sim;
        controller.pointer = new_sim;
    } else if paint {
        let prev = controller.pointer;
        controller.pointer_prev = prev;
        controller.pointer = new_sim;
    } else {
        controller.pointer_prev = new_sim;
        controller.pointer = new_sim;
    }

    controller.brush_active = paint;
}

fn ewave_egui_panel(
    mut commands: Commands,
    mut contexts: EguiContexts,
    mut sun_settings: ResMut<SunLightSettings>,
    mut controller: ResMut<EwaveController>,
    mut images: ResMut<Assets<Image>>,
    mut sun_transform: Query<&mut Transform, With<SunLight>>,
) {
    let Ok(ctx) = contexts.ctx_mut() else {
        return;
    };
    egui::Window::new("eWave").show(ctx, |ui| {
        ui.checkbox(&mut controller.paused, "Paused");
        if ui.button("Reset field").clicked() {
            controller.sim_apply_serial = controller.sim_apply_serial.wrapping_add(1);
        }
        ui.add(egui::Slider::new(&mut controller.dt, 0.01..=0.12).text("dt"));
        ui.add(egui::Slider::new(&mut controller.g, 1.0..=25.0).text("g"));
        ui.add(egui::Slider::new(&mut controller.height_scale, 0.02..=0.4).text("height scale"));
        ui.add(
            egui::Slider::new(&mut controller.tile_world, 4.0..=128.0).text("tile world (period)"),
        );
        let mut n = controller.n;
        if ui
            .add(egui::Slider::new(&mut n, 128u32..=512u32).text("grid N (power of two)"))
            .changed()
        {
            let n = n.next_power_of_two().clamp(128, 512);
            controller.rebuild(&mut commands, &mut images, n);
        }
        ui.separator();
        ui.label("Brush (RMB on patch)");
        ui.add(egui::Slider::new(&mut controller.brush_radius, 4.0..=64.0).text("radius"));
        ui.add(egui::Slider::new(&mut controller.brush_strength, 0.05..=1.5).text("strength"));
        ui.add(egui::Slider::new(&mut sun_settings.elevation, 0.0..=1.0).text("Sun light height"));
    });

    let Ok(mut tf) = sun_transform.single_mut() else {
        return;
    };
    *tf = Transform::from_xyz(0.0, sun_settings.elevation, -1.0).looking_at(Vec3::ZERO, Vec3::Y);
}
