//! Hybrid dispersive (Jeschke–Wojtan style) on a water patch with a sloped beach (GPU init) and FFT surface.
//!
//! Add `FftPlugin` before `DispersivePlugin`.
//! Right-click: small height bump written into the sim `Image` (same data as `d_state` on the GPU). Left: free camera.

use bevy::{
    camera::Exposure,
    camera_controller::free_camera::{FreeCamera, FreeCameraPlugin, FreeCameraState},
    color::LinearRgba,
    core_pipeline::tonemapping::Tonemapping,
    input::mouse::MouseButton,
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
    window::PrimaryWindow,
};
use bevy_egui::input::{EguiWantsInput, write_egui_wants_input_system};
use bevy_egui::EguiPlugin;
use bevy_fft::prelude::{
    DispersiveController, DispersiveMaterialUniform, DispersivePlugin, DispersiveSurfaceExtension,
    DispersiveSurfaceMaterial, DispersiveSurfaceTag, FftPlugin,
};

const GRID: u32 = 256;
const PATCH_HALF_EXTENT: f32 = 32.0;
const PATCH_TILE_WORLD: f32 = PATCH_HALF_EXTENT * 2.0;
const CAMERA_OFFSET: Vec3 = Vec3::new(0.0, 10.0, 55.0);
/// Sim texels: disk radius for an RMB height bump to `d_state` channel `r` (water column).
const SPLASH_RADIUS_TEXELS: i32 = 10;
const SPLASH_PEAK_ADD_H: f32 = 1.5;

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
            DispersivePlugin,
            EguiPlugin::default(),
        ))
        .add_systems(Startup, (setup, dispersive_splash_help_once))
        .add_systems(
            PostUpdate,
            (
                dispersive_rmb_splash
                    .after(write_egui_wants_input_system),
                sync_free_camera_with_egui_focus
                    .after(TransformSystems::Propagate)
                    .after(write_egui_wants_input_system),
            ),
        )
        .run();
}

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

fn dispersive_splash_help_once() {
    info!("dispersive: RMB = height splash on the water (CPU write to the sim `Image`); hold LMB or use M for the free camera.");
}

/// Add `h` in the `.r` channel of `d_state` under the cursor (see `ewave` for a full GPU brush).
fn dispersive_rmb_splash(
    mouse: Res<ButtonInput<MouseButton>>,
    egui: Res<EguiWantsInput>,
    window: Query<&Window, With<PrimaryWindow>>,
    camera: Query<(&Camera, &GlobalTransform), With<FreeCamera>>,
    surface: Query<&GlobalTransform, With<DispersiveSurfaceTag>>,
    mut images: ResMut<Assets<Image>>,
    mut controller: ResMut<DispersiveController>,
) {
    if !mouse.just_pressed(MouseButton::Right) || controller.paused {
        return;
    }
    if egui.is_pointer_over_area() {
        info!("dispersive: RMB ignored (pointer over egui).");
        return;
    }

    let n = controller.n;
    let n1 = n.saturating_sub(1) as i32;

    let Some((ix, iy)) = (|| {
        let w = window.single().ok()?;
        let cur = w.cursor_position()?;
        let (cam, cam_tf) = camera.single().ok()?;
        let ray = cam.viewport_to_world(cam_tf, cur).ok()?;
        let surf = surface.single().ok()?;
        let plane = InfinitePlane3d::new(surf.up());
        let p = ray.plane_intersection_point(surf.translation(), plane)?;
        let l = surf.affine().inverse().transform_point3(p);
        let u = (l.x / PATCH_TILE_WORLD + 0.5).clamp(0.0, 1.0);
        let v = (l.z / PATCH_TILE_WORLD + 0.5).clamp(0.0, 1.0);
        let iu = (u * n1 as f32).round() as u32;
        let iv = (v * n1 as f32).round() as u32;
        Some((iu.min(n - 1), iv.min(n - 1)))
    })() else {
        info!("dispersive: RMB no hit on the water patch (or missing window, camera, or surface).");
        return;
    };

    let Some(img) = images.get_mut(&controller.state) else {
        info!("dispersive: RMB no `Assets<Image>` entry for the state handle.");
        return;
    };

    let r2 = (SPLASH_RADIUS_TEXELS * SPLASH_RADIUS_TEXELS) as f32;
    for dy in -SPLASH_RADIUS_TEXELS..=SPLASH_RADIUS_TEXELS {
        for dx in -SPLASH_RADIUS_TEXELS..=SPLASH_RADIUS_TEXELS {
            let s = (dx * dx + dy * dy) as f32;
            if s > r2 {
                continue;
            }
            let t = 1.0 - (s / r2).sqrt();
            let add = SPLASH_PEAK_ADD_H * t * t;
            if add < 1e-6 {
                continue;
            }
            let x = (ix as i32 + dx).clamp(0, n1) as u32;
            let y = (iy as i32 + dy).clamp(0, n1) as u32;
            let Ok(c) = img.get_color_at(x, y) else {
                continue;
            };
            let l = c.to_linear();
            let h = (l.red + add).min(4.0);
            let c2 = Color::from(LinearRgba {
                red: h,
                green: l.green,
                blue: l.blue,
                alpha: l.alpha,
            });
            if img.set_color_at(x, y, c2).is_err() {
                continue;
            }
        }
    }

    controller.sim_apply_serial = controller.sim_apply_serial.wrapping_add(1);
    info!(
        "dispersive: RMB splash center texel=({}, {}), sim_apply_serial={} (bump to refresh GPU bind groups).",
        ix, iy, controller.sim_apply_serial
    );
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

fn setup(
    mut commands: Commands,
    sun: Res<SunLightSettings>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<DispersiveSurfaceMaterial>>,
    mut images: ResMut<Assets<Image>>,
    mut scattering_mediums: ResMut<Assets<ScatteringMedium>>,
) {
    let mut controller = DispersiveController::spawn(&mut commands, &mut images, GRID);
    controller.tile_world = PATCH_TILE_WORLD;
    controller.height_scale = 0.18;
    let state = controller.state.clone();
    commands.insert_resource(controller);

    let plane = Plane3d::new(Vec3::Y, Vec2::splat(PATCH_HALF_EXTENT))
        .mesh()
        .subdivisions(127)
        .build();
    let material = DispersiveSurfaceMaterial {
        base: StandardMaterial {
            base_color: Color::srgb(0.05, 0.2, 0.35).into(),
            perceptual_roughness: 0.08,
            metallic: 0.0,
            opaque_render_method: OpaqueRendererMethod::Auto,
            ..default()
        },
        extension: DispersiveSurfaceExtension {
            settings: DispersiveMaterialUniform {
                height_scale: 0.18,
                tile_world_size: PATCH_TILE_WORLD,
                grid_size: GRID as f32,
                _pad0: 0.0,
            },
            state,
        },
    };

    let patch_origin = Vec3::new(PATCH_HALF_EXTENT, 0.0, PATCH_HALF_EXTENT);
    commands.spawn((
        Mesh3d(meshes.add(plane)),
        MeshMaterial3d(materials.add(material)),
        Transform::from_translation(patch_origin),
        DispersiveSurfaceTag,
    ));

    let patch_center = patch_origin;
    let cascade_shadow_config = CascadeShadowConfigBuilder {
        first_cascade_far_bound: 0.3,
        maximum_distance: 25.0,
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
