//! Boat-wake style waves from the dispersive solver, riding on the FFT ocean.
//!
//! A 256 m dispersive patch sits at the origin and adds its waves onto the ocean through the
//! shared water LOD data, so one surface draws both and the ocean swell runs straight through the
//! patch. A scripted disturbance circles the patch like a boat. Turn it off in the panel and drag
//! with the left mouse button to make your own wake.
//!
//! The patch is still periodic and fades out over a band at its edge. Following the boat and
//! trimming the edges are separate steps.

use bevy::{
    anti_alias::fxaa::Fxaa,
    camera::Exposure,
    camera_controller::free_camera::{FreeCamera, FreeCameraPlugin, FreeCameraState},
    core_pipeline::{
        prepass::{DeferredPrepass, DepthPrepass},
        tonemapping::Tonemapping,
    },
    light::{
        Atmosphere, AtmosphereEnvironmentMapLight, CascadeShadowConfigBuilder, GlobalAmbientLight,
        atmosphere::ScatteringMedium, light_consts::lux,
    },
    material::OpaqueRendererMethod,
    pbr::{AtmosphereSettings, DefaultOpaqueRendererMethod, StandardMaterial},
    post_process::bloom::Bloom,
    prelude::*,
    shape::InfinitePlane3d,
    transform::TransformSystems,
};
use bevy_egui::{
    EguiContexts, EguiPlugin, EguiPrimaryContextPass, egui,
    input::{EguiWantsInput, write_egui_wants_input_system},
};
use bevy_fft::prelude::*;

#[path = "../common/screenshot.rs"]
mod screenshot;

/// World-space tile edge in meters for each FFT cascade, largest first.
const CASCADE_TILES: [f32; 3] = [256.0, 48.0, 9.0];
/// A cascade hands waves to the next smaller one at this many wavelengths per smaller tile.
const CASCADE_HANDOFF_WAVES: f32 = 6.0;

/// Dispersive cells per edge.
const GRID: u32 = 256;
/// Dispersive patch edge in meters.
const PATCH_WORLD: f32 = 256.0;

/// Radius in meters of the scripted boat's circle.
const BOAT_RADIUS: f32 = 55.0;
/// Speed in m/s of the scripted boat.
const BOAT_SPEED: f32 = 6.0;

fn main() {
    App::new()
        .insert_resource(DefaultOpaqueRendererMethod::deferred())
        .insert_resource(ClearColor(Color::BLACK))
        .insert_resource(GlobalAmbientLight::NONE)
        .init_resource::<ScriptedBoat>()
        .add_plugins((
            DefaultPlugins,
            FreeCameraPlugin,
            FftPlugin,
            OceanPlugin,
            DispersivePlugin,
            EguiPlugin::default(),
            screenshot::ScreenshotOnEnvPlugin,
        ))
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            spawn_surface_when_ready.after(FftSystemSet::PrepareTextures),
        )
        .add_systems(EguiPrimaryContextPass, wake_panel)
        .add_systems(
            PostUpdate,
            (
                drive_brush
                    .after(TransformSystems::Propagate)
                    .after(write_egui_wants_input_system),
                sync_free_camera_with_egui_focus.after(write_egui_wants_input_system),
            ),
        )
        .run();
}

/// A disturbance that circles the patch centre, standing in for a boat hull.
#[derive(Resource)]
struct ScriptedBoat {
    enabled: bool,
    angle: f32,
}

impl Default for ScriptedBoat {
    fn default() -> Self {
        Self {
            enabled: true,
            angle: 0.0,
        }
    }
}

#[derive(Component)]
struct BoatMarker;

/// Wavenumber band in rad/m kept by cascade `index`. Bands touch but never overlap.
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
    mut images: ResMut<Assets<Image>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut standard_materials: ResMut<Assets<StandardMaterial>>,
    mut scattering_mediums: ResMut<Assets<ScatteringMedium>>,
) {
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
                wind_speed: 4.0,
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

    let mut controller = DispersiveController::spawn(&mut commands, &mut images, GRID);
    controller.tile_world = PATCH_WORLD;
    controller.brush_radius = 6.0;
    controller.brush_strength = 1.2;
    controller.placement = Some(DispersivePlacement::default());
    commands.insert_resource(controller);

    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(2.0, 1.0, 5.0))),
        MeshMaterial3d(standard_materials.add(StandardMaterial {
            base_color: Color::srgb(0.9, 0.9, 0.85),
            perceptual_roughness: 0.4,
            ..default()
        })),
        Transform::from_xyz(BOAT_RADIUS, 0.3, 0.0),
        BoatMarker,
    ));

    let earth_medium = scattering_mediums.add(ScatteringMedium::earth(256, 256));
    commands.spawn(Atmosphere::earth(earth_medium));
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(20.0, 45.0, -110.0).looking_at(Vec3::new(20.0, 0.0, 0.0), Vec3::Y),
        FreeCamera::default(),
        AtmosphereSettings::default(),
        Exposure { ev100: 12.0 },
        Tonemapping::AcesFitted,
        Bloom::NATURAL,
        AtmosphereEnvironmentMapLight::default(),
        Msaa::Off,
        Fxaa::default(),
        DepthPrepass,
        DeferredPrepass,
    ));

    commands.spawn((
        DirectionalLight {
            shadow_maps_enabled: true,
            illuminance: lux::RAW_SUNLIGHT,
            ..default()
        },
        Transform::from_xyz(0.3, 0.25, -1.0).looking_at(Vec3::ZERO, Vec3::Y),
        CascadeShadowConfigBuilder {
            first_cascade_far_bound: 20.0,
            maximum_distance: 400.0,
            ..default()
        }
        .build(),
    ));
}

#[allow(clippy::type_complexity)]
fn spawn_surface_when_ready(
    mut commands: Commands,
    mut materials: ResMut<Assets<WaterSurfaceMaterial>>,
    cascades: Query<&OceanSimSettings, (With<FftTextures>, With<OceanFoamMask>)>,
    existing: Query<(), With<WaterSurfaceTag>>,
) {
    if !existing.is_empty() || cascades.iter().count() < CASCADE_TILES.len() {
        return;
    }
    commands.spawn((
        OceanClipmap::default(),
        MeshMaterial3d(materials.add(WaterSurfaceMaterial {
            base: StandardMaterial {
                base_color: Color::srgb(0.04, 0.12, 0.22),
                perceptual_roughness: 0.08,
                metallic: 0.0,
                opaque_render_method: OpaqueRendererMethod::Auto,
                ..default()
            },
            extension: WaterSurfaceExtension::default(),
        })),
        WaterSurfaceTag,
    ));
}

fn sync_free_camera_with_egui_focus(
    mut state: Query<&mut FreeCameraState, With<FreeCamera>>,
    egui_wants: Res<EguiWantsInput>,
) {
    if let Ok(mut camera_state) = state.single_mut() {
        camera_state.enabled = !egui_wants.wants_any_input();
    }
}

/// Moves the brush with the scripted boat, or with the cursor while the left button is held.
#[allow(clippy::too_many_arguments)]
fn drive_brush(
    time: Res<Time>,
    mouse: Res<ButtonInput<MouseButton>>,
    egui_wants: Res<EguiWantsInput>,
    window: Query<&Window, With<bevy::window::PrimaryWindow>>,
    camera: Query<(&Camera, &GlobalTransform), With<FreeCamera>>,
    mut boat: ResMut<ScriptedBoat>,
    mut marker: Query<(&mut Transform, &mut Visibility), With<BoatMarker>>,
    mut controller: ResMut<DispersiveController>,
) {
    if let Ok((mut transform, mut visibility)) = marker.single_mut() {
        *visibility = if boat.enabled {
            Visibility::Inherited
        } else {
            Visibility::Hidden
        };
        if boat.enabled {
            boat.angle += BOAT_SPEED / BOAT_RADIUS * time.delta_secs();
            let xz = Vec2::from_angle(boat.angle) * BOAT_RADIUS;
            let heading = Vec2::from_angle(boat.angle + std::f32::consts::FRAC_PI_2);
            transform.translation = Vec3::new(xz.x, 0.3, xz.y);
            transform.rotation = Quat::from_rotation_y(heading.x.atan2(heading.y));
            let sim = controller.world_to_cells(xz);
            controller.pointer_prev = controller.pointer;
            controller.pointer = sim;
            controller.brush_active = true;
            return;
        }
    }

    controller.brush_active = false;
    let (Ok(window), Ok((camera, cam_transform))) = (window.single(), camera.single()) else {
        return;
    };
    let Some(cursor) = window.cursor_position() else {
        return;
    };
    let Ok(ray) = camera.viewport_to_world(cam_transform, cursor) else {
        return;
    };
    let Some(hit) = ray.plane_intersection_point(Vec3::ZERO, InfinitePlane3d::new(Vec3::Y)) else {
        return;
    };
    let sim = controller.world_to_cells(hit.xz());
    let paint = mouse.pressed(MouseButton::Left) && !egui_wants.is_pointer_over_area();
    controller.pointer_prev = if paint && !mouse.just_pressed(MouseButton::Left) {
        controller.pointer
    } else {
        sim
    };
    controller.pointer = sim;
    controller.brush_active = paint;
}

fn wake_panel(
    mut contexts: EguiContexts,
    mut controller: ResMut<DispersiveController>,
    mut boat: ResMut<ScriptedBoat>,
) {
    let Ok(ctx) = contexts.ctx_mut() else {
        return;
    };
    egui::Window::new("Wake").show(ctx, |ui| {
        ui.checkbox(&mut boat.enabled, "Scripted boat");
        ui.checkbox(&mut controller.paused, "Paused");
        if ui.button("Reset").clicked() {
            controller.init_serial = controller.init_serial.wrapping_add(1);
        }
        if let Some(mut placement) = controller.placement {
            ui.add(egui::Slider::new(&mut placement.blend_cells, 0.0..=64.0).text("Edge fade"));
            ui.horizontal(|ui| {
                ui.label("Blend");
                ui.selectable_value(&mut placement.blend, DispersiveLodBlend::Add, "Add");
                ui.selectable_value(&mut placement.blend, DispersiveLodBlend::Replace, "Replace");
            });
            controller.placement = Some(placement);
        }
        ui.add(egui::Slider::new(&mut controller.brush_radius, 2.0..=16.0).text("Brush radius"));
        ui.add(egui::Slider::new(&mut controller.brush_strength, 0.05..=2.0).text("Strength"));
    });
}
