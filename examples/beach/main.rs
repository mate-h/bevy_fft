//! A shallow water beach inside the FFT ocean, drawn by one surface.
//!
//! A 256 m shallow water grid with the shore preset sits at the origin. It writes into the shared
//! water LOD data, so the ocean clipmap shows the beach and the open sea with the same material,
//! and the grid fades into the ocean waves along its edges. The buoys read the same data through
//! `OceanQueries`, one offshore, one in the fade band, and one in the surf.
//!
//! The deep edge is an ocean border. The ocean's long waves are written over the grid each frame,
//! and a band of cells along the deep edge blends the water height toward them, so swell enters
//! the grid, shoals up the slope, and runs up the beach. A PML behind the band absorbs the backwash.
//! The panel switches the deep edge to a wall or a drain for comparison. Drag with the left mouse
//! button to add water.

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

/// Shallow water cells per edge.
const GRID: u32 = 256;
/// Shallow water cell width in meters.
const CELL_SIZE: f32 = 1.0;
/// Still water level in solver meters. The shore preset's bed runs from 0 to 10 m, so this puts
/// the waterline partway up the slope. The solver is shifted down by this much so the sea sits at
/// the ocean's rest height.
const SEA_LEVEL: f32 = 4.0;
/// Shore preset index in `loadPreset`.
const SHORE_PRESET: u32 = 3;

/// World `xz` of the demo buoys: offshore, in the fade band, and in the surf.
const BUOY_POSITIONS: [Vec2; 3] = [
    Vec2::new(0.0, -170.0),
    Vec2::new(10.0, -124.0),
    Vec2::new(-10.0, -60.0),
];

fn main() {
    App::new()
        .insert_resource(DefaultOpaqueRendererMethod::deferred())
        .insert_resource(ClearColor(Color::BLACK))
        .insert_resource(GlobalAmbientLight::NONE)
        .add_plugins((
            DefaultPlugins,
            FreeCameraPlugin,
            FftPlugin,
            OceanPlugin,
            ShallowWaterPlugin,
            EguiPlugin::default(),
            screenshot::ScreenshotOnEnvPlugin,
        ))
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (
                spawn_surface_when_ready.after(FftSystemSet::PrepareTextures),
                float_buoys,
            ),
        )
        .add_systems(EguiPrimaryContextPass, beach_panel)
        .add_systems(
            PostUpdate,
            (
                pointer_to_sim_space
                    .after(TransformSystems::Propagate)
                    .after(write_egui_wants_input_system),
                sync_free_camera_with_egui_focus.after(write_egui_wants_input_system),
            ),
        )
        .run();
}

#[derive(Component)]
struct Buoy {
    xz: Vec2,
}

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
                wind_speed: 8.0,
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

    let mut controller = ShallowWaterController::new(&mut images, GRID, GRID);
    controller.placement = Some(ShallowWaterPlacement {
        center: Vec2::ZERO,
        cell_size: CELL_SIZE,
        datum: -SEA_LEVEL,
        blend_cells: 16.0,
    });
    controller.preset_index = SHORE_PRESET;
    controller.fill_level = SEA_LEVEL;
    controller.pml_eta_rest = SEA_LEVEL;
    controller.pml_width = 8;
    controller.bottom_border = ShallowWaterBorder::Ocean;
    controller.ocean_forcing = true;
    controller.gravity = 9.81;
    controller.dt = 1.0 / 60.0;
    controller.brush_radius = 6.0;
    commands.insert_resource(controller);

    let buoy_mesh = meshes.add(Sphere::new(1.2).mesh().uv(24, 12));
    let buoy_material = standard_materials.add(StandardMaterial {
        base_color: Color::srgb(0.95, 0.35, 0.1),
        perceptual_roughness: 0.5,
        ..default()
    });
    for xz in BUOY_POSITIONS {
        commands.spawn((
            Mesh3d(buoy_mesh.clone()),
            MeshMaterial3d(buoy_material.clone()),
            Transform::from_xyz(xz.x, 0.0, xz.y),
            Buoy { xz },
        ));
    }

    let earth_medium = scattering_mediums.add(ScatteringMedium::earth(256, 256));
    commands.spawn(Atmosphere::earth(earth_medium));
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(-90.0, 45.0, -230.0).looking_at(Vec3::new(0.0, 0.0, -40.0), Vec3::Y),
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

/// Places each buoy on the queried surface, extrapolated over the readback latency.
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
    let age = time.elapsed_secs() - results.time;
    for (buoy, mut transform) in &mut buoys {
        let Some(i) = results.points.iter().position(|p| *p == buoy.xz) else {
            continue;
        };
        let sample = results.samples[i];
        transform.translation = Vec3::new(buoy.xz.x, sample.height_after(age), buoy.xz.y);
        transform.rotation = Quat::from_rotation_arc(Vec3::Y, sample.normal());
    }
}

fn sync_free_camera_with_egui_focus(
    mut state: Query<&mut FreeCameraState, With<FreeCamera>>,
    egui_wants: Res<EguiWantsInput>,
) {
    if let Ok(mut camera_state) = state.single_mut() {
        camera_state.enabled = !egui_wants.wants_any_input();
    }
}

/// Maps the cursor onto the sea plane and into grid cells for the water brush.
fn pointer_to_sim_space(
    mouse: Res<ButtonInput<MouseButton>>,
    egui_wants: Res<EguiWantsInput>,
    window: Query<&Window, With<bevy::window::PrimaryWindow>>,
    camera: Query<(&Camera, &GlobalTransform), With<FreeCamera>>,
    mut controller: ResMut<ShallowWaterController>,
) {
    controller.brush_input_active = false;
    let (Ok(window), Ok((camera, cam_transform))) = (window.single(), camera.single()) else {
        return;
    };
    let Some(placement) = controller.placement else {
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
    let cells = UVec2::new(controller.cells_x, controller.cells_y);
    let sim = placement.world_to_cells(hit.xz(), cells);
    let paint = mouse.pressed(MouseButton::Left) && !egui_wants.is_pointer_over_area();
    controller.pointer_prev_sim = if paint && !mouse.just_pressed(MouseButton::Left) {
        controller.pointer_sim
    } else {
        sim
    };
    controller.pointer_sim = sim;
    controller.brush_input_active = paint;
}

fn beach_panel(mut contexts: EguiContexts, mut controller: ResMut<ShallowWaterController>) {
    let Ok(ctx) = contexts.ctx_mut() else {
        return;
    };
    egui::Window::new("Beach").show(ctx, |ui| {
        ui.checkbox(&mut controller.paused, "Paused");
        if ui.button("Reset").clicked() {
            controller.sim_apply_serial = controller.sim_apply_serial.wrapping_add(1);
        }
        if let Some(mut placement) = controller.placement {
            ui.add(egui::Slider::new(&mut placement.blend_cells, 0.0..=64.0).text("Edge fade"));
            controller.placement = Some(placement);
        }
        ui.add(egui::Slider::new(&mut controller.pml_width, 0u32..=32u32).text("PML width"));
        ui.add(egui::Slider::new(&mut controller.ocean_band_cells, 0.0..=16.0).text("Ocean band"));
        ui.add(egui::Slider::new(&mut controller.friction, 0.0..=0.5).text("Friction"));
        ui.horizontal(|ui| {
            ui.label("Deep edge");
            let border = &mut controller.bottom_border;
            egui::ComboBox::from_id_salt("deep_edge")
                .selected_text(format!("{border:?}"))
                .show_ui(ui, |ui| {
                    ui.selectable_value(border, ShallowWaterBorder::Ocean, "Ocean");
                    ui.selectable_value(border, ShallowWaterBorder::Wall, "Wall");
                    ui.selectable_value(border, ShallowWaterBorder::Drain, "Drain");
                });
        });
        ui.add(egui::Slider::new(&mut controller.brush_radius, 2.0..=24.0).text("Brush radius"));
    });
}
