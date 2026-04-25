//! Shallow water on a displaced PBR plane: staggered SWE on the GPU plus brush interaction.

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
    ShallowWaterBorder, ShallowWaterController, ShallowWaterMaterialUniform, ShallowWaterPlugin,
    ShallowWaterSurfaceExtension, ShallowWaterSurfaceMaterial, ShallowWaterSurfaceTag,
    round_particle_count,
};

const GRID: u32 = 256;
/// `Plane3d` half size on X and Z (world units); full tile width is `2 * PATCH_HALF_EXTENT`.
const PATCH_HALF_EXTENT: f32 = 16.0;
const PATCH_TILE_WORLD: f32 = PATCH_HALF_EXTENT * 2.0;

fn main() {
    let mut wgpu = WgpuSettings::default();
    wgpu.features |= WgpuFeatures::FLOAT32_FILTERABLE;

    App::new()
        .insert_resource(DefaultOpaqueRendererMethod::forward())
        .insert_resource(ClearColor(Color::BLACK))
        .insert_resource(GlobalAmbientLight::NONE)
        .add_plugins((
            DefaultPlugins.set(RenderPlugin {
                render_creation: RenderCreation::Automatic(wgpu),
                ..default()
            }),
            FreeCameraPlugin,
            ShallowWaterPlugin,
            EguiPlugin::default(),
        ))
        .add_systems(Startup, setup)
        .add_systems(EguiPrimaryContextPass, shallow_water_egui_panel)
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
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ShallowWaterSurfaceMaterial>>,
    mut images: ResMut<Assets<Image>>,
    mut scattering_mediums: ResMut<Assets<ScatteringMedium>>,
) {
    let controller = ShallowWaterController::new(&mut images, GRID, GRID);
    let bed_water = controller.bed_water.clone();
    commands.insert_resource(controller);

    let plane = Plane3d::new(Vec3::Y, Vec2::splat(PATCH_HALF_EXTENT))
        .mesh()
        .subdivisions(127)
        .build();
    let material = ShallowWaterSurfaceMaterial {
        base: StandardMaterial {
            base_color: Color::srgb(0.04, 0.12, 0.22).into(),
            perceptual_roughness: 0.08,
            metallic: 0.0,
            opaque_render_method: OpaqueRendererMethod::Auto,
            ..default()
        },
        extension: ShallowWaterSurfaceExtension {
            settings: ShallowWaterMaterialUniform {
                height_scale: 0.08,
                tile_world_size: PATCH_TILE_WORLD,
                grid_size: GRID as f32,
                _pad0: 0.0,
            },
            bed_water,
        },
    };

    let patch_origin = Vec3::new(PATCH_HALF_EXTENT, 0.0, PATCH_HALF_EXTENT);
    commands.spawn((
        Mesh3d(meshes.add(plane)),
        MeshMaterial3d(materials.add(material)),
        Transform::from_translation(patch_origin),
        ShallowWaterSurfaceTag,
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
            color: Color::WHITE,
            illuminance: lux::RAW_SUNLIGHT,
            shadows_enabled: true,
            ..default()
        },
        cascade_shadow_config,
        Transform::from_xyz(16.0, 12.0, 8.0).looking_at(patch_center, Vec3::Y),
    ));

    // Atmosphere + env map lighting and bloom; forward rendering (no deferred / SSR / FXAA).
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(22.0, 18.0, 28.0).looking_at(patch_center, Vec3::Y),
        FreeCamera::default(),
        Atmosphere::earthlike(scattering_mediums.add(ScatteringMedium::default())),
        AtmosphereSettings::default(),
        Exposure { ev100: 12.0 },
        Tonemapping::AcesFitted,
        Bloom::NATURAL,
        AtmosphereEnvironmentMapLight::default(),
    ));
}

fn pointer_to_sim_space(
    mouse: Res<ButtonInput<MouseButton>>,
    egui_wants: Res<EguiWantsInput>,
    window: Query<&Window, With<bevy::window::PrimaryWindow>>,
    camera: Query<(&Camera, &GlobalTransform), With<FreeCamera>>,
    surface: Query<&GlobalTransform, With<ShallowWaterSurfaceTag>>,
    mut controller: ResMut<ShallowWaterController>,
) {
    controller.brush_input_active = false;

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
    let sx = u * controller.cells_x as f32;
    let sy = v * controller.cells_y as f32;

    let new_sim = Vec2::new(sx, sy);
    let paint = mouse.pressed(MouseButton::Right) && !egui_wants.wants_pointer_input();

    if mouse.just_pressed(MouseButton::Right) && paint {
        controller.pointer_prev_sim = new_sim;
        controller.pointer_sim = new_sim;
    } else if paint {
        let prev = controller.pointer_sim;
        controller.pointer_prev_sim = prev;
        controller.pointer_sim = new_sim;
    } else {
        controller.pointer_prev_sim = new_sim;
        controller.pointer_sim = new_sim;
    }

    controller.brush_input_active = paint;
}

fn shallow_water_egui_panel(
    mut contexts: EguiContexts,
    mut controller: ResMut<ShallowWaterController>,
) {
    let Ok(ctx) = contexts.ctx_mut() else {
        return;
    };
    egui::Window::new("Shallow water").show(ctx, |ui| {
        ui.checkbox(&mut controller.paused, "Paused");
        if ui.button("Reset simulation").clicked() {
            controller.sim_apply_serial = controller.sim_apply_serial.wrapping_add(1);
        }
        ui.add(egui::Slider::new(&mut controller.dt, 0.05..=0.5).text("dt"));
        ui.add(egui::Slider::new(&mut controller.gravity, 1.0..=25.0).text("gravity"));
        ui.add(egui::Slider::new(&mut controller.friction, 0.0..=0.5).text("friction"));
        let mut particles = controller.particle_count;
        if ui
            .add(egui::Slider::new(&mut particles, 64..=262_144u32).text("particles"))
            .changed()
        {
            controller.particle_count = round_particle_count(particles);
        }
        ui.separator();
        ui.label("Brush");
        ui.add(egui::Slider::new(&mut controller.brush_radius, 4.0..=48.0).text("radius"));
        ui.add(egui::Slider::new(&mut controller.brush_force, 0.1..=1.5).text("force"));
        ui.radio_value(&mut controller.interaction_mode, 0u32, "None");
        ui.radio_value(&mut controller.interaction_mode, 1u32, "Add bed");
        ui.radio_value(&mut controller.interaction_mode, 2u32, "Remove bed");
        ui.radio_value(&mut controller.interaction_mode, 3u32, "Add water");
        ui.radio_value(&mut controller.interaction_mode, 4u32, "Remove water");
        ui.radio_value(&mut controller.interaction_mode, 5u32, "Drag water");
        ui.separator();
        ui.label("Terrain preset");
        if ui.button("Islands").clicked() {
            controller.preset_index = 0;
            controller.sim_apply_serial = controller.sim_apply_serial.wrapping_add(1);
        }
        if ui.button("River").clicked() {
            controller.preset_index = 1;
            controller.sim_apply_serial = controller.sim_apply_serial.wrapping_add(1);
        }
        if ui.button("Canyon").clicked() {
            controller.preset_index = 2;
            controller.sim_apply_serial = controller.sim_apply_serial.wrapping_add(1);
        }
        if ui.button("Shore").clicked() {
            controller.preset_index = 3;
            controller.sim_apply_serial = controller.sim_apply_serial.wrapping_add(1);
        }
        ui.separator();
        ui.label("PML (open edges)");
        ui.add(egui::Slider::new(&mut controller.pml_width, 0u32..=32u32).text("width"));
        ui.add(egui::Slider::new(&mut controller.pml_eta_rest, 0.0..=8.0).text("Rest depth"));
        ui.add(
            egui::Slider::new(&mut controller.pml_sigma_exponent, 2.0..=3.0).text("Blend sigma"),
        );
        ui.add(egui::Slider::new(&mut controller.pml_cosine_blend, 0.0..=1.0).text("Cosine blend"));
        ui.separator();
        ui.label("Border behavior");
        border_combo(ui, "Left", &mut controller.left_border);
        border_combo(ui, "Right", &mut controller.right_border);
        border_combo(ui, "Bottom", &mut controller.bottom_border);
        border_combo(ui, "Top", &mut controller.top_border);
    });
}

fn border_combo(ui: &mut egui::Ui, label: &str, value: &mut ShallowWaterBorder) {
    ui.horizontal(|ui| {
        ui.label(label);
        egui::ComboBox::from_id_salt(label)
            .selected_text(format!("{value:?}"))
            .show_ui(ui, |ui| {
                ui.selectable_value(value, ShallowWaterBorder::Wall, "Wall");
                ui.selectable_value(value, ShallowWaterBorder::Source, "Source");
                ui.selectable_value(value, ShallowWaterBorder::Drain, "Drain");
                ui.selectable_value(value, ShallowWaterBorder::Waves, "Waves");
            });
    });
}
