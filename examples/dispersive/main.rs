//! Hybrid dispersive (Jeschke–Wojtan style) on a water patch with a sloped beach (GPU init) and FFT surface.
//!
//! Add `FftPlugin` before `DispersivePlugin`.
//! Hold **right-drag** on the surface for boat-style wake strokes (mass flux along motion). Tap **right** for a radial splash.

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

/// Sim texels: disk radius for tap splash on `d_state`.
const SPLASH_RADIUS_TEXELS: i32 = 10;
const SPLASH_PEAK_ADD_H: f32 = 1.5;
/// Clamp for stamped depth; taper repeats toward this cap so stacks stay CFL-friendly.
const SPLASH_MAX_H: f32 = 4.0;
/// Radial outward velocity scale inside tap splash (flux in `.g`/`.b`).
const SPLASH_RADIAL_U: f32 = 0.45;

/// Narrower stamps along drag; mass flux aligns with wake direction Sec. 4.6-style.
const WAKE_RADIUS_TEXELS: i32 = 4;
const WAKE_PEAK_ADD_H: f32 = 0.1;
/// Horizontal velocity scale along wake tangent (multiply by radial falloff `t`).
const WAKE_ALONG_U: f32 = 0.9;

#[derive(Resource, Default)]
struct DispersiveRmbGesture {
    /// Press began on-water; splash here on release only if nothing was dragged.
    anchor: Option<(u32, u32)>,
    /// Last sampled texel while RMB held.
    last: Option<(u32, u32)>,
    /// At least one wake segment was stamped (release skips tap splash).
    stroked: bool,
}

fn main() {
    let mut wgpu = WgpuSettings::default();
    wgpu.features |= WgpuFeatures::FLOAT32_FILTERABLE;

    App::new()
        .insert_resource(DefaultOpaqueRendererMethod::forward())
        .insert_resource(ClearColor(Color::BLACK))
        .insert_resource(GlobalAmbientLight::NONE)
        .init_resource::<SunLightSettings>()
        .init_resource::<DispersiveRmbGesture>()
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
        .add_systems(Startup, (setup, dispersive_help_once))
        .add_systems(
            PostUpdate,
            (
                dispersive_rmb_water
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

fn dispersive_help_once() {
    info!(
        "dispersive: RMB tap = radial splash | hold RMB and drag on the water = wake strokes. LMB or M = camera."
    );
}

fn dispersive_rmb_water(
    mouse: Res<ButtonInput<MouseButton>>,
    egui: Res<EguiWantsInput>,
    window: Query<&Window, With<PrimaryWindow>>,
    camera: Query<(&Camera, &GlobalTransform), With<FreeCamera>>,
    surface: Query<&GlobalTransform, With<DispersiveSurfaceTag>>,
    mut images: ResMut<Assets<Image>>,
    mut controller: ResMut<DispersiveController>,
    mut gesture: ResMut<DispersiveRmbGesture>,
) {
    if controller.paused {
        if mouse.just_released(MouseButton::Right) {
            *gesture = DispersiveRmbGesture::default();
        }
        return;
    }

    let n = controller.n;
    let n1 = n.saturating_sub(1) as i32;
    let cursor_texel = water_texel_under_cursor(&window, &camera, &surface, &egui, n, n1);

    if mouse.just_pressed(MouseButton::Right) {
        if egui.is_pointer_over_area() {
            return;
        }
        gesture.stroked = false;
        gesture.anchor = cursor_texel;
        gesture.last = cursor_texel;
        return;
    }

    if mouse.just_released(MouseButton::Right) {
        if gesture.stroked || egui.is_pointer_over_area() {
            *gesture = DispersiveRmbGesture::default();
            return;
        }
        let Some(ixy) = gesture.anchor else {
            *gesture = DispersiveRmbGesture::default();
            return;
        };
        let Some(img) = images.get_mut(&controller.state) else {
            *gesture = DispersiveRmbGesture::default();
            return;
        };
        stamp_radial_splash_disk(img, ixy.0, ixy.1, n1);
        controller.sim_apply_serial = controller.sim_apply_serial.wrapping_add(1);
        *gesture = DispersiveRmbGesture::default();
        return;
    }

    if !mouse.pressed(MouseButton::Right) || egui.is_pointer_over_area() {
        return;
    }

    let Some(curr) = cursor_texel else {
        *gesture = DispersiveRmbGesture::default();
        return;
    };
    let Some(last) = gesture.last else {
        return;
    };
    if last == curr {
        return;
    }
    let Some(img) = images.get_mut(&controller.state) else {
        return;
    };
    let x0 = last.0 as i32;
    let y0 = last.1 as i32;
    let x1 = curr.0 as i32;
    let y1 = curr.1 as i32;
    let dvx = (x1 - x0) as f32;
    let dvy = (y1 - y0) as f32;
    let len = (dvx * dvx + dvy * dvy).sqrt().max(1e-6);
    let dir = Vec2::new(dvx / len, dvy / len);
    for_each_texel_segment(x0, y0, x1, y1, |ix, iy| {
        stamp_wake_disk(img, ix, iy, dir, n1);
    });
    gesture.last = Some(curr);
    gesture.stroked = true;
    controller.sim_apply_serial = controller.sim_apply_serial.wrapping_add(1);
}

fn water_texel_under_cursor(
    window: &Query<&Window, With<PrimaryWindow>>,
    camera: &Query<(&Camera, &GlobalTransform), With<FreeCamera>>,
    surface: &Query<&GlobalTransform, With<DispersiveSurfaceTag>>,
    egui: &EguiWantsInput,
    n: u32,
    n1: i32,
) -> Option<(u32, u32)> {
    if egui.is_pointer_over_area() {
        return None;
    }
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
}

fn for_each_texel_segment(x0: i32, y0: i32, x1: i32, y1: i32, mut stamp: impl FnMut(i32, i32)) {
    let mut x = x0;
    let mut y = y0;
    let dx = (x1 - x0).abs();
    let dy = (y1 - y0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx - dy;
    loop {
        stamp(x, y);
        if x == x1 && y == y1 {
            break;
        }
        let e2 = 2 * err;
        if e2 > -dy {
            err -= dy;
            x += sx;
        }
        if e2 < dx {
            err += dx;
            y += sy;
        }
    }
}

fn stamp_radial_splash_disk(img: &mut Image, ix: u32, iy: u32, n1: i32) {
    let r2 = (SPLASH_RADIUS_TEXELS * SPLASH_RADIUS_TEXELS) as f32;
    for dy in -SPLASH_RADIUS_TEXELS..=SPLASH_RADIUS_TEXELS {
        for dx in -SPLASH_RADIUS_TEXELS..=SPLASH_RADIUS_TEXELS {
            let s = (dx * dx + dy * dy) as f32;
            if s > r2 {
                continue;
            }
            let t = 1.0 - (s / r2).sqrt();
            let x = (ix as i32 + dx).clamp(0, n1) as u32;
            let y = (iy as i32 + dy).clamp(0, n1) as u32;
            let Ok(c) = img.get_color_at(x, y) else {
                continue;
            };
            let l = c.to_linear();
            let h_old = l.red;
            let qx_old = l.green;
            let qy_old = l.blue;

            let headroom = (SPLASH_MAX_H - h_old).max(0.0);
            let taper = (headroom / SPLASH_MAX_H).min(1.0);
            let add = SPLASH_PEAK_ADD_H * t * t * taper;
            if add < 1e-6 {
                continue;
            }

            let h_new = (h_old + add).min(SPLASH_MAX_H);
            let mix = if h_new > 1e-10 { h_old / h_new } else { 0.0 };
            let mut qx = qx_old * mix;
            let mut qy = qy_old * mix;

            let rx = dx as f32;
            let ry = dy as f32;
            let rlen_sq = rx * rx + ry * ry;
            if rlen_sq > 1e-10 {
                let inv_len = rlen_sq.sqrt().recip();
                let ux = SPLASH_RADIAL_U * t * taper * rx * inv_len;
                let uy = SPLASH_RADIAL_U * t * taper * ry * inv_len;
                qx += h_new * ux;
                qy += h_new * uy;
            }

            let c2 = Color::from(LinearRgba {
                red: h_new,
                green: qx,
                blue: qy,
                alpha: l.alpha,
            });
            let _ = img.set_color_at(x, y, c2);
        }
    }
}

fn stamp_wake_disk(img: &mut Image, cx: i32, cy: i32, dir: Vec2, n1: i32) {
    let r2 = (WAKE_RADIUS_TEXELS * WAKE_RADIUS_TEXELS) as f32;
    for dy in -WAKE_RADIUS_TEXELS..=WAKE_RADIUS_TEXELS {
        for dx in -WAKE_RADIUS_TEXELS..=WAKE_RADIUS_TEXELS {
            let s = (dx * dx + dy * dy) as f32;
            if s > r2 {
                continue;
            }
            let t = 1.0 - (s / r2).sqrt();
            let x = (cx + dx).clamp(0, n1) as u32;
            let y = (cy + dy).clamp(0, n1) as u32;
            let Ok(c) = img.get_color_at(x, y) else {
                continue;
            };
            let l = c.to_linear();
            let h_old = l.red;
            let qx_old = l.green;
            let qy_old = l.blue;

            let headroom = (SPLASH_MAX_H - h_old).max(0.0);
            let taper = (headroom / SPLASH_MAX_H).min(1.0);
            let add = WAKE_PEAK_ADD_H * t * t * taper;
            if add < 1e-6 {
                continue;
            }

            let h_new = (h_old + add).min(SPLASH_MAX_H);
            let mix = if h_new > 1e-10 { h_old / h_new } else { 0.0 };
            let mut qx = qx_old * mix;
            let mut qy = qy_old * mix;

            let ux = WAKE_ALONG_U * t * taper * dir.x;
            let uy = WAKE_ALONG_U * t * taper * dir.y;
            qx += h_new * ux;
            qy += h_new * uy;

            let c2 = Color::from(LinearRgba {
                red: h_new,
                green: qx,
                blue: qy,
                alpha: l.alpha,
            });
            let _ = img.set_color_at(x, y, c2);
        }
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
    controller.dt = 0.025;
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
