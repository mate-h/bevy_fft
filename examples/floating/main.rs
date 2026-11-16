//! Crates, a log and a ball floating on the FFT ocean, moved by Rapier.
//!
//! Each body is voxelized once with [`voxelize`]. [`FloatingBodyPlugin`] evaluates buoyancy and
//! drag on its voxels against the water LOD data on the GPU and reads back one force and torque per
//! body. This example hands them to Rapier as an [`ExternalForce`], with the buoyancy corrected for
//! the frames the readback takes, and feeds Rapier's velocities back for drag. Rapier adds gravity
//! and the collisions between bodies.
//!
//! The panel shows each body's submerged share and drops them all again from above the water.

use std::sync::Arc;

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
    pbr::{AtmosphereSettings, DefaultOpaqueRendererMethod},
    post_process::bloom::Bloom,
    prelude::*,
};
use bevy_egui::{EguiContexts, EguiPlugin, EguiPrimaryContextPass, egui, input::EguiWantsInput};
use bevy_fft::prelude::*;
use bevy_rapier3d::prelude::*;

#[path = "../common/screenshot.rs"]
mod screenshot;

/// World-space tile edge in meters for each FFT cascade, largest first.
const CASCADE_TILES: [f32; 3] = [256.0, 48.0, 9.0];
/// A cascade hands waves to the next smaller one at this many wavelengths per smaller tile.
const CASCADE_HANDOFF_WAVES: f32 = 6.0;
/// Edge of a hull voxel in meters.
const VOXEL: f32 = 0.25;
/// Height above the water the bodies drop from.
const DROP_HEIGHT: f32 = 2.0;

fn main() {
    App::new()
        .insert_resource(DefaultOpaqueRendererMethod::deferred())
        .insert_resource(ClearColor(Color::BLACK))
        .insert_resource(GlobalAmbientLight::NONE)
        .add_plugins((
            screenshot::default_plugins(),
            FreeCameraPlugin,
            FftPlugin,
            OceanPlugin,
            FloatingBodyPlugin,
            RapierPhysicsPlugin::<NoUserData>::default(),
            EguiPlugin::default(),
            screenshot::ScreenshotOnEnvPlugin,
        ))
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (
                spawn_surface_when_ready.after(FftSystemSet::PrepareTextures),
                apply_water_forces,
                sync_free_camera_with_egui_focus,
            ),
        )
        .add_systems(EguiPrimaryContextPass, floating_panel)
        .add_systems(Last, pace_headless_frames)
        .run();
}

/// Holds screenshot runs, which have no window to wait on, to 60 frames per second. Without it
/// the CPU runs dozens of frames ahead of the GPU and the forces arrive that late, where a
/// window's vsync keeps them two or three frames behind.
fn pace_headless_frames(mut last: Local<Option<std::time::Instant>>) {
    if std::env::var_os("BEVY_FFT_SCREENSHOT").is_none() {
        return;
    }
    let frame = std::time::Duration::from_secs_f64(1.0 / 60.0);
    if let Some(previous) = *last {
        let elapsed = previous.elapsed();
        if elapsed < frame {
            std::thread::sleep(frame - elapsed);
        }
    }
    *last = Some(std::time::Instant::now());
}

/// Where a body starts, so the panel can drop it again.
#[derive(Component)]
struct DropFrom(Transform);

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
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
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
                wind_speed: 7.0,
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

    let wood = materials.add(StandardMaterial {
        base_color: Color::srgb(0.55, 0.38, 0.22),
        perceptual_roughness: 0.8,
        ..default()
    });
    let crate_wood = materials.add(StandardMaterial {
        base_color: Color::srgb(0.75, 0.6, 0.35),
        perceptual_roughness: 0.7,
        ..default()
    });
    let rubber = materials.add(StandardMaterial {
        base_color: Color::srgb(0.9, 0.25, 0.1),
        perceptual_roughness: 0.5,
        ..default()
    });

    // Shape, collider, density in kg/m³, material, start position and rotation.
    let log = Quat::from_rotation_z(std::f32::consts::FRAC_PI_2) * Quat::from_rotation_x(0.3);
    let bodies = [
        (
            Mesh::from(Cuboid::new(1.0, 1.0, 1.0)),
            Collider::cuboid(0.5, 0.5, 0.5),
            500.0,
            crate_wood.clone(),
            Vec3::new(-3.0, 0.0, 0.0),
            Quat::IDENTITY,
        ),
        (
            Mesh::from(Cuboid::new(1.2, 0.6, 0.8)),
            Collider::cuboid(0.6, 0.3, 0.4),
            300.0,
            crate_wood,
            Vec3::new(-0.5, 0.0, 2.5),
            Quat::from_rotation_y(0.6),
        ),
        (
            Mesh::from(Cylinder::new(0.3, 4.0)),
            Collider::cylinder(2.0, 0.3),
            600.0,
            wood,
            Vec3::new(1.5, 0.0, -1.0),
            log,
        ),
        (
            Sphere::new(0.5).mesh().ico(3).unwrap(),
            Collider::ball(0.5),
            200.0,
            rubber,
            Vec3::new(3.5, 0.0, 1.5),
            Quat::IDENTITY,
        ),
    ];
    for (mesh, collider, density, material, at, rotation) in bodies {
        let hull = Arc::new(voxelize(&mesh, VOXEL).expect("primitive meshes are closed"));
        let mass = density * hull.volume();
        let mut body = FloatingBody::new(hull, mass);
        // Rapier's velocities drive the drag.
        body.estimate_velocity = false;
        let start = Transform::from_translation(at + Vec3::Y * DROP_HEIGHT).with_rotation(rotation);
        commands.spawn((
            Mesh3d(meshes.add(mesh)),
            MeshMaterial3d(material),
            start,
            DropFrom(start),
            body,
            RigidBody::Dynamic,
            collider,
            ColliderMassProperties::Mass(mass),
            Velocity::default(),
            ExternalForce::default(),
        ));
    }

    let earth_medium = scattering_mediums.add(ScatteringMedium::earth(256, 256));
    commands.spawn(Atmosphere::earth(earth_medium));
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(-2.0, 5.0, -14.0).looking_at(Vec3::new(0.0, 0.0, 0.5), Vec3::Y),
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
        Transform::from_xyz(0.3, 0.35, -1.0).looking_at(Vec3::ZERO, Vec3::Y),
        CascadeShadowConfigBuilder {
            first_cascade_far_bound: 15.0,
            maximum_distance: 200.0,
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

/// Hands the latest water forces to Rapier and Rapier's velocities to the force pass.
fn apply_water_forces(
    settings: Res<FloatingBodySettings>,
    time: Res<Time>,
    mut bodies: Query<(
        &mut FloatingBody,
        &mut ExternalForce,
        &Velocity,
        &GlobalTransform,
    )>,
) {
    for (mut body, mut external, velocity, transform) in &mut bodies {
        body.linear_velocity = velocity.linear;
        body.angular_velocity = velocity.angular;
        let center = transform.transform_point(body.center_of_mass);
        let (force, torque) = body.compensated(
            center,
            velocity.linear,
            velocity.angular,
            time.delta_secs(),
            &settings,
        );
        external.force = force;
        external.torque = torque;
    }
}

fn floating_panel(
    mut contexts: EguiContexts,
    mut bodies: Query<(&FloatingBody, &DropFrom, &mut Transform, &mut Velocity)>,
) {
    let Ok(ctx) = contexts.ctx_mut() else {
        return;
    };
    egui::Window::new("Floating bodies").show(ctx, |ui| {
        let mut drop = false;
        for (body, ..) in &bodies {
            let share = body.submerged_volume / body.hull.volume().max(1e-6);
            ui.label(format!(
                "{:.0} kg: {:.0}% under, {:.1} kN up",
                body.mass,
                100.0 * share,
                body.force.y / 1000.0
            ));
        }
        if ui.button("Drop again").clicked() {
            drop = true;
        }
        if drop {
            for (_, from, mut transform, mut velocity) in &mut bodies {
                *transform = from.0;
                *velocity = Velocity::default();
            }
        }
    });
}

fn sync_free_camera_with_egui_focus(
    mut state: Query<&mut FreeCameraState, With<FreeCamera>>,
    egui_wants: Res<EguiWantsInput>,
) {
    if let Ok(mut camera_state) = state.single_mut() {
        camera_state.enabled = !egui_wants.wants_any_input();
    }
}
