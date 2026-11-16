//! Floating bodies in shallow water: a crate dropped into a channel, a log towed across it, and a
//! log lying across it as a dam, moved by Rapier.
//!
//! A shallow water grid of 0.25 m cells fills a 64 m channel with a river that flows in at its
//! upstream end and leaves at normal depth downstream, both from a [`RiverSpline`]. Every body
//! is voxelized once. [`FloatingBodyPlugin`] presses each hull into the grid through a pressure
//! head (Kellomäki 2014), so a body pushes water aside and makes waves, and evaluates buoyancy and
//! drag on its voxels against the same surface. The crate is dynamic: it splashes in and floats
//! downstream. It presses the water only while it moves up or down faster than 0.6 m/s, since a
//! free body that keeps pressing it bobs without end. The towed log is kinematic and drags a wake
//! back and forth across the channel. The dam log is fixed on the bed, and the river pours over
//! it.
//!
//! A fountain on the near bank sprays water particles, until they spawn from the water itself.
//!
//! The panel drops the crate again and shows its submerged share.

use std::sync::Arc;

use bevy::{
    anti_alias::fxaa::Fxaa,
    camera::Exposure,
    camera_controller::free_camera::{FreeCamera, FreeCameraPlugin, FreeCameraState},
    core_pipeline::tonemapping::Tonemapping,
    light::{
        Atmosphere, AtmosphereEnvironmentMapLight, CascadeShadowConfigBuilder, GlobalAmbientLight,
        atmosphere::ScatteringMedium, light_consts::lux,
    },
    material::OpaqueRendererMethod,
    pbr::AtmosphereSettings,
    post_process::bloom::Bloom,
    prelude::*,
};
use bevy_egui::{EguiContexts, EguiPlugin, EguiPrimaryContextPass, egui, input::EguiWantsInput};
use bevy_fft::{particles::*, prelude::*, shallow_water::*};
use bevy_rapier3d::prelude::*;

#[path = "../common/screenshot.rs"]
mod screenshot;

/// Channel length along X and width along Z in meters.
const LENGTH: f32 = 64.0;
const WIDTH: f32 = 32.0;
const CELL: f32 = 0.25;
/// Bed slope along +x, and the discharge the river carries in m³/s.
const SLOPE: f32 = 0.0001;
const DISCHARGE: f32 = 20.0;
const MANNING_N: f32 = 0.03;
/// Still depth the channel starts at, about the river's normal depth.
const DEPTH: f32 = 1.5;
/// Edge of a hull voxel in meters.
const VOXEL: f32 = 0.2;
/// Vertical speed above which the crate presses the water.
const CRATE_SPLASH_SPEED: f32 = 0.6;
/// Where the crate drops from.
const CRATE_DROP: Vec3 = Vec3::new(-18.0, 3.0, 6.0);
/// The towed log runs across the channel at this x, between these z, at this speed.
const TOW_X: f32 = -4.0;
const TOW_REACH: f32 = 11.0;
const TOW_SPEED: f32 = 2.0;
/// The dam log lies across the channel at this x.
const DAM_X: f32 = 16.0;
const DAM_RADIUS: f32 = 0.35;

fn main() {
    App::new()
        .insert_resource(ClearColor(Color::BLACK))
        .insert_resource(GlobalAmbientLight::NONE)
        // No ocean writes the LOD data, so it starts each frame from still water.
        .insert_resource(WaterLodClear)
        .insert_resource(WaterLodSettings {
            slice_count: 6,
            resolution: 512,
            base_texel_size: 0.125,
        })
        .add_plugins((
            screenshot::default_plugins(),
            FreeCameraPlugin,
            ShallowWaterPlugin,
            FloatingBodyPlugin,
            WaterParticlesPlugin,
            RapierPhysicsPlugin::<NoUserData>::default(),
            EguiPlugin::default(),
            screenshot::ScreenshotOnEnvPlugin,
        ))
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (
                apply_water_forces,
                tow_log,
                sync_free_camera_with_egui_focus,
            ),
        )
        .add_systems(EguiPrimaryContextPass, pool_panel)
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

#[derive(Component)]
struct Crate;

#[derive(Component)]
struct TowedLog;

/// Grid cells along x and z.
fn cells() -> UVec2 {
    (Vec2::new(LENGTH, WIDTH) / CELL).round().as_uvec2()
}

/// Bed height in solver meters at world `x`. The solver's zero sits at the downstream bed.
fn bed(x: f32) -> f32 {
    SLOPE * (0.5 * LENGTH - x)
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut water_materials: ResMut<Assets<WaterSurfaceMaterial>>,
    mut images: ResMut<Assets<Image>>,
    mut scattering_mediums: ResMut<Assets<ScatteringMedium>>,
) {
    let cells = cells();
    let mut controller = ShallowWaterController::new(&mut images, cells.x, cells.y);
    let placement = ShallowWaterPlacement {
        center: Vec2::ZERO,
        cell_size: CELL,
        // The still surface at the channel's middle sits at world height 0.
        datum: -(DEPTH + bed(0.0)),
        blend_cells: 0.0,
    };
    let heights: Vec<f32> = (0..cells.x * cells.y)
        .map(|i| bed(((i % cells.x) as f32 + 0.5) * CELL - 0.5 * LENGTH))
        .collect();
    controller.bed_source = images.add(bed_source_image(cells.x, cells.y, &heights));
    controller.preset_index = SHALLOW_WATER_PRESET_BED_SOURCE;
    controller.fill_level = DEPTH + bed(0.0);
    controller.gravity = 9.81;
    controller.dt = 1.0 / 60.0;
    controller.manning_n = MANNING_N;
    controller.pressure_head = Some(images.add(shallow_r32_image(cells.x, cells.y)));
    controller.placement = Some(placement);
    let mut profile = ShallowWaterBorderProfile::new(cells);
    profile.add_river(
        &RiverSpline {
            points: vec![
                Vec2::new(-0.5 * LENGTH - 10.0, 0.0),
                Vec2::new(0.5 * LENGTH + 10.0, 0.0),
            ],
            width: WIDTH,
            discharge: DISCHARGE,
            outflow_slope: SLOPE,
        },
        &placement,
    );
    controller.border_profile = Some(images.add(profile.image()));
    commands.insert_resource(controller);

    commands.spawn((
        WaterQuadtree::default(),
        MeshMaterial3d(water_materials.add(WaterSurfaceMaterial {
            base: StandardMaterial {
                base_color: Color::WHITE,
                specular_transmission: 1.0,
                ior: 1.333,
                reflectance: 0.356,
                perceptual_roughness: 0.08,
                metallic: 0.0,
                opaque_render_method: OpaqueRendererMethod::Forward,
                ..default()
            },
            extension: WaterSurfaceExtension {
                optics: WaterOpticsUniform {
                    enabled: 1,
                    ..default()
                },
                ..default()
            },
        })),
        WaterSurfaceTag,
        bevy::light::NotShadowCaster,
    ));

    // The channel bed, and banks that rise above the water around it.
    let floor_y = placement.datum;
    let gravel = materials.add(StandardMaterial {
        base_color: Color::srgb(0.45, 0.42, 0.36),
        perceptual_roughness: 0.95,
        ..default()
    });
    let grass = materials.add(StandardMaterial {
        base_color: Color::srgb(0.25, 0.38, 0.16),
        perceptual_roughness: 0.9,
        ..default()
    });
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(LENGTH, 0.2, WIDTH))),
        MeshMaterial3d(gravel),
        Transform::from_xyz(0.0, floor_y - 0.1, 0.0),
        RigidBody::Fixed,
        Collider::cuboid(0.5 * LENGTH, 0.1, 0.5 * WIDTH),
    ));
    let bank_top = 0.6;
    let bank_height = bank_top - floor_y + 0.2;
    for (size, at) in [
        (
            Vec2::new(LENGTH + 80.0, 40.0),
            Vec2::new(0.0, 0.5 * WIDTH + 20.0),
        ),
        (
            Vec2::new(LENGTH + 80.0, 40.0),
            Vec2::new(0.0, -0.5 * WIDTH - 20.0),
        ),
        (Vec2::new(40.0, WIDTH), Vec2::new(-0.5 * LENGTH - 20.0, 0.0)),
        (Vec2::new(40.0, WIDTH), Vec2::new(0.5 * LENGTH + 20.0, 0.0)),
    ] {
        commands.spawn((
            Mesh3d(meshes.add(Cuboid::new(size.x, bank_height, size.y))),
            MeshMaterial3d(grass.clone()),
            Transform::from_xyz(at.x, bank_top - 0.5 * bank_height, at.y),
            RigidBody::Fixed,
            Collider::cuboid(0.5 * size.x, 0.5 * bank_height, 0.5 * size.y),
        ));
    }

    let wood = materials.add(StandardMaterial {
        base_color: Color::srgb(0.5, 0.34, 0.2),
        perceptual_roughness: 0.8,
        ..default()
    });
    let crate_wood = materials.add(StandardMaterial {
        base_color: Color::srgb(0.75, 0.6, 0.35),
        perceptual_roughness: 0.7,
        ..default()
    });
    let along_z = Quat::from_rotation_x(std::f32::consts::FRAC_PI_2);
    let body = |mesh: &Mesh, density: f32| {
        let hull = Arc::new(voxelize(mesh, VOXEL).expect("primitive meshes are closed"));
        let mass = density * hull.volume();
        let mut body = FloatingBody::new(hull, mass);
        body.estimate_velocity = false;
        (body, mass)
    };

    let crate_mesh = Mesh::from(Cuboid::new(1.2, 1.2, 1.2));
    let (crate_body, mass) = body(&crate_mesh, 450.0);
    commands.spawn((
        Crate,
        Mesh3d(meshes.add(crate_mesh)),
        MeshMaterial3d(crate_wood),
        Transform::from_translation(CRATE_DROP).with_rotation(Quat::from_rotation_y(0.4)),
        crate_body,
        RigidBody::Dynamic,
        Collider::cuboid(0.6, 0.6, 0.6),
        ColliderMassProperties::Mass(mass),
        Velocity::default(),
        ExternalForce::default(),
    ));

    let towed_mesh = Mesh::from(Cylinder::new(0.35, 5.0));
    let (towed_body, _) = body(&towed_mesh, 600.0);
    commands.spawn((
        TowedLog,
        Mesh3d(meshes.add(towed_mesh)),
        MeshMaterial3d(wood.clone()),
        // Lying along x, half under, moving across the channel.
        Transform::from_xyz(TOW_X, -0.05, -TOW_REACH)
            .with_rotation(Quat::from_rotation_z(std::f32::consts::FRAC_PI_2)),
        towed_body,
        RigidBody::KinematicVelocityBased,
        Collider::cylinder(2.5, 0.35),
        Velocity::linear(Vec3::Z * TOW_SPEED),
        ExternalForce::default(),
    ));

    let dam_mesh = Mesh::from(Cylinder::new(DAM_RADIUS, WIDTH));
    let (dam_body, _) = body(&dam_mesh, 700.0);
    commands.spawn((
        Mesh3d(meshes.add(dam_mesh)),
        MeshMaterial3d(wood),
        Transform::from_xyz(DAM_X, floor_y + DAM_RADIUS, 0.0).with_rotation(along_z),
        dam_body,
        RigidBody::Fixed,
        Collider::cylinder(0.5 * WIDTH, DAM_RADIUS),
        Velocity::default(),
        ExternalForce::default(),
    ));

    // A fountain on the near bank, until the particles spawn from the water itself.
    commands.spawn((
        Transform::from_xyz(-20.0, 0.2, -0.5 * WIDTH + 1.0),
        WaterParticleEmitter {
            velocity: Vec3::new(1.0, 7.0, 3.5),
            spread: 0.8,
            rate: 3000.0,
            lifetime: 3.0,
            size: 0.06,
            ..default()
        },
    ));

    let earth_medium = scattering_mediums.add(ScatteringMedium::earth(256, 256));
    commands.spawn(Atmosphere::earth(earth_medium));
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(-30.0, 14.0, -26.0).looking_at(Vec3::new(-2.0, -1.0, 2.0), Vec3::Y),
        FreeCamera::default(),
        WaterViewer,
        AtmosphereSettings::default(),
        Exposure { ev100: 12.0 },
        Tonemapping::AcesFitted,
        Bloom::NATURAL,
        AtmosphereEnvironmentMapLight::default(),
        Msaa::Off,
        Fxaa::default(),
    ));
    commands.spawn((
        DirectionalLight {
            shadow_maps_enabled: true,
            illuminance: lux::RAW_SUNLIGHT,
            ..default()
        },
        Transform::from_xyz(0.4, 0.6, -0.5).looking_at(Vec3::ZERO, Vec3::Y),
        CascadeShadowConfigBuilder {
            first_cascade_far_bound: 20.0,
            maximum_distance: 150.0,
            ..default()
        }
        .build(),
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
        Has<Crate>,
    )>,
) {
    for (mut body, mut external, velocity, transform, is_crate) in &mut bodies {
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
        // A free body that presses the water while it bobs feeds its own heave back through the
        // water a few frames late and never settles. The crate presses only while it moves up or
        // down fast, so it splashes when it lands and then floats on the surface it reads.
        if is_crate {
            body.displaces_water = velocity.linear.y.abs() > CRATE_SPLASH_SPEED;
        }
        external.torque = torque;
    }
}

/// Turns the towed log around at the banks.
fn tow_log(mut logs: Query<(&Transform, &mut Velocity), With<TowedLog>>) {
    for (transform, mut velocity) in &mut logs {
        let z = transform.translation.z;
        if z > TOW_REACH && velocity.linear.z > 0.0 || z < -TOW_REACH && velocity.linear.z < 0.0 {
            velocity.linear.z = -velocity.linear.z;
        }
    }
}

fn pool_panel(
    mut contexts: EguiContexts,
    mut crates: Query<(&FloatingBody, &mut Transform, &mut Velocity), With<Crate>>,
    mut controller: ResMut<ShallowWaterController>,
) {
    let Ok(ctx) = contexts.ctx_mut() else {
        return;
    };
    egui::Window::new("Pool").show(ctx, |ui| {
        for (body, mut transform, mut velocity) in &mut crates {
            let share = body.submerged_volume / body.hull.volume().max(1e-6);
            ui.label(format!(
                "Crate {:.0} kg: {:.0}% under",
                body.mass,
                100.0 * share
            ));
            if ui.button("Drop the crate").clicked() {
                *transform = Transform::from_translation(CRATE_DROP);
                *velocity = Velocity::default();
            }
        }
        ui.add(egui::Slider::new(&mut controller.manning_n, 0.0..=0.08).text("Manning n"));
        if ui.button("Reset water").clicked() {
            controller.sim_apply_serial += 1;
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
