//! A motor boat on the FFT ocean.
//!
//! The hull is lofted in `examples/common/boat_hull.rs` and voxelized once.
//! [`FloatingBodyPlugin`] evaluates the water's forces on its voxels on the GPU, and Rapier
//! integrates the boat with them, as in the `floating` example.
//!
//! A chase camera follows the boat, and `F` switches to the free camera. The water LOD data is
//! centred on the camera, and the finest slice the hull's voxels sample is only a few meters
//! wide, so the chase camera keeps close enough for the boat to feel the finest waves.
//!
//! - `BOAT_WIND=<m/s>` sets the wind speed of the sea (default 7).
//! - `BOAT_CHASE=<m>` sets the chase distance.
//! - `BOAT_LOG=1` prints the boat's height, roll, pitch and forces once a second
//!   (`BOAT_LOG_EVERY=<s>` to change that), and the readback latency of the forces.
//!
//! Moving the chase camera from 8 m to 30 m made no measurable difference to the boat's heave
//! and roll in a 7 m/s sea, so the LOD slices around the camera reach the hull at those distances.
//! The free camera can fly much further, and then the boat feels coarser water.

use std::sync::Arc;

use bevy::{
    anti_alias::fxaa::Fxaa,
    camera::Exposure,
    camera_controller::free_camera::{FreeCamera, FreeCameraPlugin, FreeCameraState},
    core_pipeline::{
        prepass::{DeferredPrepass, DepthPrepass},
        tonemapping::Tonemapping,
    },
    input::mouse::AccumulatedMouseScroll,
    light::{
        Atmosphere, AtmosphereEnvironmentMapLight, CascadeShadowConfigBuilder, GlobalAmbientLight,
        atmosphere::ScatteringMedium, light_consts::lux,
    },
    material::OpaqueRendererMethod,
    pbr::{AtmosphereSettings, DefaultOpaqueRendererMethod},
    post_process::bloom::Bloom,
    prelude::*,
    transform::TransformSystems,
};
use bevy_egui::{EguiContexts, EguiPlugin, EguiPrimaryContextPass, egui, input::EguiWantsInput};
use bevy_fft::prelude::*;
use bevy_rapier3d::prelude::*;

#[path = "../common/boat_hull.rs"]
mod boat_hull;
#[path = "../common/screenshot.rs"]
mod screenshot;

/// World-space tile edge in meters for each FFT cascade, largest first.
const CASCADE_TILES: [f32; 3] = [256.0, 48.0, 9.0];
/// A cascade hands waves to the next smaller one at this many wavelengths per smaller tile.
const CASCADE_HANDOFF_WAVES: f32 = 6.0;
/// Edge of a hull voxel in meters.
const VOXEL: f32 = 0.2;
/// Sea water density in kg/m³.
const SEA_WATER: f32 = 1025.0;
/// Readbacks of the water forces to wait for before the boat is let go.
const SETTLE_READBACKS: u32 = 30;
/// Chase camera distance behind the boat and its bounds, in meters. Beyond about 20 m the hull
/// samples coarser water LOD slices than the finest.
const CHASE_DISTANCE: f32 = 14.0;
const CHASE_MIN: f32 = 6.0;
const CHASE_MAX: f32 = 30.0;

fn main() {
    App::new()
        .insert_resource(DefaultOpaqueRendererMethod::deferred())
        .insert_resource(ClearColor(Color::BLACK))
        .insert_resource(GlobalAmbientLight::NONE)
        .insert_resource(Chase {
            distance: env_f32("BOAT_CHASE").unwrap_or(CHASE_DISTANCE),
            free: false,
        })
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
                (toggle_camera, reset_boat, log_boat),
            ),
        )
        .add_systems(
            PostUpdate,
            chase_camera
                .after(PhysicsSet::Writeback)
                .before(TransformSystems::Propagate),
        )
        .add_systems(EguiPrimaryContextPass, boat_panel)
        .add_systems(Last, pace_headless_frames)
        .run();
}

fn env_f32(name: &str) -> Option<f32> {
    std::env::var(name).ok()?.parse().ok()
}

/// The boat the camera follows and the helm steers.
#[derive(Component)]
struct Boat;

/// Chase camera state.
#[derive(Resource)]
struct Chase {
    /// Distance behind the boat in meters.
    distance: f32,
    /// The free camera has control.
    free: bool,
}

/// Holds screenshot runs, which have no window to wait on, to 60 frames per second, and to the
/// GPU. Without a window's vsync the CPU runs as far ahead of a busy GPU as Bevy lets it, and the
/// water forces then arrive up to 24 frames, 0.4 s, late: no hull stays afloat on forces that old.
/// Waiting for the GPU each frame keeps them two or three frames late, as a window does.
fn pace_headless_frames(
    device: Res<bevy::render::renderer::RenderDevice>,
    mut last: Local<Option<std::time::Instant>>,
) {
    if std::env::var_os("BEVY_FFT_SCREENSHOT").is_none() {
        return;
    }
    let _ = device.poll(bevy::render::render_resource::PollType::Wait {
        submission_index: None,
        timeout: Some(std::time::Duration::from_millis(200)),
    });
    let frame = std::time::Duration::from_secs_f64(1.0 / 60.0);
    if let Some(previous) = *last {
        let elapsed = previous.elapsed();
        if elapsed < frame {
            std::thread::sleep(frame - elapsed);
        }
    }
    *last = Some(std::time::Instant::now());
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
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut scattering_mediums: ResMut<Assets<ScatteringMedium>>,
) {
    let n = 256u32;
    let wind_speed = env_f32("BOAT_WIND").unwrap_or(7.0);
    for (index, &tile) in CASCADE_TILES.iter().enumerate() {
        let (band_k_min, band_k_max) = cascade_band(index);
        commands.spawn((
            FftSource::square_inverse_only(n),
            OceanSimSettings {
                texture_size: n,
                tile_size: tile,
                band_k_min,
                band_k_max,
                wind_speed,
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

    spawn_boat(&mut commands, &mut meshes, &mut materials);

    let earth_medium = scattering_mediums.add(ScatteringMedium::earth(256, 256));
    commands.spawn(Atmosphere::earth(earth_medium));
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 5.0, CHASE_DISTANCE).looking_at(Vec3::ZERO, Vec3::Y),
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
            first_cascade_far_bound: 20.0,
            maximum_distance: 300.0,
            ..default()
        }
        .build(),
    ));
}

/// The boat: hull, deck house and windscreen, floating at rest on its design waterline.
fn spawn_boat(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
) {
    let hull = boat_hull::build_hull(boat_hull::HullShape::default(), VOXEL, SEA_WATER);
    let shape = hull.shape;
    info!(
        "boat hull: {:.2} m³ displaced, {:.0} kg, block coefficient {:.2}",
        hull.displacement,
        hull.mass,
        hull.displacement / (shape.length * shape.beam * shape.draft)
    );
    let collider = Collider::convex_hull(&hull.points).expect("the hull has volume");
    let mut body = FloatingBody::new(Arc::new(hull.voxels), hull.mass);
    body.center_of_mass = hull.center_of_mass;
    // Rapier's velocities drive the drag.
    body.estimate_velocity = false;

    let paint = materials.add(StandardMaterial {
        base_color: Color::srgb(0.92, 0.93, 0.95),
        perceptual_roughness: 0.35,
        ..default()
    });
    let teak = materials.add(StandardMaterial {
        base_color: Color::srgb(0.55, 0.4, 0.26),
        perceptual_roughness: 0.7,
        ..default()
    });
    let glass = materials.add(StandardMaterial {
        base_color: Color::srgb(0.05, 0.07, 0.09),
        perceptual_roughness: 0.05,
        reflectance: 0.6,
        ..default()
    });

    let deck_y = shape.freeboard;
    commands
        .spawn((
            Boat,
            Name::new("Boat"),
            Mesh3d(meshes.add(hull.mesh)),
            MeshMaterial3d(paint.clone()),
            Transform::default(),
            body,
            RigidBody::Dynamic,
            collider,
            ColliderMassProperties::MassProperties(MassProperties {
                local_center_of_mass: hull.center_of_mass,
                mass: hull.mass,
                principal_inertia_local_frame: Quat::IDENTITY,
                principal_inertia: hull.inertia,
            }),
            Velocity::default(),
            ExternalForce::default(),
            Sleeping::disabled(),
        ))
        .with_children(|boat| {
            // Teak deck just above the hull's own deck, so the two don't fight.
            boat.spawn((
                Mesh3d(meshes.add(Cuboid::new(1.9, 0.02, 3.2))),
                MeshMaterial3d(teak),
                Transform::from_xyz(0.0, deck_y + 0.02, 1.4),
            ));
            // Deck house, a little forward of midships, with a sloped windscreen.
            boat.spawn((
                Mesh3d(meshes.add(Cuboid::new(1.7, 0.9, 1.8))),
                MeshMaterial3d(paint),
                Transform::from_xyz(0.0, deck_y + 0.45, -0.6),
            ));
            boat.spawn((
                Mesh3d(meshes.add(Cuboid::new(1.6, 0.6, 0.03))),
                MeshMaterial3d(glass),
                Transform::from_xyz(0.0, deck_y + 0.65, -1.55)
                    .with_rotation(Quat::from_rotation_x(-0.5)),
            ));
        });
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
        &mut RigidBody,
        &mut Transform,
    )>,
    mut last_frame: Local<u32>,
    mut history: Local<std::collections::VecDeque<(u64, Vec3)>>,
    mut counter: Local<u64>,
    mut latency: Local<(u32, u64, u32)>,
) {
    *counter += 1;
    for (mut body, mut external, velocity, transform, mut rigid, mut held) in &mut bodies {
        // Hold the boat until the force pass has caught up, so it doesn't sink while the
        // pipelines compile, and meanwhile move it up or down to where it displaces its own
        // weight, so it isn't let go in a trough or under a crest.
        let ready = body.frame > SETTLE_READBACKS;
        if !ready && body.frame > 0 && body.waterplane_area > 0.0 {
            let excess = body.submerged_volume - body.mass / SEA_WATER;
            let step = (excess / body.waterplane_area).clamp(-0.05, 0.05);
            if body.frame != *last_frame {
                *last_frame = body.frame;
                held.translation.y += step;
            }
        }
        let kind = if ready {
            RigidBody::Dynamic
        } else {
            RigidBody::Fixed
        };
        if *rigid != kind {
            *rigid = kind;
        }
        body.linear_velocity = velocity.linear;
        body.angular_velocity = velocity.angular;
        let center = transform.transform_point(body.center_of_mass);
        if std::env::var_os("BOAT_LOG").is_some() {
            history.push_back((*counter, center));
            if history.len() > 120 {
                history.pop_front();
            }
            if body.frame != latency.0 {
                latency.0 = body.frame;
                if let Some(&(f, _)) = history.iter().min_by(|a, b| {
                    a.1.distance_squared(body.sampled_center)
                        .total_cmp(&b.1.distance_squared(body.sampled_center))
                }) {
                    latency.1 += *counter - f;
                    latency.2 += 1;
                }
            }
            if latency.2 >= 60 {
                println!(
                    "readback latency {:.1} frames",
                    latency.1 as f32 / latency.2 as f32
                );
                latency.1 = 0;
                latency.2 = 0;
            }
        }
        let (force, torque) = body.compensated(
            center,
            velocity.linear,
            velocity.angular,
            time.delta_secs(),
            &settings,
        );
        let (damp_force, damp_torque) = radiation_damping(
            &body,
            transform.rotation(),
            velocity.linear,
            velocity.angular,
        );
        let (force, torque) = with_added_mass(
            &body,
            &settings,
            transform.rotation(),
            force + damp_force,
            torque + damp_torque,
        );
        external.force = force;
        external.torque = torque;
    }
}

/// Added mass in heave and added inertia in pitch and roll, as shares of the boat's own.
///
/// A hull that heaves or pitches has to move water with it. Strip theory gives about twice the
/// boat's mass in heave for this hull, and the same order in pitch; roll moves much less water.
const ADDED_MASS: Vec3 = Vec3::new(1.0, 1.5, 0.25);

/// Scales the water's net vertical force and its pitch and roll torques to act on the boat plus
/// the water it carries. Rapier knows only the boat's own mass, and gravity acts on that alone,
/// so the net vertical force, buoyancy and drag minus weight, is divided by the total mass
/// share. Surge, sway and yaw move little water and keep their forces.
///
/// Without it the boat heaves and pitches several times too fast, it bounces off waves, and the
/// forces, which arrive a few frames late, get out of step with its motion.
fn with_added_mass(
    body: &FloatingBody,
    settings: &FloatingBodySettings,
    rotation: Quat,
    force: Vec3,
    torque: Vec3,
) -> (Vec3, Vec3) {
    let wet = (body.submerged_volume * SEA_WATER / body.mass.max(1.0)).clamp(0.0, 1.0);
    let share = Vec3::ONE / (Vec3::ONE + ADDED_MASS * wet);
    let weight = body.mass * settings.gravity;
    let force = force.with_y(weight + (force.y - weight) * share.y);
    let local = rotation.inverse() * torque;
    let local = Vec3::new(local.x * share.x, local.y, local.z * share.z);
    (force, rotation * local)
}

/// Damping rates of heave, roll and pitch in 1/s, from about 15–25% of critical damping for this
/// hull.
const HEAVE_DAMPING: f32 = 1.5;
const ROLL_DAMPING: f32 = 1.5;
const PITCH_DAMPING: f32 = 2.0;

/// Damping of heave, roll and pitch that stands in for wave radiation.
///
/// A real hull that heaves, rolls or pitches makes waves, and the energy they carry off damps
/// the motion. The voxel forces have no such term, and their restoring torque arrives a few frames
/// late, which pumps energy into the motion instead. Without this a boat at rest in calm water
/// starts pitching within seconds and soon flips.
fn radiation_damping(
    body: &FloatingBody,
    rotation: Quat,
    linear: Vec3,
    angular: Vec3,
) -> (Vec3, Vec3) {
    let submerged = (body.submerged_volume * SEA_WATER / body.mass.max(1.0)).clamp(0.0, 1.0);
    let force = -Vec3::Y * body.mass * HEAVE_DAMPING * linear.y * submerged;
    // Roll and pitch rates in the boat's frame, from the hull's moments of inertia.
    let local = rotation.inverse() * angular;
    let inertia = body.mass * boat_hull::inertia_per_kg(&boat_hull::HullShape::default());
    let torque_local = -Vec3::new(
        inertia.x * PITCH_DAMPING * local.x,
        0.0,
        inertia.z * ROLL_DAMPING * local.z,
    ) * submerged;
    (force, rotation * torque_local)
}

/// `F` hands the camera between the chase and the free camera.
fn toggle_camera(
    keys: Res<ButtonInput<KeyCode>>,
    egui_wants: Res<EguiWantsInput>,
    mut chase: ResMut<Chase>,
    mut state: Query<&mut FreeCameraState, With<FreeCamera>>,
) {
    if keys.just_pressed(KeyCode::KeyF) && !egui_wants.wants_any_keyboard_input() {
        chase.free = !chase.free;
    }
    if let Ok(mut camera_state) = state.single_mut() {
        camera_state.enabled = chase.free && !egui_wants.wants_any_input();
    }
}

/// `R` sets the boat upright where it is, at rest.
fn reset_boat(
    keys: Res<ButtonInput<KeyCode>>,
    egui_wants: Res<EguiWantsInput>,
    mut boat: Query<(&mut Transform, &mut Velocity), With<Boat>>,
) {
    if !keys.just_pressed(KeyCode::KeyR) || egui_wants.wants_any_keyboard_input() {
        return;
    }
    let Ok((mut transform, mut velocity)) = boat.single_mut() else {
        return;
    };
    let forward = (transform.rotation * Vec3::NEG_Z).with_y(0.0);
    let yaw = (-forward.x).atan2(-forward.z);
    transform.translation.y = 0.0;
    transform.rotation = Quat::from_rotation_y(yaw);
    *velocity = Velocity::default();
}

/// Keeps the camera behind and above the boat, on a spring, levelled against the boat's roll and
/// pitch. The mouse wheel sets the distance.
fn chase_camera(
    time: Res<Time>,
    scroll: Res<AccumulatedMouseScroll>,
    egui_wants: Res<EguiWantsInput>,
    mut chase: ResMut<Chase>,
    boat: Query<&Transform, (With<Boat>, Without<Camera3d>)>,
    mut camera: Query<&mut Transform, With<Camera3d>>,
) {
    if chase.free {
        return;
    }
    if !egui_wants.is_pointer_over_area() {
        chase.distance =
            (chase.distance * (1.0 - 0.1 * scroll.delta.y)).clamp(CHASE_MIN, CHASE_MAX);
    }
    let (Ok(boat), Ok(mut camera)) = (boat.single(), camera.single_mut()) else {
        return;
    };
    let forward = (boat.rotation * Vec3::NEG_Z)
        .with_y(0.0)
        .normalize_or(Vec3::NEG_Z);
    let target = boat.translation.with_y(0.0) + forward * 4.0 + Vec3::Y * 1.0;
    let eye = boat.translation.with_y(0.0) - forward * chase.distance
        + Vec3::Y * (1.5 + 0.3 * chase.distance);
    // A critically damped follow with a time constant of about half a second.
    let blend = 1.0 - (-time.delta_secs() * 4.0).exp();
    camera.translation = camera.translation.lerp(eye, blend);
    camera.look_at(target, Vec3::Y);
}

/// With `BOAT_LOG=1`, prints the boat's mean height, heave range, roll and pitch once a second.
fn log_boat(
    time: Res<Time>,
    boat: Query<(&Transform, &FloatingBody), With<Boat>>,
    mut window: Local<Vec<Vec3>>,
    mut next: Local<f32>,
) {
    if std::env::var_os("BOAT_LOG").is_none() {
        return;
    }
    let Ok((transform, body)) = boat.single() else {
        return;
    };
    let (yaw, pitch, roll) = transform.rotation.to_euler(EulerRot::YXZ);
    let _ = yaw;
    window.push(Vec3::new(transform.translation.y, roll, pitch));
    if time.elapsed_secs() < *next {
        return;
    }
    *next = time.elapsed_secs() + env_f32("BOAT_LOG_EVERY").unwrap_or(1.0);
    let n = window.len().max(1) as f32;
    let mean = window.iter().copied().sum::<Vec3>() / n;
    let (lo, hi) = window.iter().fold((f32::MAX, f32::MIN), |(lo, hi), v| {
        (lo.min(v.x), hi.max(v.x))
    });
    let roll = window
        .iter()
        .map(|v| v.y.abs())
        .fold(0.0, f32::max)
        .to_degrees();
    let pitch = window
        .iter()
        .map(|v| v.z.abs())
        .fold(0.0, f32::max)
        .to_degrees();
    println!(
        "t {:6.1}  y {:+.3} ({:+.3}..{:+.3})  roll<{:4.1}°  pitch<{:4.1}°  submerged {:.2} m³  frame {}  buoy {:.0} drag {:.0} N  torque {:.0}",
        time.elapsed_secs(),
        mean.x,
        lo,
        hi,
        roll,
        pitch,
        body.submerged_volume,
        body.frame,
        body.buoyancy,
        body.drag,
        body.torque
    );
    window.clear();
}

fn boat_panel(mut contexts: EguiContexts, chase: Res<Chase>) {
    let Ok(ctx) = contexts.ctx_mut() else {
        return;
    };
    egui::Window::new("Boat").show(ctx, |ui| {
        ui.label(if chase.free {
            "F: back to the chase camera"
        } else {
            "F: free camera. Wheel: distance. R: set upright"
        });
    });
}
