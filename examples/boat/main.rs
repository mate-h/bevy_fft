//! A motor boat on the FFT ocean.
//!
//! The hull is lofted in `examples/common/boat_hull.rs` and voxelized once.
//! [`FloatingBodyPlugin`] evaluates the water's forces on its voxels on the GPU, and Rapier
//! integrates the boat with them, as in the `floating` example.
//!
//! A propeller drives it, and a rudder and a skeg steer it and keep it running straight
//! (`examples/common/boat_drive.rs`). W and S, or the up and down arrows, move the throttle lever;
//! A and D, or the left and right arrows, hold the rudder over; Space stops the engine; R sets the
//! boat upright.
//!
//! A chase camera follows the boat, and `F` switches to the free camera. The water LOD data is
//! centred on the camera, and the finest slice the hull's voxels sample is only a few meters
//! wide, so the chase camera keeps close enough for the boat to feel the finest waves.
//!
//! - `BOAT_WIND=<m/s>` sets the wind speed of the sea (default 7).
//! - `BOAT_CHASE=<m>` sets the chase distance.
//! - `BOAT_TOW=<m/s>` holds the boat at that speed straight ahead, for measuring drag.
//! - `BOAT_TOP=<m>` looks down on the boat and its wake from that height.
//! - `BOAT_WAKE_CELL=<m>` sets the wake patch's cell width (default 0.5).
//! - `BOAT_HELM=<throttle>,<rudder degrees>` holds the helm, for runs without a keyboard.
//! - `BOAT_LOG=1` prints the boat's height, roll, pitch and forces once a second
//!   (`BOAT_LOG_EVERY=<s>` to change that), and the readback latency of the forces.
//!
//! Moving the chase camera from 8 m to 30 m made no measurable difference to the boat's heave
//! and roll in a 7 m/s sea, so the LOD slices around the camera reach the hull at those distances.
//! The free camera can fly much further, and then the boat feels coarser water.

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
use bevy_fft::{particles::WaterParticlesPlugin, prelude::*};
use bevy_rapier3d::prelude::*;
use boat::{Boat, BoatDrive, SEA_WATER};

#[path = "../common/boat.rs"]
mod boat;
#[path = "../common/boat_drive.rs"]
mod boat_drive;
#[path = "../common/boat_hull.rs"]
mod boat_hull;
#[path = "../common/screenshot.rs"]
mod screenshot;

/// World-space tile edge in meters for each FFT cascade, largest first.
const CASCADE_TILES: [f32; 3] = [256.0, 48.0, 9.0];
/// A cascade hands waves to the next smaller one at this many wavelengths per smaller tile.
const CASCADE_HANDOFF_WAVES: f32 = 6.0;
/// Chase camera distance from the point it looks at, and its bounds, in meters.
const CHASE_DISTANCE: f32 = 20.0;
/// How far the chase camera looks down, in degrees.
const CHASE_ELEVATION_DEG: f32 = 28.0;
/// The chase camera looks at a point this share of its distance behind the boat.
const CHASE_AIM_BEHIND: f32 = 0.15;
const CHASE_MIN: f32 = 6.0;
const CHASE_MAX: f32 = 30.0;

fn main() {
    App::new()
        .insert_resource(DefaultOpaqueRendererMethod::deferred())
        .insert_resource(ClearColor(Color::BLACK))
        .insert_resource(GlobalAmbientLight::NONE)
        .init_resource::<WakeCapture>()
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
            DispersivePlugin,
            WaterParticlesPlugin,
            RapierPhysicsPlugin::<NoUserData>::default(),
            boat::BoatPhysicsPlugin,
            EguiPlugin::default(),
            screenshot::ScreenshotOnEnvPlugin,
        ))
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (
                spawn_surface_when_ready.after(FftSystemSet::PrepareTextures),
                (toggle_camera, reset_boat, log_boat, log_latency, tow_boat),
                read_helm.before(boat::apply_drive),
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

/// The wake patch's state, read back for `BOAT_LOG`.
#[derive(Resource, Default)]
struct WakeCapture(Vec<u8>);

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
    mut images: ResMut<Assets<Image>>,
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

    // The wake patch: deep-water waves the hull presses into the surface, following the boat.
    let wake = boat::wake_patch(
        &mut commands,
        &mut images,
        env_f32("BOAT_WAKE_CELL").unwrap_or(boat::WAKE_CELL),
    );
    if std::env::var_os("BOAT_LOG").is_some() {
        commands
            .spawn(bevy::render::gpu_readback::Readback::texture(
                wake.state.clone(),
            ))
            .observe(
                |event: On<bevy::render::gpu_readback::ReadbackComplete>,
                 mut capture: ResMut<WakeCapture>| {
                    capture.0 = event.data.clone();
                },
            );
    }
    commands.insert_resource(wake);

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
    let hull = boat_hull::build_hull(boat_hull::HullShape::default(), boat::VOXEL, SEA_WATER);
    let shape = hull.shape;
    info!(
        "boat hull: {:.2} m³ displaced, {:.0} kg, block coefficient {:.2}",
        hull.displacement,
        hull.mass,
        hull.displacement / (shape.length * shape.beam * shape.draft)
    );
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
            boat::boat_physics(&hull),
            Name::new("Boat"),
            Mesh3d(meshes.add(hull.mesh)),
            MeshMaterial3d(paint.clone()),
            Transform::default(),
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
    if let Some(height) = env_f32("BOAT_TOP") {
        // A look down on the boat and its wake from above and a little behind.
        let forward = (boat.rotation * Vec3::NEG_Z)
            .with_y(0.0)
            .normalize_or(Vec3::NEG_Z);
        camera.translation = boat.translation.with_y(height) - forward * 0.6 * height;
        camera.look_at(
            boat.translation.with_y(0.0) - forward * 0.25 * height,
            Vec3::Y,
        );
        return;
    }
    let forward = (boat.rotation * Vec3::NEG_Z)
        .with_y(0.0)
        .normalize_or(Vec3::NEG_Z);
    // Aimed a little behind the boat and looking down on it, so the boat sits above the middle
    // of the picture and its wake runs down the lower half toward the camera.
    let elevation = CHASE_ELEVATION_DEG.to_radians();
    let target = boat.translation.with_y(0.0) - forward * CHASE_AIM_BEHIND * chase.distance;
    let eye = target - forward * chase.distance * elevation.cos()
        + Vec3::Y * chase.distance * elevation.sin();
    // A critically damped follow with a time constant of about half a second.
    let blend = 1.0 - (-time.delta_secs() * 4.0).exp();
    camera.translation = camera.translation.lerp(eye, blend);
    camera.look_at(target, Vec3::Y);
}

/// With `BOAT_LOG=1`, prints the boat's mean height, heave range, roll and pitch once a second.
#[allow(clippy::too_many_arguments)]
fn log_boat(
    time: Res<Time>,
    wake: Res<WakeCapture>,
    controller: Res<DispersiveController>,
    helm: Query<&BoatDrive>,
    boat: Query<(&Transform, &FloatingBody), With<Boat>>,
    mut window: Local<Vec<Vec3>>,
    mut drag: Local<(f32, f32)>,
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
    let forward = (transform.rotation * Vec3::NEG_Z)
        .with_y(0.0)
        .normalize_or(Vec3::NEG_Z);
    drag.0 += body.drag.dot(forward);
    drag.1 += 1.0;
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
    if !wake.0.is_empty() {
        // Surface deviation of the patch's cells, against the patch's median as the rest level:
        // the dent under the hull and the largest wave more than 10 m from it.
        let n = controller.n as usize;
        let texel = wake.0.len() / (n * n);
        let h: Vec<f32> = (0..n * n)
            .map(|i| f32::from_le_bytes(wake.0[i * texel..i * texel + 4].try_into().unwrap()))
            .collect();
        let mut sorted = h.clone();
        sorted.sort_by(f32::total_cmp);
        let rest = sorted[sorted.len() / 2];
        let at = controller.world_to_cells(transform.translation.xz());
        let cell = controller.cell_size();
        let (mut dent, mut wave) = (0.0f32, 0.0f32);
        for (i, &v) in h.iter().enumerate() {
            let d = Vec2::new((i % n) as f32 + 0.5, (i / n) as f32 + 0.5).distance(at) * cell;
            if d < 2.0 {
                dent = dent.min(v - rest);
            } else if d > 10.0 {
                wave = wave.max((v - rest).abs());
            }
        }
        println!("  wake: dent {dent:+.3} m, largest wave beyond 10 m {wave:.3} m");
    }
    if let Ok(drive) = helm.single() {
        let forward = (transform.rotation * Vec3::NEG_Z)
            .with_y(0.0)
            .normalize_or(Vec3::NEG_Z);
        let heading = forward.x.atan2(-forward.z).to_degrees().rem_euclid(360.0);
        println!(
            "  speed {:.2} m/s, heading {:5.1}°, at ({:.0}, {:.0}), throttle {:.2}, rudder {:+.0}°",
            body.sampled_velocity.dot(forward),
            heading,
            transform.translation.x,
            transform.translation.z,
            drive.helm.throttle,
            drive.helm.rudder.to_degrees()
        );
    }
    println!("  mean drag ahead {:.0} N", drag.0 / drag.1.max(1.0));
    *drag = (0.0, 0.0);
    window.clear();
}

/// Speed, helm and heading, with the controls, and a tuning section for the drive, the sea and
/// the water forces.
#[allow(clippy::type_complexity)]
fn boat_panel(
    mut contexts: EguiContexts,
    chase: Res<Chase>,
    mut settings: ResMut<FloatingBodySettings>,
    mut boat: Query<(&mut BoatDrive, &Transform, &Velocity, &FloatingBody)>,
    mut cascades: Query<&mut OceanSimSettings>,
) {
    let Ok(ctx) = contexts.ctx_mut() else {
        return;
    };
    let Ok((mut drive, transform, velocity, body)) = boat.single_mut() else {
        return;
    };
    egui::Window::new("Boat").show(ctx, |ui| {
        let forward = (transform.rotation * Vec3::NEG_Z).with_y(0.0).normalize_or(Vec3::NEG_Z);
        let speed = velocity.linear.dot(forward);
        let heading = forward.x.atan2(-forward.z).to_degrees().rem_euclid(360.0);
        ui.heading(format!("{:.1} kn", speed * 1.943_844));
        ui.label(format!("{speed:.2} m/s, heading {heading:03.0}°"));
        ui.add(
            egui::ProgressBar::new(drive.helm.throttle.abs())
                .text(format!("throttle {:+.0}%", drive.helm.throttle * 100.0)),
        );
        let rudder = drive.helm.rudder.to_degrees();
        ui.label(format!(
            "rudder {:.0}° {}",
            rudder.abs(),
            if rudder > 0.5 {
                "starboard"
            } else if rudder < -0.5 {
                "port"
            } else {
                ""
            }
        ));
        let wet = body.submerged_volume * SEA_WATER / body.mass.max(1.0);
        ui.label(format!("displacement {:.0}%", wet * 100.0));
        ui.separator();
        ui.label(if chase.free {
            "F: back to the chase camera"
        } else {
            "W/S: throttle. A/D: rudder. Space: stop.\nR: set upright. F: free camera. Wheel: distance."
        });
        ui.collapsing("Tuning", |ui| {
            let params = &mut drive.params;
            ui.add(egui::Slider::new(&mut params.max_thrust, 500.0..=12000.0).text("Thrust N"));
            ui.add(egui::Slider::new(&mut params.rudder_area, 0.02..=0.5).text("Rudder m²"));
            ui.add(egui::Slider::new(&mut params.skeg_area, 0.0..=1.0).text("Skeg m²"));
            let mut max_rudder = params.max_rudder.to_degrees();
            ui.add(egui::Slider::new(&mut max_rudder, 5.0..=45.0).text("Rudder max °"));
            params.max_rudder = max_rudder.to_radians();
            ui.separator();
            let mut wind = cascades.iter().next().map_or(0.0, |c| c.wind_speed);
            if ui
                .add(egui::Slider::new(&mut wind, 0.0..=14.0).text("Wind m/s"))
                .changed()
            {
                for mut cascade in &mut cascades {
                    cascade.wind_speed = wind;
                }
            }
            ui.add(
                egui::Slider::new(&mut settings.pressure_quadratic, 0.0..=2000.0)
                    .text("Pressure drag"),
            );
            ui.add(
                egui::Slider::new(&mut settings.splash.rate, 0.0..=8000.0).text("Splash rate"),
            );
        });
    });
}

/// With `BOAT_TOW=<m/s>`, holds the boat's horizontal velocity at that speed straight ahead, so
/// the log shows the drag at that speed. Heave, pitch and roll stay free.
fn tow_boat(mut boat: Query<(&Transform, &mut Velocity, &RigidBody), With<Boat>>) {
    let Some(speed) = env_f32("BOAT_TOW") else {
        return;
    };
    let Ok((transform, mut velocity, rigid)) = boat.single_mut() else {
        return;
    };
    if *rigid != RigidBody::Dynamic {
        return;
    }
    let forward = (transform.rotation * Vec3::NEG_Z)
        .with_y(0.0)
        .normalize_or(Vec3::NEG_Z);
    velocity.linear = (forward * speed).with_y(velocity.linear.y);
    velocity.angular.y = 0.0;
}

/// W and S (or the arrows) move the throttle lever, A and D hold the rudder over, and Space
/// centres both. `BOAT_HELM=<throttle>,<rudder degrees>` holds the helm there instead.
fn read_helm(
    time: Res<Time>,
    keys: Res<ButtonInput<KeyCode>>,
    egui_wants: Res<EguiWantsInput>,
    chase: Res<Chase>,
    mut boat: Query<&mut BoatDrive>,
) {
    let Ok(mut drive) = boat.single_mut() else {
        return;
    };
    let dt = time.delta_secs();
    let scripted = std::env::var("BOAT_HELM").ok().and_then(|v| {
        let (t, r) = v.split_once(',')?;
        Some((t.parse::<f32>().ok()?, r.parse::<f32>().ok()?.to_radians()))
    });
    let (lever, rudder) = if let Some(helm) = scripted {
        helm
    } else if chase.free || egui_wants.wants_any_keyboard_input() {
        (drive.lever, 0.0)
    } else {
        let pressed = |a: KeyCode, b: KeyCode| keys.pressed(a) || keys.pressed(b);
        let mut lever = drive.lever;
        if pressed(KeyCode::KeyW, KeyCode::ArrowUp) {
            lever += drive.params.throttle_rate * dt;
        }
        if pressed(KeyCode::KeyS, KeyCode::ArrowDown) {
            lever -= drive.params.throttle_rate * dt;
        }
        if keys.just_pressed(KeyCode::Space) {
            lever = 0.0;
        }
        let mut rudder = 0.0;
        if pressed(KeyCode::KeyD, KeyCode::ArrowRight) {
            rudder += drive.params.max_rudder;
        }
        if pressed(KeyCode::KeyA, KeyCode::ArrowLeft) {
            rudder -= drive.params.max_rudder;
        }
        (lever.clamp(-drive.params.reverse, 1.0), rudder)
    };
    drive.lever = lever;
    drive.rudder = rudder;
}

/// With `BOAT_LOG`, prints how many frames the water forces take to come back, once a second.
/// It finds the frame whose centre of mass the forces were computed for.
fn log_latency(
    boat: Query<(&FloatingBody, &GlobalTransform), With<Boat>>,
    mut history: Local<std::collections::VecDeque<(u64, Vec3)>>,
    mut counter: Local<u64>,
    mut latency: Local<(u32, u64, u32)>,
) {
    if std::env::var_os("BOAT_LOG").is_none() {
        return;
    }
    let Ok((body, transform)) = boat.single() else {
        return;
    };
    *counter += 1;
    history.push_back((*counter, transform.transform_point(body.center_of_mass)));
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
        *latency = (latency.0, 0, 0);
    }
}
