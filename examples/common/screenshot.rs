//! Opt-in screenshot for example runs.
//!
//! Set `BEVY_FFT_SCREENSHOT=path/to/frame.png` to save one frame after a warm-up and then exit.
//! `BEVY_FFT_SCREENSHOT_FRAMES` sets the warm-up length in frames (default 600), and
//! `BEVY_FFT_SCREENSHOT_SIZE` sets the image size as `WIDTHxHEIGHT` (default 2560x1440).
//! `BEVY_FFT_SCREENSHOT_COUNT` saves that many frames in a row, named `stem_0.ext`, `stem_1.ext`,
//! and so on. Without the first variable the plugin does nothing.
//!
//! Screenshot runs open no window, so they work in the background without anything popping up.
//! Use [`default_plugins`] in place of `DefaultPlugins` for that. Cameras that would draw to the
//! primary window draw to an offscreen image instead, the capture reads that image, and the app
//! updates continuously. Black captures are discarded and retried a few times before the app gives
//! up and exits.

use bevy::app::PluginGroupBuilder;
use bevy::camera::{ImageRenderTarget, RenderTarget};
use bevy::prelude::*;
use bevy::render::render_resource::TextureFormat;
use bevy::render::view::screenshot::{Screenshot, ScreenshotCaptured, save_to_disk};
use bevy::window::{ExitCondition, WindowRef};
use bevy::winit::{UpdateMode, WinitSettings};

pub struct ScreenshotOnEnvPlugin;

/// Frames between a black capture and the next attempt.
const RETRY_FRAMES: u32 = 60;
/// Attempts before giving up on a non-black frame.
const MAX_ATTEMPTS: u32 = 10;
/// Frames to keep running after a save so the file reaches the disk.
const EXIT_DELAY_FRAMES: u32 = 30;

fn screenshot_requested() -> bool {
    std::env::var_os("BEVY_FFT_SCREENSHOT").is_some()
}

/// `DefaultPlugins`, without a primary window during screenshot runs.
pub fn default_plugins() -> PluginGroupBuilder {
    let plugins = DefaultPlugins.build();
    if !screenshot_requested() {
        return plugins;
    }
    plugins.set(WindowPlugin {
        primary_window: None,
        exit_condition: ExitCondition::DontExit,
        ..default()
    })
}

fn screenshot_size() -> UVec2 {
    std::env::var("BEVY_FFT_SCREENSHOT_SIZE")
        .ok()
        .and_then(|v| {
            let (w, h) = v.split_once('x')?;
            Some(UVec2::new(w.parse().ok()?, h.parse().ok()?))
        })
        .unwrap_or(UVec2::new(2560, 1440))
        .max(UVec2::ONE)
}

#[derive(Resource)]
struct ScreenshotRequest {
    path: String,
    frames_left: u32,
    attempts: u32,
    in_flight: bool,
    saved: bool,
    exit_after: u32,
    target: Handle<Image>,
    /// How many frames to save, counting the one about to be captured.
    burst_left: u32,
    burst_index: u32,
    burst_total: u32,
}

impl Plugin for ScreenshotOnEnvPlugin {
    fn build(&self, app: &mut App) {
        let Ok(path) = std::env::var("BEVY_FFT_SCREENSHOT") else {
            return;
        };
        let frames_left = std::env::var("BEVY_FFT_SCREENSHOT_FRAMES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(600);
        let burst_total = std::env::var("BEVY_FFT_SCREENSHOT_COUNT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1u32)
            .max(1);
        let size = screenshot_size();
        let target =
            app.world_mut()
                .resource_mut::<Assets<Image>>()
                .add(Image::new_target_texture(
                    size.x,
                    size.y,
                    TextureFormat::Rgba8UnormSrgb,
                    None,
                ));
        app.insert_resource(ScreenshotRequest {
            path,
            frames_left,
            attempts: 0,
            in_flight: false,
            saved: false,
            exit_after: EXIT_DELAY_FRAMES,
            target,
            burst_left: burst_total,
            burst_index: 0,
            burst_total,
        })
        .insert_resource(WinitSettings {
            focused_mode: UpdateMode::Continuous,
            unfocused_mode: UpdateMode::Continuous,
        })
        .add_systems(
            Update,
            (render_window_cameras_offscreen, take_screenshot_then_exit).chain(),
        );
    }
}

/// Points every camera that targets the primary window at the offscreen image.
fn render_window_cameras_offscreen(
    request: Res<ScreenshotRequest>,
    mut targets: Query<&mut RenderTarget, With<Camera>>,
) {
    for mut target in &mut targets {
        if matches!(&*target, RenderTarget::Window(WindowRef::Primary)) {
            *target = RenderTarget::Image(ImageRenderTarget::from(request.target.clone()));
        }
    }
}

fn take_screenshot_then_exit(
    mut commands: Commands,
    mut request: ResMut<ScreenshotRequest>,
    mut exit: MessageWriter<AppExit>,
) {
    if request.saved || request.attempts >= MAX_ATTEMPTS {
        if request.exit_after == 0 {
            if !request.saved {
                error!("No non-black frame after {MAX_ATTEMPTS} screenshot attempts");
            }
            exit.write(if request.saved {
                AppExit::Success
            } else {
                AppExit::error()
            });
        } else {
            request.exit_after -= 1;
        }
        return;
    }
    if request.in_flight {
        return;
    }
    if request.frames_left > 0 {
        request.frames_left -= 1;
        return;
    }
    request.in_flight = true;
    request.attempts += 1;
    let path = burst_path(&request.path, request.burst_index, request.burst_total);
    eprintln!("screenshot capture {path}");
    commands
        .spawn(Screenshot::image(request.target.clone()))
        .observe(
            move |captured: On<ScreenshotCaptured>, mut request: ResMut<ScreenshotRequest>| {
                request.in_flight = false;
                let black = captured
                    .image
                    .data
                    .as_ref()
                    .is_none_or(|d| d.chunks(4).all(|px| px.iter().take(3).all(|&c| c == 0)));
                if black {
                    request.frames_left = RETRY_FRAMES;
                    return;
                }
                let mut save = save_to_disk(path.clone());
                save(captured);
                request.burst_index += 1;
                request.burst_left = request.burst_left.saturating_sub(1);
                if request.burst_left > 0 {
                    request.frames_left = 0;
                    request.attempts = 0;
                    return;
                }
                request.saved = true;
            },
        );
}

/// `path` unchanged for a single frame, otherwise `stem_0.ext`, `stem_1.ext`, and so on.
fn burst_path(path: &str, index: u32, total: u32) -> String {
    if total <= 1 {
        return path.to_string();
    }
    let path = std::path::Path::new(path);
    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("frame");
    let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("png");
    path.parent()
        .unwrap_or_else(|| std::path::Path::new(""))
        .join(format!("{stem}_{index}.{ext}"))
        .to_string_lossy()
        .into_owned()
}
