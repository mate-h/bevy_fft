//! Opt-in screenshot for example runs.
//!
//! Set `BEVY_FFT_SCREENSHOT=path/to/frame.png` to save one frame after a warm-up and then exit.
//! `BEVY_FFT_SCREENSHOT_FRAMES` sets the warm-up length in frames (default 600). Without the
//! variable the plugin does nothing.
//!
//! Cameras that draw to the primary window draw to an offscreen image of the same size instead,
//! and the capture reads that image. A covered or unfocused window can present all-black frames,
//! so this keeps captures independent of what is on screen. The app keeps updating at full rate
//! while unfocused. The window must stay visible, since macOS stops driving the frame loop of a
//! hidden window. Black captures are still discarded and retried a few times before the app gives
//! up and exits.

use bevy::camera::{ImageRenderTarget, RenderTarget};
use bevy::prelude::*;
use bevy::render::render_resource::TextureFormat;
use bevy::render::view::screenshot::{Screenshot, ScreenshotCaptured, save_to_disk};
use bevy::window::{PrimaryWindow, WindowRef};
use bevy::winit::{UpdateMode, WinitSettings};

pub struct ScreenshotOnEnvPlugin;

/// Frames between a black capture and the next attempt.
const RETRY_FRAMES: u32 = 60;
/// Attempts before giving up on a non-black frame.
const MAX_ATTEMPTS: u32 = 10;
/// Frames to keep running after a save so the file reaches the disk.
const EXIT_DELAY_FRAMES: u32 = 30;

#[derive(Resource)]
struct ScreenshotRequest {
    path: String,
    frames_left: u32,
    attempts: u32,
    in_flight: bool,
    saved: bool,
    exit_after: u32,
    target: Option<Handle<Image>>,
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
        app.insert_resource(ScreenshotRequest {
            path,
            frames_left,
            attempts: 0,
            in_flight: false,
            saved: false,
            exit_after: EXIT_DELAY_FRAMES,
            target: None,
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

/// Points every camera that targets the primary window at one offscreen image of the window's size.
fn render_window_cameras_offscreen(
    mut request: ResMut<ScreenshotRequest>,
    mut images: ResMut<Assets<Image>>,
    windows: Query<(Entity, &Window), With<PrimaryWindow>>,
    mut targets: Query<&mut RenderTarget, With<Camera>>,
) {
    let Ok((primary, window)) = windows.single() else {
        return;
    };
    for mut target in &mut targets {
        let RenderTarget::Window(window_ref) = &*target else {
            continue;
        };
        let on_primary = match window_ref {
            WindowRef::Primary => true,
            WindowRef::Entity(e) => *e == primary,
        };
        if !on_primary {
            continue;
        }
        let handle = request
            .target
            .get_or_insert_with(|| {
                images.add(Image::new_target_texture(
                    window.physical_width().max(1),
                    window.physical_height().max(1),
                    TextureFormat::Rgba8UnormSrgb,
                    None,
                ))
            })
            .clone();
        *target = RenderTarget::Image(ImageRenderTarget::from(handle));
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
    let screenshot = match &request.target {
        Some(image) => Screenshot::image(image.clone()),
        None => Screenshot::primary_window(),
    };
    commands.spawn(screenshot).observe(
        |captured: On<ScreenshotCaptured>, mut request: ResMut<ScreenshotRequest>| {
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
            let mut save = save_to_disk(request.path.clone());
            save(captured);
            request.saved = true;
        },
    );
}
