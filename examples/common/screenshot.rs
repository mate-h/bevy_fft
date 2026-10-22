//! Opt-in screenshot for example runs.
//!
//! Set `BEVY_FFT_SCREENSHOT=path/to/frame.png` to save one frame after a warm-up and then exit.
//! `BEVY_FFT_SCREENSHOT_FRAMES` sets the warm-up length in frames (default 600). Without the
//! variable the plugin does nothing.
//!
//! A window that is hidden or in the background can present all-black frames on some platforms.
//! Black captures are discarded and retried a few times before the app gives up and exits.

use bevy::prelude::*;
use bevy::render::view::screenshot::{Screenshot, ScreenshotCaptured, save_to_disk};

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
        })
        .add_systems(Update, take_screenshot_then_exit);
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
    commands.spawn(Screenshot::primary_window()).observe(
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
