//! Opt-in screenshot for example runs.
//!
//! Set `BEVY_FFT_SCREENSHOT=path/to/frame.png` to save one frame after a warm-up and then exit.
//! `BEVY_FFT_SCREENSHOT_FRAMES` sets the warm-up length in frames (default 600). Without the
//! variable the plugin does nothing.

use bevy::prelude::*;
use bevy::render::view::screenshot::{Screenshot, save_to_disk};

pub struct ScreenshotOnEnvPlugin;

#[derive(Resource)]
struct ScreenshotRequest {
    path: String,
    frames_left: u32,
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
            exit_after: 30,
        })
        .add_systems(Update, take_screenshot_then_exit);
    }
}

fn take_screenshot_then_exit(
    mut commands: Commands,
    mut request: ResMut<ScreenshotRequest>,
    mut exit: MessageWriter<AppExit>,
) {
    if request.frames_left > 0 {
        request.frames_left -= 1;
        if request.frames_left == 0 {
            commands
                .spawn(Screenshot::primary_window())
                .observe(save_to_disk(request.path.clone()));
        }
        return;
    }
    // Give the capture a few frames to reach the disk before exiting.
    if request.exit_after == 0 {
        exit.write(AppExit::Success);
    } else {
        request.exit_after -= 1;
    }
}
