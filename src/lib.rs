pub mod complex;
pub mod fft;

use bevy_app::{App, Plugin};
use bevy_render::render_resource::Texture;

pub struct FftPlugin;

impl Plugin for FftPlugin {
    fn build(&self, app: &mut App) {}
}
