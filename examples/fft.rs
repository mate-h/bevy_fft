use bevy::prelude::*;
use bevy_fft::{
    complex::c32,
    fft::{FftPlugin, FftSource},
};

fn main() {
    App::new()
        .add_plugins((DefaultPlugins, FftPlugin))
        .add_systems(Startup, setup)
        .run();
}

fn setup(mut commands: Commands, asset_server: Res<AssetServer>) {
    // Camera
    commands.spawn(Camera2d::default());

    // Load source image
    let image_handle = asset_server.load("tiles.png");

    // Calculate FFT roots
    let mut roots = [c32::new(0.0, 0.0); 8192];
    for i in 0..8192 {
        let theta = -2.0 * std::f32::consts::PI * (i as f32) / 8192.0;
        roots[i] = c32::new(theta.cos(), theta.sin());
    }

    // Spawn entity with FFT components
    commands.spawn((
        FftSource {
            image: image_handle.clone(),
            size: UVec2::new(256, 256),
            orders: 8,
            padding: UVec2::ZERO,
            roots,
        },
        Sprite {
            custom_size: Some(Vec2::new(256.0, 256.0)),
            image: image_handle,
            ..default()
        },
        Transform::from_xyz(0.0, 0.0, 0.0),
    ));
}
