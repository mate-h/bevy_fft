use bevy::{log, prelude::*};
use bevy_fft::{
    complex::image::ComplexImage,
    fft::{FftPlugin, FftSettings},
};

fn main() {
    App::new()
        .add_plugins((DefaultPlugins, FftPlugin))
        .add_systems(Startup, setup)
        .run();
}

#[derive(Resource)]
struct ComplexImages {
    images: Vec<ComplexImage>,
}

fn setup(mut commands: Commands, asset_server: Res<AssetServer>) {
    // Camera
    commands.spawn(Camera2d::default());

    // Load source image
    let image_handle = asset_server.load("tiles.png");

    // Create complex image from source
    let complex_image = ComplexImage {
        source: image_handle.clone(),
    };

    commands.insert_resource(ComplexImages {
        images: vec![complex_image],
    });

    commands.spawn((
        FftSettings {
            size: UVec2::new(256, 256),
            orders: 8,
            padding: UVec2::ZERO,
        },
        Sprite {
            custom_size: Some(Vec2::new(256.0, 256.0)),
            image: image_handle,
            ..default()
        },
        Transform::from_xyz(0.0, 0.0, 0.0),
    ));
}
