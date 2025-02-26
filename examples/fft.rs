use bevy::prelude::*;
use bevy_fft::{
    complex::c32,
    fft::{FftPlugin, FftSource, FftTextures},
};

fn main() {
    App::new()
        .add_plugins((DefaultPlugins, FftPlugin))
        .add_systems(Startup, setup)
        .add_systems(Update, update_output_sprites)
        .run();
}

#[derive(Component)]
struct FftOutputRe;

#[derive(Component)]
struct FftOutputIm;

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

    // Spawn source image entity with FFT components
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
        Transform::from_xyz(-300.0, 0.0, 0.0),
    ));

    // Spawn output sprite entity (will be updated with FFT texture)
    commands.spawn((
        Sprite {
            custom_size: Some(Vec2::new(256.0, 256.0)),
            ..default()
        },
        Transform::from_xyz(0.0, 0.0, 0.0),
        FftOutputRe,
    ));

    commands.spawn((
        Sprite {
            custom_size: Some(Vec2::new(256.0, 256.0)),
            ..default()
        },
        Transform::from_xyz(150.0, 0.0, 0.0),
        FftOutputIm,
    ));
}

// System to update the output sprite with the FFT texture
fn update_output_sprites(
    fft_query: Query<&FftTextures>,
    mut re_query: Query<&mut Sprite, (With<FftOutputRe>, Without<FftOutputIm>)>,
    mut im_query: Query<&mut Sprite, (With<FftOutputIm>, Without<FftOutputRe>)>,
) {
    if let Ok(fft_textures) = fft_query.get_single() {
        if let Ok(mut re_sprite) = re_query.get_single_mut() {
            re_sprite.image = fft_textures.re.clone();
        }
        if let Ok(mut im_sprite) = im_query.get_single_mut() {
            im_sprite.image = fft_textures.im.clone();
        }
    }
}
