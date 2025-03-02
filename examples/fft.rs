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
struct OutputImage;

// 1. Create an identifier component
#[derive(Component)]
struct GridPosition {
    index: usize,
}

fn setup(mut commands: Commands) {
    // Camera
    commands.spawn(Camera2d::default());

    // Clone the images to release the immutable borrow on 'images'
    let size = Vec2::splat(256.0);
    // Calculate FFT roots
    let mut roots = [c32::new(0.0, 0.0); 8192];
    for order in 0..13 {
        let base = 1 << order;
        let count = base >> 1;
        for k in 0..count {
            let theta = -2.0 * std::f32::consts::PI * (k as f32) / (base as f32);
            let root = c32::new(theta.cos(), theta.sin());
            roots[base + k] = root;
        }
    }

    // Spawn FFT entity
    commands.spawn(FftSource {
        size: size.as_uvec2(),
        orders: 8,
        padding: UVec2::ZERO,
        roots,
        inverse: false,
    });

    let columns = 4;
    let rows = 2;
    let spacing = 256.0 + 8.0;
    let start_x = -((columns - 1) as f32) * spacing / 2.0;
    let start_y = ((rows - 1) as f32) * spacing / 2.0;

    for row in 0..rows {
        for col in 0..columns {
            let index = row * columns + col;
            let x = start_x + col as f32 * spacing;
            let y = start_y - row as f32 * spacing;

            commands.spawn((
                Sprite {
                    custom_size: Some(size),
                    ..default()
                },
                Transform::from_xyz(x, y, 0.0),
                OutputImage,
                GridPosition { index },
            ));
        }
    }
}

// System to update the output sprite with the FFT texture
fn update_output_sprites(
    fft_query: Query<&FftTextures>,
    mut outputs: Query<(&mut Sprite, &GridPosition), With<OutputImage>>,
) {
    if let Ok(fft_textures) = fft_query.get_single() {
        for (mut sprite, grid_pos) in outputs.iter_mut() {
            match grid_pos.index {
                0 => sprite.image = fft_textures.buffer_a_re.clone(),
                1 => sprite.image = fft_textures.buffer_a_im.clone(),
                2 => sprite.image = fft_textures.buffer_b_re.clone(),
                3 => sprite.image = fft_textures.buffer_b_im.clone(),
                4 => sprite.image = fft_textures.buffer_c_re.clone(),
                5 => sprite.image = fft_textures.buffer_c_im.clone(),
                6 => sprite.image = fft_textures.buffer_d_re.clone(),
                7 => sprite.image = fft_textures.buffer_d_im.clone(),
                _ => sprite.image = fft_textures.buffer_a_re.clone(),
            }
        }
    }
}
