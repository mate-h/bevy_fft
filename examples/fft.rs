use bevy::prelude::*;
use bevy_fft::{
    complex::c32,
    fft::{FftPlugin, FftSource, FftTextures},
};
use bevy_render::render_resource::{TextureFormat, TextureUsages};
use image::DynamicImage;

fn main() {
    App::new()
        .add_plugins((DefaultPlugins, FftPlugin))
        .add_systems(Startup, setup)
        .add_systems(Update, (handle_image_load, update_output_sprites))
        .run();
}

#[derive(Component)]
struct FftOutputRe;

#[derive(Component)]
struct FftOutputIm;

// Resource to track the loading state
#[derive(Resource)]
struct LoadingImage {
    handle: Handle<Image>,
}

fn setup(mut commands: Commands, asset_server: Res<AssetServer>) {
    // Camera
    commands.spawn(Camera2d::default());

    // Load source image and store the handle
    let image_handle = asset_server.load("circles_pattern.png");
    commands.insert_resource(LoadingImage {
        handle: image_handle,
    });
}

fn handle_image_load(
    mut commands: Commands,
    loading: Option<Res<LoadingImage>>,
    mut images: ResMut<Assets<Image>>,
    asset_server: Res<AssetServer>,
) {
    let Some(loading) = loading else { return };

    // Check if the image has finished loading
    if let Some(image) = images.get(&loading.handle) {
        let width = image.texture_descriptor.size.width;
        let height = image.texture_descriptor.size.height;

        // First convert to DynamicImage (this will be in RGBA8 format)
        let dynamic_image = image.clone().try_into_dynamic().unwrap();

        // Convert to RGBA32F format
        let float_dynamic_image = DynamicImage::ImageRgba32F(dynamic_image.into_rgba32f());

        // Create new float image
        let mut float_image = Image::from_dynamic(
            float_dynamic_image,
            false, // not sRGB
            image.asset_usage,
        );

        // Set usage flags
        float_image.texture_descriptor.usage = TextureUsages::COPY_DST
            | TextureUsages::COPY_SRC
            | TextureUsages::STORAGE_BINDING
            | TextureUsages::TEXTURE_BINDING;

        // Add the new image to assets
        let float_image_handle = images.add(float_image);

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

        // Spawn FFT entity with the new float image
        commands.spawn((
            FftSource {
                image: float_image_handle.clone(),
                size: UVec2::new(width as u32, height as u32),
                orders: 8,
                padding: UVec2::ZERO,
                roots,
                inverse: false,
            },
            Sprite {
                custom_size: Some(Vec2::new(width as f32, height as f32)),
                image: loading.handle.clone(), // Keep original image for display
                ..default()
            },
            Transform::from_xyz(-300.0, 0.0, 0.0),
        ));

        // Spawn output sprites
        commands.spawn((
            Sprite {
                custom_size: Some(Vec2::new(width as f32, height as f32)),
                ..default()
            },
            Transform::from_xyz(0.0, 0.0, 0.0),
            FftOutputRe,
        ));

        commands.spawn((
            Sprite {
                custom_size: Some(Vec2::new(width as f32, height as f32)),
                ..default()
            },
            Transform::from_xyz(300.0, 0.0, 0.0),
            FftOutputIm,
        ));

        // Remove the loading resource since we're done
        commands.remove_resource::<LoadingImage>();
    }
}

// System to update the output sprite with the FFT texture
fn update_output_sprites(
    fft_query: Query<&FftTextures>,
    mut re_query: Query<&mut Sprite, (With<FftOutputRe>, Without<FftOutputIm>)>,
    mut im_query: Query<&mut Sprite, (With<FftOutputIm>, Without<FftOutputRe>)>,
) {
    if let Ok(fft_textures) = fft_query.get_single() {
        if let Ok(mut re_sprite) = re_query.get_single_mut() {
            re_sprite.image = fft_textures.buffer_d_re.clone();
        }
        if let Ok(mut im_sprite) = im_query.get_single_mut() {
            im_sprite.image = fft_textures.buffer_d_im.clone();
        }
    }
}
