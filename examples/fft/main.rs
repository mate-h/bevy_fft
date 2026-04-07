//! Round-trip FFT and IFFT on a 256x256 texture with a radial band-pass filter.

mod band_pass;
mod pattern;
mod ui;

use band_pass::{BandPassParams, BandPassPlugin};
use bevy::{
    asset::RenderAssetUsages,
    image::Image,
    prelude::*,
    render::render_resource::{Extent3d, TextureDimension, TextureFormat, TextureUsages},
};
use bevy_fft::fft::prelude::*;
use pattern::{InputPatternTextures, PatternPlugin, switch_pattern};
use ui::{BandPassUiPlugin, FftDemoStartup};

fn main() {
    App::new()
        .add_plugins((
            DefaultPlugins,
            FftPlugin,
            PatternPlugin,
            BandPassPlugin,
            BandPassUiPlugin,
        ))
        .add_systems(Startup, setup.in_set(FftDemoStartup::Scene))
        .add_systems(
            Update,
            (
                attach_input_preview.after(prepare_fft_textures),
                bind_demo_sprites.after(attach_input_preview),
                switch_pattern,
            ),
        )
        .run();
}

#[derive(Component)]
struct DemoSprite(u8);

fn setup(mut commands: Commands) {
    commands.spawn(Camera2d);
    commands.spawn((
        FftSource::grid_256_forward_then_inverse(),
        BandPassParams {
            band_center: 0.37,
            band_width: 0.5,
        },
    ));

    let tile = 256.0;
    let gap = 16.0;
    let step = tile + gap;
    let start = -step;

    for i in 0u8..3 {
        commands.spawn((
            Sprite {
                custom_size: Some(Vec2::splat(tile)),
                ..default()
            },
            Transform::from_xyz(start + step * i as f32, 0.0, 0.0),
            DemoSprite(i),
        ));
    }
}

fn attach_input_preview(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    query: Query<(Entity, &FftSource, &FftTextures), Without<InputPatternTextures>>,
) {
    for (entity, source, _) in &query {
        let extent = Extent3d {
            width: source.size.x,
            height: source.size.y,
            depth_or_array_layers: 1,
        };
        let mut image = Image::new_fill(
            extent,
            TextureDimension::D2,
            &[0; 16],
            TextureFormat::Rgba32Float,
            RenderAssetUsages::default(),
        );
        image.texture_descriptor.usage = TextureUsages::COPY_DST | TextureUsages::TEXTURE_BINDING;

        let re = images.add(image);
        commands.entity(entity).insert(InputPatternTextures { re });
    }
}

fn bind_demo_sprites(
    fft: Query<(&FftTextures, &InputPatternTextures)>,
    mut sprites: Query<(&mut Sprite, &DemoSprite)>,
) {
    let Ok((tex, input)) = fft.single() else {
        return;
    };
    for (mut sprite, DemoSprite(i)) in &mut sprites {
        sprite.image = match *i {
            0 => input.re.clone(),
            1 => tex.power_spectrum.clone(),
            2 => tex.spatial_output.clone(),
            _ => continue,
        };
    }
}
