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
    sprite_render::AlphaMode2d,
};
use bevy_fft::prelude::*;
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
                attach_input_preview.after(FftSystemSet::PrepareTextures),
                bind_demo_textures.after(attach_input_preview),
                switch_pattern,
            ),
        )
        .run();
}

#[derive(Component)]
struct DemoSprite(u8);

/// Spatial output tile: alpha carries non-color data, so use opaque `ColorMaterial` instead of `Sprite`.
#[derive(Component)]
struct DemoSpatialMesh(Handle<ColorMaterial>);

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
) {
    commands.spawn(Camera2d);
    commands.spawn((
        FftSource::square_forward_then_inverse(256),
        BandPassParams {
            band_center: 0.37,
            band_width: 0.5,
        },
    ));

    let tile = 256.0;
    let gap = 16.0;
    let step = tile + gap;
    let start = -step;
    let quad = meshes.add(Rectangle::from_size(Vec2::splat(tile)));

    for i in 0u8..3 {
        let x = start + step * i as f32;
        if i == 2 {
            let material = materials.add(ColorMaterial {
                alpha_mode: AlphaMode2d::Opaque,
                ..default()
            });
            commands.spawn((
                Mesh2d(quad.clone()),
                MeshMaterial2d(material.clone()),
                Transform::from_xyz(x, 0.0, 0.0),
                DemoSpatialMesh(material),
            ));
        } else {
            commands.spawn((
                Sprite {
                    custom_size: Some(Vec2::splat(tile)),
                    ..default()
                },
                Transform::from_xyz(x, 0.0, 0.0),
                DemoSprite(i),
            ));
        }
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

fn bind_demo_textures(
    fft: Query<(&FftTextures, &InputPatternTextures)>,
    mut sprites: Query<(&mut Sprite, &DemoSprite)>,
    spatial: Query<&DemoSpatialMesh>,
    mut materials: ResMut<Assets<ColorMaterial>>,
) {
    let Ok((tex, input)) = fft.single() else {
        return;
    };
    for (mut sprite, DemoSprite(i)) in &mut sprites {
        sprite.image = match *i {
            0 => input.re.clone(),
            1 => tex.power_spectrum.clone(),
            _ => continue,
        };
    }
    let Ok(DemoSpatialMesh(material)) = spatial.single() else {
        return;
    };
    let Some(mat) = materials.get_mut(material) else {
        return;
    };
    mat.texture = Some(tex.spatial_output.clone());
}
