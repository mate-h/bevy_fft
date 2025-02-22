use bevy_asset::{Asset, Handle};
use bevy_image::Image;
use bevy_reflect::prelude::ReflectDefault;
use bevy_reflect::Reflect;
use bevy_render::texture::GpuImage;

#[derive(Asset, Reflect, Debug, Clone)]
#[reflect(opaque)]
#[reflect(Default, Debug)]
pub struct ComplexImage {
    pub source: Handle<Image>,
}

impl Default for ComplexImage {
    fn default() -> Self {
        Self {
            source: Handle::default(),
        }
    }
}
