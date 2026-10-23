//! Scratch copies of the LOD data for writers that blend over what earlier writers stored.
//!
//! A solver that covers only part of the world, such as a shallow water grid, fades into the data
//! beneath it at its edges, so it has to read the texel it overwrites. Storage textures of this
//! format cannot be read and written in the same pass on every backend, so each blending writer
//! first copies the arrays into [`WaterLodScratch`], reads the copy, and writes the LOD data.

use bevy::{
    prelude::*,
    render::{
        render_asset::RenderAssets,
        render_resource::{
            CommandEncoder, Extent3d, Texture, TextureDescriptor, TextureDimension, TextureUsages,
            TextureView, TextureViewDescriptor, TextureViewDimension,
        },
        renderer::RenderDevice,
        texture::GpuImage,
    },
};

use super::WaterLodData;

struct ScratchArray {
    texture: Texture,
    view: TextureView,
}

/// Render-world copies of [`WaterLodData::displacement`] and [`WaterLodData::foam`], sized to
/// match them. Bind [`Self::displacement`] and [`Self::foam`] as sampled `texture_2d_array`s.
#[derive(Resource, Default)]
pub struct WaterLodScratch {
    displacement: Option<ScratchArray>,
    foam: Option<ScratchArray>,
    size: Extent3d,
}

impl WaterLodScratch {
    pub fn displacement(&self) -> Option<&TextureView> {
        self.displacement.as_ref().map(|a| &a.view)
    }

    pub fn foam(&self) -> Option<&TextureView> {
        self.foam.as_ref().map(|a| &a.view)
    }

    /// Copies the current LOD data into the scratch arrays. Returns false when either side is not
    /// ready, in which case the caller should skip its blend pass this frame.
    pub fn copy_from_lod(
        &self,
        encoder: &mut CommandEncoder,
        images: &RenderAssets<GpuImage>,
        lod: &WaterLodData,
    ) -> bool {
        let (Some(disp), Some(foam)) = (images.get(&lod.displacement), images.get(&lod.foam))
        else {
            return false;
        };
        let (Some(disp_copy), Some(foam_copy)) = (&self.displacement, &self.foam) else {
            return false;
        };
        if disp.texture_descriptor.size != self.size || foam.texture_descriptor.size != self.size {
            return false;
        }
        encoder.copy_texture_to_texture(
            disp.texture.as_image_copy(),
            disp_copy.texture.as_image_copy(),
            self.size,
        );
        encoder.copy_texture_to_texture(
            foam.texture.as_image_copy(),
            foam_copy.texture.as_image_copy(),
            self.size,
        );
        true
    }
}

fn scratch_array(device: &RenderDevice, source: &GpuImage, label: &'static str) -> ScratchArray {
    let texture = device.create_texture(&TextureDescriptor {
        label: Some(label),
        size: source.texture_descriptor.size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: source.texture_descriptor.format,
        usage: TextureUsages::COPY_DST | TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&TextureViewDescriptor {
        dimension: Some(TextureViewDimension::D2Array),
        ..default()
    });
    ScratchArray { texture, view }
}

/// Creates or resizes the scratch arrays to match the LOD data.
pub(super) fn prepare_water_lod_scratch(
    data: Option<Res<WaterLodData>>,
    images: Res<RenderAssets<GpuImage>>,
    device: Res<RenderDevice>,
    mut scratch: ResMut<WaterLodScratch>,
) {
    let Some(data) = data else {
        return;
    };
    let (Some(disp), Some(foam)) = (images.get(&data.displacement), images.get(&data.foam)) else {
        return;
    };
    let size = disp.texture_descriptor.size;
    if scratch.displacement.is_some() && scratch.size == size {
        return;
    }
    scratch.displacement = Some(scratch_array(
        &device,
        disp,
        "water_lod_scratch_displacement",
    ));
    scratch.foam = Some(scratch_array(&device, foam, "water_lod_scratch_foam"));
    scratch.size = size;
}
