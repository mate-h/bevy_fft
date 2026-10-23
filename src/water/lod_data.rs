use bevy::{
    asset::RenderAssetUsages,
    prelude::*,
    render::{
        RenderApp,
        extract_resource::ExtractResource,
        render_resource::{
            Extent3d, ShaderType, TextureDimension, TextureFormat, TextureUsages,
            TextureViewDescriptor, TextureViewDimension, UniformBuffer,
        },
        renderer::{RenderDevice, RenderQueue},
    },
};

/// Most slices a [`WaterLodData`] can hold. Must match `WATER_LOD_MAX_SLICES` in `lod.wesl`.
pub const WATER_LOD_MAX_SLICES: usize = 12;

/// Size and reach of the LOD data. Read once at startup.
#[derive(Resource, Clone, Copy, Debug, Reflect)]
pub struct WaterLodSettings {
    /// Number of slices, from 1 to [`WATER_LOD_MAX_SLICES`].
    pub slice_count: u32,
    /// Texels per slice edge.
    pub resolution: u32,
    /// Texel width of the finest slice in meters.
    pub base_texel_size: f32,
}

impl Default for WaterLodSettings {
    fn default() -> Self {
        Self {
            slice_count: 10,
            resolution: 256,
            base_texel_size: 1.0 / 32.0,
        }
    }
}

impl WaterLodSettings {
    fn clamped_count(&self) -> u32 {
        self.slice_count.clamp(1, WATER_LOD_MAX_SLICES as u32)
    }

    /// Texel width of slice `i` in meters.
    pub fn texel_size(&self, i: u32) -> f32 {
        self.base_texel_size * (1u32 << i) as f32
    }

    /// Edge length of slice `i` in meters.
    pub fn slice_width(&self, i: u32) -> f32 {
        self.texel_size(i) * self.resolution as f32
    }
}

/// Slice placement shared with the GPU. Layout matches `WaterLodSlices` in `lod.wesl`.
#[derive(Clone, Copy, Debug, Default, ShaderType, Reflect)]
pub struct WaterLodSlices {
    /// Per slice: snapped centre X, snapped centre Z, texel width, slice width (meters).
    pub slices: [Vec4; WATER_LOD_MAX_SLICES],
    pub count: u32,
    pub resolution: u32,
    pub _pad0: u32,
    pub _pad1: u32,
}

/// Camera-centred LOD texture arrays and where each slice currently sits.
///
/// `displacement` stores, per texel, the height slope along X, the slope along Z, the height in
/// meters above the surface mesh plane, and the wind-aligned horizontal chop in meters. These are
/// the same lanes as the FFT spatial output, already scaled, so contributions from several solvers
/// add up linearly. `foam` stores foam coverage in its red channel and the depth of the water
/// column in meters in its green channel.
#[derive(Resource, Clone, ExtractResource)]
#[extract_app(RenderApp)]
pub struct WaterLodData {
    pub displacement: Handle<Image>,
    pub foam: Handle<Image>,
    pub slices: WaterLodSlices,
}

impl WaterLodData {
    /// Index of the finest slice whose interior contains `xz`, if any.
    pub fn finest_slice_containing(&self, xz: Vec2) -> Option<u32> {
        (0..self.slices.count).find(|&i| {
            let s = self.slices.slices[i as usize];
            let half = s.w * 0.5 - s.z;
            (xz.x - s.x).abs() <= half && (xz.y - s.y).abs() <= half
        })
    }
}

/// The camera whose position centres the [`WaterLodData`] and the ocean clipmap.
/// Without it, the first [`Camera3d`] is used.
#[derive(Component, Clone, Copy, Debug, Default, Reflect)]
pub struct WaterViewer;

fn lod_array_image(settings: &WaterLodSettings) -> Image {
    let mut image = Image::new_fill(
        Extent3d {
            width: settings.resolution,
            height: settings.resolution,
            depth_or_array_layers: settings.clamped_count(),
        },
        TextureDimension::D2,
        &[0; 16],
        TextureFormat::Rgba32Float,
        RenderAssetUsages::default(),
    );
    image.texture_descriptor.usage = TextureUsages::STORAGE_BINDING
        | TextureUsages::TEXTURE_BINDING
        | TextureUsages::COPY_DST
        | TextureUsages::COPY_SRC;
    image.texture_view_descriptor = Some(TextureViewDescriptor {
        dimension: Some(TextureViewDimension::D2Array),
        ..default()
    });
    image
}

pub(super) fn create_water_lod_data(
    mut commands: Commands,
    settings: Res<WaterLodSettings>,
    mut images: ResMut<Assets<Image>>,
) {
    let slices = WaterLodSlices {
        count: settings.clamped_count(),
        resolution: settings.resolution,
        ..default()
    };
    commands.insert_resource(WaterLodData {
        displacement: images.add(lod_array_image(&settings)),
        foam: images.add(lod_array_image(&settings)),
        slices,
    });
}

/// Snaps each slice centre to a two-texel grid under the viewer. Moving in whole texel pairs
/// keeps every texel on the same world position as before, so the data never swims.
pub(super) fn update_water_lod_slices(
    settings: Res<WaterLodSettings>,
    viewers: Query<&GlobalTransform, With<WaterViewer>>,
    cameras: Query<&GlobalTransform, With<Camera3d>>,
    mut data: ResMut<WaterLodData>,
) {
    let eye = viewers
        .iter()
        .next()
        .or_else(|| cameras.iter().next())
        .map_or(Vec3::ZERO, |t| t.translation());
    let count = settings.clamped_count();
    let mut slices = data.slices;
    for i in 0..count {
        let texel = settings.texel_size(i);
        let step = 2.0 * texel;
        let center = (eye.xz() / step).round() * step;
        slices.slices[i as usize] = Vec4::new(center.x, center.y, texel, settings.slice_width(i));
    }
    slices.count = count;
    slices.resolution = settings.resolution;
    data.slices = slices;
}

/// Render-world uniform buffer holding the current [`WaterLodSlices`].
#[derive(Resource, Default)]
pub struct WaterLodGpu {
    pub slices: UniformBuffer<WaterLodSlices>,
}

pub(super) fn prepare_water_lod_gpu(
    data: Option<Res<WaterLodData>>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    mut gpu: ResMut<WaterLodGpu>,
) {
    let Some(data) = data else {
        return;
    };
    gpu.slices.set(data.slices);
    gpu.slices.write_buffer(&device, &queue);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slices_double_in_width() {
        let s = WaterLodSettings::default();
        assert_eq!(s.slice_width(1), 2.0 * s.slice_width(0));
        assert_eq!(s.slice_width(0), 256.0 / 32.0);
    }

    #[test]
    fn finest_slice_prefers_inner_slices() {
        let settings = WaterLodSettings {
            slice_count: 3,
            resolution: 16,
            base_texel_size: 1.0,
        };
        let mut slices = WaterLodSlices {
            count: 3,
            resolution: 16,
            ..default()
        };
        for i in 0..3 {
            slices.slices[i] = Vec4::new(
                0.0,
                0.0,
                settings.texel_size(i as u32),
                settings.slice_width(i as u32),
            );
        }
        let data = WaterLodData {
            displacement: Handle::default(),
            foam: Handle::default(),
            slices,
        };
        assert_eq!(data.finest_slice_containing(Vec2::new(3.0, 0.0)), Some(0));
        assert_eq!(data.finest_slice_containing(Vec2::new(10.0, 0.0)), Some(1));
        assert_eq!(data.finest_slice_containing(Vec2::new(100.0, 0.0)), None);
    }
}
