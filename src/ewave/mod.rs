//! eWave: Tessendorf exponential iWave integrator in k-space (FFT) with periodic boundary.
//! Requires [`FftPlugin`](crate::fft::FftPlugin) (pulled in automatically by [`EwavePlugin`]).
//!
//! FFT bin layout: forward and inverse passes use the usual stock layout (DC at index `(0,0)` on
//! each axis, negative frequencies in the upper index range). Per-bin k-space code must map `(i, j)` to
//! `(k_x, k_y)` with that ordering, not a centered `2π (i - N/2) / L` grid, unless the spectrum is
//! explicitly shifted to match. The `ewave_k_step` entry point in `assets/ewave/ewave.wgsl` encodes
//! the convention to pair with this crate’s FFT.

mod render;

pub mod shaders {
    use bevy::asset::{Handle, uuid_handle};
    use bevy::shader::Shader;

    pub const EWAVE_SURFACE: Handle<Shader> = uuid_handle!("7e0e1a2b-3c4d-5e6f-7890-abcdef123456");
}

pub use render::{
    EwaveGpuResources, EwavePipelines, EwaveSimLabel, EwaveSimNode, EwaveSimUniform,
    plug_ewave_render_app, prepare_ewave_fft_roots_buffer, prepare_ewave_gpu,
    splice_ewave_before_camera,
};

use bevy::{
    asset::load_internal_asset,
    image::Image,
    pbr::{
        ExtendedMaterial, MaterialExtension, MaterialPlugin, MeshMaterial3d, StandardMaterial,
    },
    prelude::*,
    reflect::{Reflect, TypePath},
    render::{
        RenderApp,
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        render_resource::{AsBindGroup, ShaderType},
    },
    shader::{Shader, ShaderRef},
};

use crate::fft::FftPlugin;

/// All GPU images for the eWave sim (one `Rg` for spatial `h` and `φ`, eight FFT buffer planes, four spectra).
#[derive(Clone, Reflect)]
pub struct EwaveFftTextures {
    pub h_phi: Handle<Image>,
    pub buffer_a_re: Handle<Image>,
    pub buffer_a_im: Handle<Image>,
    pub buffer_b_re: Handle<Image>,
    pub buffer_b_im: Handle<Image>,
    pub buffer_c_re: Handle<Image>,
    pub buffer_c_im: Handle<Image>,
    pub buffer_d_re: Handle<Image>,
    pub buffer_d_im: Handle<Image>,
    pub h_hat_re: Handle<Image>,
    pub h_hat_im: Handle<Image>,
    pub p_hat_re: Handle<Image>,
    pub p_hat_im: Handle<Image>,
}

fn rgba32(n: u32) -> Image {
    use bevy::asset::RenderAssetUsages;
    use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat, TextureUsages};
    let mut image = Image::new_fill(
        Extent3d {
            width: n,
            height: n,
            depth_or_array_layers: 1,
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
    image
}

impl EwaveFftTextures {
    fn new(images: &mut Assets<Image>, n: u32) -> Self {
        Self {
            h_phi: images.add(ewave_h_phi_image(n, n)),
            buffer_a_re: images.add(rgba32(n)),
            buffer_a_im: images.add(rgba32(n)),
            buffer_b_re: images.add(rgba32(n)),
            buffer_b_im: images.add(rgba32(n)),
            buffer_c_re: images.add(rgba32(n)),
            buffer_c_im: images.add(rgba32(n)),
            buffer_d_re: images.add(rgba32(n)),
            buffer_d_im: images.add(rgba32(n)),
            h_hat_re: images.add(rgba32(n)),
            h_hat_im: images.add(rgba32(n)),
            p_hat_re: images.add(rgba32(n)),
            p_hat_im: images.add(rgba32(n)),
        }
    }
}

/// Main-world state; extracted to the render world for simulation and materials.
#[derive(Resource, Clone, ExtractResource, Reflect)]
pub struct EwaveController {
    pub n: u32,
    pub fft: crate::fft::FftSource,
    pub textures: EwaveFftTextures,
    /// World width of the patch (same units as `g`).
    pub tile_world: f32,
    pub g: f32,
    pub dt: f32,
    pub height_scale: f32,
    pub paused: bool,
    pub sim_apply_serial: u32,
    pub brush_active: bool,
    pub brush_radius: f32,
    pub brush_strength: f32,
    pub pointer: Vec2,
    pub pointer_prev: Vec2,
}

impl EwaveController {
    pub fn new(images: &mut Assets<Image>, n: u32) -> Self {
        let fft = crate::fft::FftSource::try_square_forward_then_inverse(n)
            .expect("eWave needs power-of-two grid size");
        let textures = EwaveFftTextures::new(images, n);
        Self {
            n,
            fft,
            textures,
            tile_world: 32.0,
            g: 9.81,
            dt: 0.04,
            height_scale: 0.1,
            paused: false,
            sim_apply_serial: 1,
            brush_active: false,
            brush_radius: 18.0,
            brush_strength: 0.5,
            pointer: Vec2::ZERO,
            pointer_prev: Vec2::ZERO,
        }
    }

    pub fn rebuild(&mut self, images: &mut Assets<Image>, n: u32) {
        if n == self.n {
            return;
        }
        self.n = n;
        self.fft = crate::fft::FftSource::try_square_forward_then_inverse(n)
            .expect("eWave needs power-of-two grid size");
        self.textures = EwaveFftTextures::new(images, n);
        self.sim_apply_serial = self.sim_apply_serial.wrapping_add(1);
    }

    pub fn h_phi(&self) -> &Handle<Image> {
        &self.textures.h_phi
    }
}

/// `h` and `φ` live in **.xy** (same convention as a `Rg32` would). `Rgba32Float` is used because
/// `Rg32Float` read/write storage is not supported on Metal (see `docs/shallow_water.md`).
fn ewave_h_phi_image(w: u32, h: u32) -> Image {
    use bevy::asset::RenderAssetUsages;
    use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat, TextureUsages};
    let mut image = Image::new_fill(
        Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &[0u8; 16],
        TextureFormat::Rgba32Float,
        RenderAssetUsages::default(),
    );
    image.texture_descriptor.usage = TextureUsages::STORAGE_BINDING
        | TextureUsages::TEXTURE_BINDING
        | TextureUsages::COPY_DST;
    image
}

#[derive(Clone, Copy, Default, Reflect, ShaderType)]
pub struct EwaveMaterialUniform {
    pub height_scale: f32,
    pub tile_world_size: f32,
    pub grid_size: f32,
    pub _pad0: f32,
}

#[derive(Asset, AsBindGroup, TypePath, Clone)]
pub struct EwaveSurfaceExtension {
    #[uniform(100)]
    pub settings: EwaveMaterialUniform,
    #[texture(101)]
    #[sampler(102)]
    pub h_phi: Handle<Image>,
}

impl Default for EwaveSurfaceExtension {
    fn default() -> Self {
        Self {
            settings: EwaveMaterialUniform {
                height_scale: 0.1,
                tile_world_size: 32.0,
                grid_size: 256.0,
                _pad0: 0.0,
            },
            h_phi: default(),
        }
    }
}

impl MaterialExtension for EwaveSurfaceExtension {
    fn vertex_shader() -> ShaderRef {
        ShaderRef::Handle(shaders::EWAVE_SURFACE.clone())
    }
    fn fragment_shader() -> ShaderRef {
        ShaderRef::Handle(shaders::EWAVE_SURFACE.clone())
    }
    fn prepass_vertex_shader() -> ShaderRef {
        ShaderRef::Handle(shaders::EWAVE_SURFACE.clone())
    }
    fn deferred_vertex_shader() -> ShaderRef {
        ShaderRef::Handle(shaders::EWAVE_SURFACE.clone())
    }
    fn deferred_fragment_shader() -> ShaderRef {
        ShaderRef::Handle(shaders::EWAVE_SURFACE.clone())
    }
}

pub type EwaveSurfaceMaterial = ExtendedMaterial<StandardMaterial, EwaveSurfaceExtension>;

#[derive(Component)]
pub struct EwaveSurfaceTag;

fn sync_ewave_mesh_material(
    sim: Res<EwaveController>,
    mut materials: ResMut<Assets<EwaveSurfaceMaterial>>,
    q: Query<&MeshMaterial3d<EwaveSurfaceMaterial>, With<EwaveSurfaceTag>>,
) {
    let Ok(h) = q.single() else {
        return;
    };
    let Some(mat) = materials.get_mut(&h.0) else {
        return;
    };
    let g = sim.n as f32;
    mat.extension.settings.grid_size = g;
    mat.extension.settings.tile_world_size = sim.tile_world;
    mat.extension.settings.height_scale = sim.height_scale;
    mat.extension.h_phi = sim.textures.h_phi.clone();
}

pub struct EwavePlugin;

impl Plugin for EwavePlugin {
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<FftPlugin>() {
            app.add_plugins(FftPlugin);
        }
        load_internal_asset!(
            app,
            shaders::EWAVE_SURFACE,
            "surface.wgsl",
            Shader::from_wgsl
        );
        app.register_type::<EwaveController>()
            .register_type::<EwaveFftTextures>()
            .register_type::<EwaveMaterialUniform>()
            .add_plugins(ExtractResourcePlugin::<EwaveController>::default())
            .add_plugins(MaterialPlugin::<EwaveSurfaceMaterial>::default())
            .add_systems(PostUpdate, sync_ewave_mesh_material);
    }

    fn finish(&self, app: &mut App) {
        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            plug_ewave_render_app(render_app);
        }
    }
}
