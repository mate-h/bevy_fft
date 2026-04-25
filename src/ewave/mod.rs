//! eWave: Tessendorf exponential iWave integrator in k-space (FFT) with periodic boundary.
//! Register [`FftPlugin`](crate::fft::FftPlugin) **before** [`EwavePlugin`]. Otherwise `EwavePlugin::finish`
//! runs too early and the render world does not have [`crate::fft::resources::FftBindGroupLayouts`] yet.
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
    plug_ewave_render_app, prepare_ewave_gpu, splice_ewave_before_camera,
};

use bevy::{
    asset::load_internal_asset,
    ecs::{query::QueryItem, system::lifetimeless::Read},
    image::Image,
    pbr::{ExtendedMaterial, MaterialExtension, MaterialPlugin, MeshMaterial3d, StandardMaterial},
    prelude::*,
    reflect::{Reflect, TypePath},
    render::{
        RenderApp,
        extract_component::{ExtractComponent, ExtractComponentPlugin},
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        render_resource::{AsBindGroup, ShaderType},
    },
    shader::{Shader, ShaderRef},
};

use crate::fft::{FftPlugin, FftSkipStockPipeline};

/// Marks the main-world entity that carries the eWave grid ([`FftSource`](crate::fft::FftSource),
/// [`FftTextures`](crate::fft::resources::FftTextures), and [`EwaveGridImages`]).
#[derive(Component, Clone, Copy, Default, Reflect)]
pub struct EwaveSimRoot;

impl ExtractComponent for EwaveSimRoot {
    type QueryData = Read<EwaveSimRoot>;
    type QueryFilter = ();
    type Out = EwaveSimRoot;

    fn extract_component(item: QueryItem<'_, '_, Self::QueryData>) -> Option<Self::Out> {
        Some(*item)
    }
}

/// GPU images used by eWave outside the shared [`FftTextures`](crate::fft::resources::FftTextures) set:
/// spatial `h` and `φ`, and the four spectrum planes.
#[derive(Component, Clone, Reflect)]
pub struct EwaveGridImages {
    pub h_phi: Handle<Image>,
    pub h_hat_re: Handle<Image>,
    pub h_hat_im: Handle<Image>,
    pub p_hat_re: Handle<Image>,
    pub p_hat_im: Handle<Image>,
}

impl ExtractComponent for EwaveGridImages {
    type QueryData = Read<EwaveGridImages>;
    type QueryFilter = ();
    type Out = EwaveGridImages;

    fn extract_component(item: QueryItem<'_, '_, Self::QueryData>) -> Option<Self::Out> {
        Some(item.clone())
    }
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

impl EwaveGridImages {
    fn new(images: &mut Assets<Image>, n: u32) -> Self {
        Self {
            h_phi: images.add(ewave_h_phi_image(n, n)),
            h_hat_re: images.add(rgba32(n)),
            h_hat_im: images.add(rgba32(n)),
            p_hat_re: images.add(rgba32(n)),
            p_hat_im: images.add(rgba32(n)),
        }
    }
}

/// Main-world simulation parameters; extracted to the render world. The FFT grid lives on
/// [`EwaveSimRoot`] as [`crate::fft::FftSource`] plus [`crate::fft::resources::FftTextures`].
#[derive(Resource, Clone, ExtractResource, Reflect)]
pub struct EwaveController {
    /// Entity with [`EwaveSimRoot`], [`FftSkipStockPipeline`], [`crate::fft::FftSource`], [`EwaveGridImages`], and [`crate::fft::resources::FftTextures`].
    pub sim_entity: Entity,
    /// Cached [`EwaveGridImages::h_phi`] for materials and setup.
    pub h_phi: Handle<Image>,
    pub n: u32,
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
    /// Spawns the FFT entity with [`crate::fft::FftSource::try_square_forward_then_inverse`].
    pub fn spawn(commands: &mut Commands, images: &mut Assets<Image>, n: u32) -> Self {
        let grid = EwaveGridImages::new(images, n);
        let h_phi = grid.h_phi.clone();
        let fft = crate::fft::FftSource::try_square_forward_then_inverse(n)
            .expect("eWave needs power-of-two grid size");
        let sim_entity = commands
            .spawn((EwaveSimRoot, FftSkipStockPipeline, fft, grid))
            .id();
        Self {
            sim_entity,
            h_phi,
            n,
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

    pub fn rebuild(&mut self, commands: &mut Commands, images: &mut Assets<Image>, n: u32) {
        if n == self.n {
            return;
        }
        let tile_world = self.tile_world;
        let g = self.g;
        let dt = self.dt;
        let height_scale = self.height_scale;
        let paused = self.paused;
        let brush_active = self.brush_active;
        let brush_radius = self.brush_radius;
        let brush_strength = self.brush_strength;
        let pointer = self.pointer;
        let pointer_prev = self.pointer_prev;
        let sim_apply_serial = self.sim_apply_serial.wrapping_add(1);
        commands.entity(self.sim_entity).despawn();
        let mut next = Self::spawn(commands, images, n);
        next.tile_world = tile_world;
        next.g = g;
        next.dt = dt;
        next.height_scale = height_scale;
        next.paused = paused;
        next.brush_active = brush_active;
        next.brush_radius = brush_radius;
        next.brush_strength = brush_strength;
        next.pointer = pointer;
        next.pointer_prev = pointer_prev;
        next.sim_apply_serial = sim_apply_serial;
        *self = next;
    }

    pub fn h_phi(&self) -> &Handle<Image> {
        &self.h_phi
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
        &[0; 16],
        TextureFormat::Rgba32Float,
        RenderAssetUsages::default(),
    );
    image.texture_descriptor.usage =
        TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST;
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
    mat.extension.h_phi = sim.h_phi.clone();
}

pub struct EwavePlugin;

impl Plugin for EwavePlugin {
    fn build(&self, app: &mut App) {
        load_internal_asset!(
            app,
            shaders::EWAVE_SURFACE,
            "surface.wgsl",
            Shader::from_wgsl
        );
        app.register_type::<EwaveController>()
            .register_type::<EwaveSimRoot>()
            .register_type::<EwaveGridImages>()
            .register_type::<EwaveMaterialUniform>()
            .add_plugins(ExtractResourcePlugin::<EwaveController>::default())
            .add_plugins(ExtractComponentPlugin::<EwaveSimRoot>::default())
            .add_plugins(ExtractComponentPlugin::<EwaveGridImages>::default())
            .add_plugins(MaterialPlugin::<EwaveSurfaceMaterial>::default())
            .add_systems(PostUpdate, sync_ewave_mesh_material);
    }

    fn finish(&self, app: &mut App) {
        assert!(
            app.is_plugin_added::<FftPlugin>(),
            "EwavePlugin requires FftPlugin to be registered first (e.g. add_plugins((FftPlugin, EwavePlugin)))."
        );
        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            plug_ewave_render_app(render_app);
        }
    }
}
