//! Jeschke–Wojtan hybrid: diffusion decomposition, bar shallow water, Airy/FFT (Algorithm 2), transport, merge.
//! Register [`FftPlugin`](crate::fft::FftPlugin) before [`DispersivePlugin`].
//! Do not use [`EwavePlugin`](crate::ewave::EwavePlugin) in the same app: both splice after [`FftNode::ResolveOutputs`].
//! The bulk (`bar`) step follows [Chentanez and Müller 2010] (CMF10), matching [`shallow_water`](crate::shallow_water):
//! MacCormack semi-Lagrangian advection on staggered face velocities, upwind mass flux for `h` with depth limiting,
//! then η-pressure on faces and gather to cell `q` (see `bar_*` entry points in `assets/dispersive/dispersive.wgsl`).
//! Spectral Airy uses four reference depths (paper Sec. 4.3), half-cell phase on ĥ for staggered coupling, β from Appendix B,
//! and merge includes the check flux q̌ ≈ h̃ ū (paper Eq. 17) alongside div(q̄ + q̃).

mod render;

pub mod physics;

use bevy::{
    asset::{Asset, Assets, Handle, load_internal_asset},
    ecs::{query::QueryItem, system::lifetimeless::Read},
    image::Image,
    pbr::{ExtendedMaterial, MaterialExtension, MeshMaterial3d, StandardMaterial},
    prelude::*,
    reflect::{Reflect, TypePath},
    render::{
        RenderApp,
        extract_component::{ExtractComponent, ExtractComponentPlugin},
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        render_resource::{AsBindGroup, Extent3d, ShaderType, TextureDimension, TextureFormat, TextureUsages},
    },
    shader::{Shader, ShaderRef},
};

pub use render::{
    DispersiveGpuResources, DispersivePipelines, DispersiveSimLabel, DispersiveSimNode,
    DispersiveSimUniform, plug_dispersive_render_app, prepare_dispersive_gpu,
    splice_dispersive_before_camera,
};

use crate::fft::{FftPlugin, FftSkipStockPipeline};


/// Marks the sim entity: [`FftSource`](crate::fft::FftSource), [`FftTextures`](crate::fft::resources::FftTextures), [`DispersiveGridImages`].
#[derive(Component, Clone, Copy, Default, Reflect)]
pub struct DispersiveSimRoot;

impl ExtractComponent for DispersiveSimRoot {
    type QueryData = Read<DispersiveSimRoot>;
    type QueryFilter = ();
    type Out = DispersiveSimRoot;

    fn extract_component(item: QueryItem<'_, '_, Self::QueryData>) -> Option<Self::Out> {
        Some(*item)
    }
}

/// Extra GPU textures: simulation fields and spectrums.
#[derive(Component, Clone, Reflect)]
pub struct DispersiveGridImages {
    pub state: Handle<Image>,
    pub bar: Handle<Image>,
    pub tilde: Handle<Image>,
    pub bed: Handle<Image>,
    pub scratch: Handle<Image>,
    pub h_spec_re: Handle<Image>,
    pub h_spec_im: Handle<Image>,
    /// Real and imaginary parts of the q-component spectrum (`.r` / `.g`) between multi-depth Airy passes.
    pub q_spec_backup: Handle<Image>,
    /// Spatial qx after Airy at each reference depth (`.xyzw` = depths 0..3).
    pub airy_stack_qx: Handle<Image>,
    pub airy_stack_qy: Handle<Image>,
    /// Staggered bulk face `u` (CMF10), size `(n+1) × n`.
    pub bar_face_u: Handle<Image>,
    /// Staggered bulk face `w`, size `n × (n+1)`.
    pub bar_face_w: Handle<Image>,
    pub bar_mac_u: Handle<Image>,
    pub bar_mac_w: Handle<Image>,
}

impl ExtractComponent for DispersiveGridImages {
    type QueryData = Read<DispersiveGridImages>;
    type QueryFilter = ();
    type Out = DispersiveGridImages;

    fn extract_component(item: QueryItem<'_, '_, Self::QueryData>) -> Option<Self::Out> {
        Some(item.clone())
    }
}

fn rgba32(n: u32) -> Image {
    rgba32_wh(n, n)
}

fn rgba32_wh(w: u32, h: u32) -> Image {
    use bevy::asset::RenderAssetUsages;
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
    image.texture_descriptor.usage = TextureUsages::STORAGE_BINDING
        | TextureUsages::TEXTURE_BINDING
        | TextureUsages::COPY_DST
        | TextureUsages::COPY_SRC;
    image
}

fn r32_wh(w: u32, h: u32) -> Image {
    use bevy::asset::RenderAssetUsages;
    let mut image = Image::new_fill(
        Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &[0; 4],
        TextureFormat::R32Float,
        RenderAssetUsages::default(),
    );
    image.texture_descriptor.usage = TextureUsages::STORAGE_BINDING
        | TextureUsages::TEXTURE_BINDING
        | TextureUsages::COPY_DST
        | TextureUsages::COPY_SRC;
    image
}

impl DispersiveGridImages {
    fn new(images: &mut Assets<Image>, n: u32) -> Self {
        Self {
            state: images.add(rgba32(n)),
            bar: images.add(rgba32(n)),
            tilde: images.add(rgba32(n)),
            bed: images.add(rgba32(n)),
            scratch: images.add(rgba32(n)),
            h_spec_re: images.add(rgba32(n)),
            h_spec_im: images.add(rgba32(n)),
            q_spec_backup: images.add(rgba32(n)),
            airy_stack_qx: images.add(rgba32(n)),
            airy_stack_qy: images.add(rgba32(n)),
            bar_face_u: images.add(r32_wh(n + 1, n)),
            bar_face_w: images.add(r32_wh(n, n + 1)),
            bar_mac_u: images.add(rgba32_wh(n + 1, n)),
            bar_mac_w: images.add(rgba32_wh(n, n + 1)),
        }
    }
}

/// Main-world parameters (extracted to render world).
#[derive(Resource, Clone, ExtractResource, Reflect)]
pub struct DispersiveController {
    pub sim_entity: Entity,
    pub state: Handle<Image>,
    pub n: u32,
    pub tile_world: f32,
    pub g: f32,
    pub dt: f32,
    pub gamma_surf: f32,
    pub diffusion_iters: u32,
    /// Minimum depth (m) used in ω(k, h) when h̄ is tiny.
    pub h_bar_omega: f32,
    /// Reference water depths (m) for spectral interpolation, ascending (paper Fig. 8 style: 1, 4, 16, 64 m).
    pub airy_reference_depths: [f32; 4],
    /// CMF10 depth limiting: `h_avgmax = beta * dx / (g dt)` on neighbor average (see `shallow_water`).
    pub h_avgmax_beta: f32,
    /// CMF10 face velocity clamp: `± alpha * dx / dt`.
    pub vel_clamp_alpha: f32,
    pub paused: bool,
    pub sim_apply_serial: u32,
    /// Bump to run `init_sloped_beach` on GPU once.
    pub init_serial: u32,
    pub height_scale: f32,
}

impl DispersiveController {
    /// Spawns the FFT + dispersive entity (power-of-two `n` only).
    pub fn spawn(commands: &mut Commands, images: &mut Assets<Image>, n: u32) -> Self {
        let grid = DispersiveGridImages::new(images, n);
        let state = grid.state.clone();
        let fft = crate::fft::FftSource::try_square_forward_then_inverse(n)
            .expect("dispersive needs power-of-two n");
        let sim_entity = commands
            .spawn((DispersiveSimRoot, FftSkipStockPipeline, fft, grid))
            .id();
        Self {
            sim_entity,
            state,
            n,
            tile_world: 64.0,
            g: 9.81,
            dt: 0.04,
            gamma_surf: 0.25,
            diffusion_iters: 32,
            h_bar_omega: 2.0,
            airy_reference_depths: [1.0, 4.0, 16.0, 64.0],
            h_avgmax_beta: 2.0,
            vel_clamp_alpha: 0.5,
            paused: false,
            sim_apply_serial: 0,
            init_serial: 1,
            height_scale: 0.2,
        }
    }
}

#[derive(Clone, Copy, Default, Reflect, ShaderType)]
pub struct DispersiveMaterialUniform {
    pub height_scale: f32,
    pub tile_world_size: f32,
    pub grid_size: f32,
    pub _pad0: f32,
}

#[derive(Asset, AsBindGroup, TypePath, Clone)]
pub struct DispersiveSurfaceExtension {
    #[uniform(100)]
    pub settings: DispersiveMaterialUniform,
    #[texture(101)]
    #[sampler(102)]
    pub state: Handle<Image>,
}

impl Default for DispersiveSurfaceExtension {
    fn default() -> Self {
        Self {
            settings: DispersiveMaterialUniform {
                height_scale: 0.2,
                tile_world_size: 64.0,
                grid_size: 256.0,
                _pad0: 0.0,
            },
            state: default(),
        }
    }
}

impl MaterialExtension for DispersiveSurfaceExtension {
    fn vertex_shader() -> ShaderRef {
        ShaderRef::Handle(shaders::DISPERSIVE_SURFACE.clone())
    }
    fn fragment_shader() -> ShaderRef {
        ShaderRef::Handle(shaders::DISPERSIVE_SURFACE.clone())
    }
    fn prepass_vertex_shader() -> ShaderRef {
        ShaderRef::Handle(shaders::DISPERSIVE_SURFACE.clone())
    }
    fn deferred_vertex_shader() -> ShaderRef {
        ShaderRef::Handle(shaders::DISPERSIVE_SURFACE.clone())
    }
    fn deferred_fragment_shader() -> ShaderRef {
        ShaderRef::Handle(shaders::DISPERSIVE_SURFACE.clone())
    }
}

pub type DispersiveSurfaceMaterial = ExtendedMaterial<StandardMaterial, DispersiveSurfaceExtension>;

#[derive(Component)]
pub struct DispersiveSurfaceTag;

pub mod shaders {
    use bevy::asset::{Handle, uuid_handle};
    use bevy::shader::Shader;

    pub const DISPERSIVE_SURFACE: Handle<Shader> =
        uuid_handle!("6f1e0a1b-2c3d-4e5f-7890-abcdef000001");
}

fn sync_dispersive_mesh(
    sim: Res<DispersiveController>,
    mut materials: ResMut<Assets<DispersiveSurfaceMaterial>>,
    q: Query<&MeshMaterial3d<DispersiveSurfaceMaterial>, With<DispersiveSurfaceTag>>,
) {
    let Ok(h) = q.single() else {
        return;
    };
    let Some(mat) = materials.get_mut(&h.0) else {
        return;
    };
    mat.extension.settings.grid_size = sim.n as f32;
    mat.extension.settings.tile_world_size = sim.tile_world;
    mat.extension.settings.height_scale = sim.height_scale;
    mat.extension.state = sim.state.clone();
}

pub struct DispersivePlugin;

impl Plugin for DispersivePlugin {
    fn build(&self, app: &mut App) {
        load_internal_asset!(
            app,
            shaders::DISPERSIVE_SURFACE,
            "surface.wgsl",
            Shader::from_wgsl
        );
        app.register_type::<DispersiveController>()
            .register_type::<DispersiveSimRoot>()
            .register_type::<DispersiveGridImages>()
            .add_plugins(ExtractResourcePlugin::<DispersiveController>::default())
            .add_plugins(ExtractComponentPlugin::<DispersiveSimRoot>::default())
            .add_plugins(ExtractComponentPlugin::<DispersiveGridImages>::default())
            .add_plugins(MaterialPlugin::<DispersiveSurfaceMaterial>::default())
            .add_systems(PostUpdate, sync_dispersive_mesh);
    }

    fn finish(&self, app: &mut App) {
        assert!(
            app.is_plugin_added::<FftPlugin>(),
            "DispersivePlugin requires FftPlugin (add FftPlugin before DispersivePlugin)."
        );
        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            plug_dispersive_render_app(render_app);
        }
    }
}
