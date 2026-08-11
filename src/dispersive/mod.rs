//! Jeschke–Wojtan hybrid: diffusion decomposition, bar shallow water, Airy/FFT (Algorithm 2),
//! transport, merge, then soft spectral damp of total η and q. Register
//! [`FftPlugin`](crate::fft::FftPlugin) before [`DispersivePlugin`].
//! Do not use [`EwavePlugin`](crate::ewave::EwavePlugin) in the same app: both splice after FFT resolve.
//!
//! Interaction must stay on the GPU. Height splashes use `brush_flow_impulse`; drag wakes add
//! mass flux to the surface-wave field via `brush_tilde_wake` after decompose. Do not mutate the
//! CPU [`Image`] for `state` after init: re-uploading that buffer wipes the live GPU field.
//! See `docs/dispersive.md` for the hybrid split and stability notes.
//!
//! FFT bin layout matches eWave and the README: DC at `(0,0)`, positive then wrapped-negative bins.
//! `k_xy` in `assets/dispersive/dispersive.wgsl` must stay on that convention. A centered
//! `i - N/2` grid only matches if the spectrum was explicitly fftshifted (ocean path).

mod render;

pub mod shaders {
    use bevy::asset::{Handle, uuid_handle};
    use bevy::shader::Shader;

    pub const DISPERSIVE_SURFACE: Handle<Shader> =
        uuid_handle!("6f1e0a1b-2c3d-4e5f-7890-abcdef000001");
}

pub use render::{
    DispersiveGpuResources, DispersivePipelines, DispersiveSimLabel, DispersiveSimUniform,
    plug_dispersive_render_app, prepare_dispersive_gpu, run_dispersive_sim,
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
        sync_component::SyncComponent,
    },
    shader::{Shader, ShaderRef},
};

use crate::fft::{FftPlugin, FftSkipStockPipeline};

/// Marks the sim entity: [`FftSource`](crate::fft::FftSource), [`FftTextures`](crate::fft::resources::FftTextures), [`DispersiveGridImages`].
#[derive(Component, Clone, Copy, Default, Reflect)]
pub struct DispersiveSimRoot;

impl SyncComponent for DispersiveSimRoot {
    type Target = Self;
}

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
    /// Previous-step cell-centered bulk velocity for Alg. 3 midpoint ū.
    pub bar_vel_prev: Handle<Image>,
}

impl SyncComponent for DispersiveGridImages {
    type Target = Self;
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

fn r32_wh(w: u32, h: u32) -> Image {
    use bevy::asset::RenderAssetUsages;
    use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat, TextureUsages};
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
    image.texture_descriptor.usage =
        TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST;
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
            bar_vel_prev: images.add(rgba32(n)),
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
    /// Bump while paused to run one hybrid step.
    pub step_serial: u32,
    /// Bump only for bind-group rebuild after grid recreate. Not used for brushes.
    pub sim_apply_serial: u32,
    /// Bump to re-run `init_basin` on the GPU (true reset).
    pub init_serial: u32,
    pub brush_active: bool,
    pub brush_radius: f32,
    pub brush_strength: f32,
    pub pointer: Vec2,
    pub pointer_prev: Vec2,
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
            // Paper-scale Δx≈1 m at N=256. Finer dx makes FTCS decompose ~useless (dT≪0.25)
            // and leaves high-k junk in ĥ for Airy to amplify.
            tile_world: 256.0,
            g: 9.81,
            dt: 1.0 / 60.0,
            gamma_surf: 0.25,
            diffusion_iters: 128,
            h_bar_omega: 2.0,
            airy_reference_depths: [1.0, 4.0, 16.0, 64.0],
            h_avgmax_beta: 2.0,
            vel_clamp_alpha: 0.5,
            paused: false,
            step_serial: 0,
            sim_apply_serial: 0,
            init_serial: 1,
            brush_active: false,
            brush_radius: 14.0,
            brush_strength: 0.7,
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
        let gamma_surf = self.gamma_surf;
        let diffusion_iters = self.diffusion_iters;
        let h_bar_omega = self.h_bar_omega;
        let airy_reference_depths = self.airy_reference_depths;
        let h_avgmax_beta = self.h_avgmax_beta;
        let vel_clamp_alpha = self.vel_clamp_alpha;
        let paused = self.paused;
        let step_serial = self.step_serial;
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
        next.gamma_surf = gamma_surf;
        next.diffusion_iters = diffusion_iters;
        next.h_bar_omega = h_bar_omega;
        next.airy_reference_depths = airy_reference_depths;
        next.h_avgmax_beta = h_avgmax_beta;
        next.vel_clamp_alpha = vel_clamp_alpha;
        next.paused = paused;
        next.step_serial = step_serial;
        next.brush_active = brush_active;
        next.brush_radius = brush_radius;
        next.brush_strength = brush_strength;
        next.pointer = pointer;
        next.pointer_prev = pointer_prev;
        next.sim_apply_serial = sim_apply_serial;
        next.init_serial = 1;
        *self = next;
    }
}

#[derive(Clone, Copy, Default, Reflect, ShaderType)]
pub struct DispersiveMaterialUniform {
    pub tile_world_size: f32,
    pub grid_size: f32,
    pub _pad0: f32,
    pub _pad1: f32,
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
                tile_world_size: 64.0,
                grid_size: 256.0,
                _pad0: 0.0,
                _pad1: 0.0,
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

fn sync_dispersive_mesh(
    sim: Res<DispersiveController>,
    mut materials: ResMut<Assets<DispersiveSurfaceMaterial>>,
    q: Query<&MeshMaterial3d<DispersiveSurfaceMaterial>, With<DispersiveSurfaceTag>>,
) {
    let Ok(h) = q.single() else {
        return;
    };
    let Some(mut mat) = materials.get_mut(&h.0) else {
        return;
    };
    mat.extension.settings.grid_size = sim.n as f32;
    mat.extension.settings.tile_world_size = sim.tile_world;
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
            .register_type::<DispersiveMaterialUniform>()
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
