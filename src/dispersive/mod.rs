//! Jeschke–Wojtan hybrid: diffusion decomposition, bar shallow water, Airy/FFT (Algorithm 2),
//! transport, merge, then soft spectral damp of total η and q. Register
//! [`FftPlugin`] before [`DispersivePlugin`].
//! Do not use [`EwavePlugin`](crate::ewave::EwavePlugin) in the same app: both splice after FFT resolve.
//!
//! Interaction must stay on the GPU. Height splashes use `brush_flow_impulse`; drag wakes add
//! mass flux to the surface-wave field via `brush_tilde_wake` after decompose. Do not mutate the
//! CPU [`Image`] for `state` after init: re-uploading that buffer wipes the live GPU field.
//! See `docs/dispersive.md` for the hybrid split and stability notes.
//!
//! Set [`DispersiveController::placement`] and add [`WaterPlugin`](crate::water::WaterPlugin),
//! which [`OceanPlugin`](crate::ocean::OceanPlugin) also brings in, to write the patch into the
//! shared [`WaterLodData`](crate::water::WaterLodData). [`DispersiveLodBlend`] picks whether it
//! adds onto the ocean or replaces it.
//!
//! FFT bin layout matches eWave and the README: DC at `(0,0)`, positive then wrapped-negative bins.
//! `k_xy` in `assets/dispersive/dispersive.wgsl` must stay on that convention. A centered
//! `i - N/2` grid only matches if the spectrum was explicitly fftshifted (ocean path).

mod lod_write;
mod render;

pub mod shaders {
    pub const DISPERSIVE_SURFACE: &str = "embedded://bevy_fft/dispersive/surface.wesl";
    pub const DISPERSIVE_LOD_WRITE: &str = "embedded://bevy_fft/dispersive/lod_write.wesl";
}

pub use render::{
    DispersiveGpuResources, DispersivePipelines, DispersiveSimLabel, DispersiveSimUniform,
    plug_dispersive_render_app, prepare_dispersive_gpu, run_dispersive_sim,
};

use bevy::{
    asset::embedded_asset,
    ecs::{query::QueryItem, system::lifetimeless::Read},
    image::Image,
    pbr::{ExtendedMaterial, MaterialExtension, MaterialPlugin, MeshMaterial3d, StandardMaterial},
    prelude::*,
    reflect::{Reflect, TypePath},
    render::{
        Render, RenderApp, RenderSystems,
        extract_component::{ExtractComponent, ExtractComponentPlugin},
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        render_resource::{AsBindGroup, ShaderType},
        sync_component::SyncComponent,
    },
    shader::ShaderRef,
};

use crate::fft::{FftPlugin, FftSkipStockPipeline};

/// Marks the sim entity: [`FftSource`](crate::fft::FftSource), [`FftTextures`](crate::fft::resources::FftTextures), [`DispersiveGridImages`].
#[derive(Component, Clone, Copy, Default, Reflect)]
pub struct DispersiveSimRoot;

impl SyncComponent<RenderApp> for DispersiveSimRoot {
    type Target = Self;
}

impl ExtractComponent<RenderApp> for DispersiveSimRoot {
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

impl SyncComponent<RenderApp> for DispersiveGridImages {
    type Target = Self;
}

impl ExtractComponent<RenderApp> for DispersiveGridImages {
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
    // Copyable out too, so tools and tests can read the state back.
    image.texture_descriptor.usage = TextureUsages::STORAGE_BINDING
        | TextureUsages::TEXTURE_BINDING
        | TextureUsages::COPY_DST
        | TextureUsages::COPY_SRC;
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
            bar_vel_prev: images.add(rgba32(n)),
        }
    }
}

/// The grids [`FloatingBodyPlugin`](crate::bodies::FloatingBodyPlugin) writes for a dispersive
/// patch, one `R32Float` texel per cell. See [`DispersiveController::enable_bodies`].
#[derive(Clone, Debug, Reflect)]
pub struct DispersiveBodyMaps {
    /// Depth in meters that floating bodies press the surface down by.
    pub pressure_head: Handle<Image>,
    /// Gilligan's obstruction map: 1 in open water, 0 where hulls fill the cell at the waterline.
    pub obstruction: Handle<Image>,
}

/// Wavelength in meters of the transverse waves in the Kelvin wake of a boat moving at `speed`
/// m/s: the deep-water waves whose phase speed matches the boat, λ = 2π U² / g.
pub fn kelvin_wavelength(speed: f32, gravity: f32) -> f32 {
    std::f32::consts::TAU * speed * speed / gravity.max(1e-6)
}

/// Whether water `depth` meters deep is deep enough for the Airy-only mode
/// ([`DispersiveController::deep_water_depth`]) for waves of `wavelength` meters. Waves feel the
/// bottom once it is shallower than about half their wavelength, where the full hybrid has to take
/// over so they shoal and run up.
pub fn deep_mode_suits(depth: f32, wavelength: f32) -> bool {
    depth >= 0.5 * wavelength
}

/// How a placed dispersive patch combines with the water LOD data below it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Reflect)]
pub enum DispersiveLodBlend {
    /// Adds the patch's deviation from rest onto the data below, so ocean waves keep running
    /// through it. Suits wakes and ripples on open water, where the waves superpose linearly.
    #[default]
    Add,
    /// Replaces the data below, including the water depth lane. Suits flooding and shorelines,
    /// where the patch's own bed and depth matter.
    Replace,
}

/// Where a dispersive patch sits in the world, for writing into the shared
/// [`WaterLodData`](crate::water::WaterLodData). The cell width is
/// [`DispersiveController::tile_world`] divided by [`DispersiveController::n`].
#[derive(Clone, Copy, Debug, Reflect)]
pub struct DispersivePlacement {
    /// World X and Z of the patch centre in meters.
    pub center: Vec2,
    /// Height of the solver's rest surface above the surface mesh plane, in meters.
    pub datum: f32,
    /// Width in cells of the band along the patch edge where it fades out.
    pub blend_cells: f32,
    pub blend: DispersiveLodBlend,
}

impl Default for DispersivePlacement {
    fn default() -> Self {
        Self {
            center: Vec2::ZERO,
            datum: 0.0,
            blend_cells: 24.0,
            blend: DispersiveLodBlend::Add,
        }
    }
}

/// Main-world parameters (extracted to render world).
#[derive(Resource, Clone, ExtractResource, Reflect)]
#[extract_app(RenderApp)]
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
    /// Width in cells of the band along each edge where Tessendorf's edge trim (Gilligan §3.6)
    /// absorbs waves, so they neither wrap around the periodic patch nor reflect. Zero turns it
    /// off and leaves the patch periodic.
    pub edge_trim_cells: f32,
    /// Exponent of the trim's taper, `(d / edge_trim_cells)^alpha` at distance `d` from an edge.
    /// Gilligan uses 0.05. Larger values reflect more.
    pub edge_trim_alpha: f32,
    /// Runs the patch in Airy-only deep mode at this depth in meters. It skips the bulk shallow
    /// water and the decomposition, and evolves every wave in Fourier space with
    /// ω² = g k tanh(k depth): Jeschke and Wojtan's Algorithm 2 on its own, in effect Tessendorf's
    /// eWave. Kelvin wakes and their angle come out right at any boat speed, and each step skips
    /// the decomposition's diffusion passes, the bulk step, and three of the four Airy passes.
    /// `None` runs the full hybrid, which shallow water needs for shoaling and run-up.
    /// [`Self::choose_mode`] picks one from the depth under the patch.
    pub deep_water_depth: Option<f32>,
    /// Lets [`FloatingBody`](crate::bodies::FloatingBody)s act on a placed patch. Set it with
    /// [`Self::enable_bodies`].
    pub body_maps: Option<DispersiveBodyMaps>,
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
    /// World placement. When set and [`WaterPlugin`](crate::water::WaterPlugin) is present, the
    /// patch is written into the shared water LOD data every frame.
    pub placement: Option<DispersivePlacement>,
    /// Whole cells [`Self::follow`] has moved the patch by in all. The GPU moves the water by
    /// whatever this has changed by since the last frame it ran.
    pub moved: IVec2,
}

impl DispersiveController {
    /// Width of one cell in meters.
    /// Picks Airy-only deep mode or the full hybrid for water `depth` meters deep under the patch
    /// and wake waves `wavelength` meters long, such as [`kelvin_wavelength`] gives for the boat's
    /// speed. Deep mode starts once the water is 0.6 wavelengths deep and the hybrid takes over
    /// again below 0.4, so a patch near the threshold doesn't flip every frame. Call it each frame.
    pub fn choose_mode(&mut self, depth: f32, wavelength: f32) {
        let ratio = depth / wavelength.max(1e-3);
        if ratio >= 0.6 {
            self.deep_water_depth = Some(depth);
        } else if ratio < 0.4 {
            self.deep_water_depth = None;
        } else if let Some(deep) = &mut self.deep_water_depth {
            *deep = depth;
        }
    }

    /// Lets floating bodies act on the patch, once it has a [`Self::placement`] and
    /// [`FloatingBodyPlugin`](crate::bodies::FloatingBodyPlugin) runs. Waves reflect off their
    /// hulls through Gilligan's obstruction map (§3.3 and §3.4), and in deep mode the hulls press
    /// the surface down, so a moving body trails a wake by itself. [`Self::rebuild`] drops the
    /// maps, since they have the patch's size.
    pub fn enable_bodies(&mut self, images: &mut Assets<Image>) {
        let map = |images: &mut Assets<Image>, value: f32| {
            let mut image = r32_wh(self.n, self.n);
            image.data = Some(value.to_le_bytes().repeat((self.n * self.n) as usize));
            images.add(image)
        };
        self.body_maps = Some(DispersiveBodyMaps {
            pressure_head: map(images, 0.0),
            obstruction: map(images, 1.0),
        });
    }

    /// Keeps the patch centred on world `xz`, such as a boat's position, by moving it whole cells
    /// at a time once `xz` is `margin_cells` or more from the centre along X or Z. The water moves
    /// with the patch without resampling, and the strip that comes in over the leading edge
    /// starts at rest, so the ocean below shows through it. [`Self::pointer`] and
    /// [`Self::pointer_prev`] move with the cells. Call it each frame before setting the brush;
    /// it needs a [`Self::placement`].
    pub fn follow(&mut self, xz: Vec2, margin_cells: f32) {
        let Some(placement) = &mut self.placement else {
            return;
        };
        let cell = self.tile_world / self.n.max(1) as f32;
        let offset = (xz - placement.center) / cell;
        if offset.abs().max_element() < margin_cells.max(1.0) {
            return;
        }
        let step = offset.round().as_ivec2();
        placement.center += step.as_vec2() * cell;
        self.moved += step;
        self.pointer -= step.as_vec2();
        self.pointer_prev -= step.as_vec2();
    }

    pub fn cell_size(&self) -> f32 {
        self.tile_world / self.n.max(1) as f32
    }

    /// Grid coordinates in cells of world `xz`, with cell `i` spanning `i..i + 1`. Uses the
    /// placement centre, or the origin without one.
    pub fn world_to_cells(&self, xz: Vec2) -> Vec2 {
        let center = self.placement.map_or(Vec2::ZERO, |p| p.center);
        (xz - center) / self.cell_size() + 0.5 * self.n as f32
    }

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
            edge_trim_cells: 12.0,
            edge_trim_alpha: 0.05,
            deep_water_depth: None,
            body_maps: None,
            paused: false,
            step_serial: 0,
            sim_apply_serial: 0,
            init_serial: 1,
            brush_active: false,
            brush_radius: 14.0,
            brush_strength: 0.7,
            pointer: Vec2::ZERO,
            pointer_prev: Vec2::ZERO,
            placement: None,
            moved: IVec2::ZERO,
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
        let (edge_trim_cells, edge_trim_alpha) = (self.edge_trim_cells, self.edge_trim_alpha);
        let deep_water_depth = self.deep_water_depth;
        let paused = self.paused;
        let step_serial = self.step_serial;
        let brush_active = self.brush_active;
        let brush_radius = self.brush_radius;
        let brush_strength = self.brush_strength;
        let pointer = self.pointer;
        let pointer_prev = self.pointer_prev;
        let placement = self.placement;
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
        next.edge_trim_cells = edge_trim_cells;
        next.edge_trim_alpha = edge_trim_alpha;
        next.deep_water_depth = deep_water_depth;
        next.paused = paused;
        next.step_serial = step_serial;
        next.brush_active = brush_active;
        next.brush_radius = brush_radius;
        next.brush_strength = brush_strength;
        next.pointer = pointer;
        next.pointer_prev = pointer_prev;
        next.placement = placement;
        // The render side compares against the total it last applied, which a rebuild keeps.
        next.moved = self.moved;
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
        shaders::DISPERSIVE_SURFACE.into()
    }
    fn fragment_shader() -> ShaderRef {
        shaders::DISPERSIVE_SURFACE.into()
    }
    fn deferred_vertex_shader() -> ShaderRef {
        shaders::DISPERSIVE_SURFACE.into()
    }
    fn deferred_fragment_shader() -> ShaderRef {
        shaders::DISPERSIVE_SURFACE.into()
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
        embedded_asset!(app, "surface.wesl");
        embedded_asset!(app, "lod_write.wesl");
        app.register_type::<DispersiveController>()
            .register_type::<DispersivePlacement>()
            .register_type::<DispersiveLodBlend>()
            .register_type::<DispersiveSimRoot>()
            .register_type::<DispersiveGridImages>()
            .register_type::<DispersiveMaterialUniform>()
            .add_plugins(ExtractResourcePlugin::<DispersiveController>::default())
            .add_plugins(ExtractComponentPlugin::<DispersiveSimRoot>::default())
            .add_plugins(ExtractComponentPlugin::<DispersiveGridImages>::default())
            .add_plugins(MaterialPlugin::<DispersiveSurfaceMaterial>::default())
            .register_type::<DispersiveBodyMaps>()
            .add_systems(
                PostUpdate,
                (sync_dispersive_mesh, sync_dispersive_footprint),
            );
    }

    fn finish(&self, app: &mut App) {
        assert!(
            app.is_plugin_added::<FftPlugin>(),
            "DispersivePlugin requires FftPlugin (add FftPlugin before DispersivePlugin)."
        );
        let with_lod = app.is_plugin_added::<crate::water::WaterPlugin>();
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        plug_dispersive_render_app(render_app);
        if with_lod {
            render_app
                .init_resource::<lod_write::DispersiveLodPipeline>()
                .init_resource::<lod_write::DispersiveLodBindGroup>()
                .add_systems(
                    Render,
                    lod_write::prepare_dispersive_lod_write
                        .in_set(RenderSystems::PrepareBindGroups),
                )
                .add_systems(
                    bevy::render::renderer::RenderGraph,
                    lod_write::run_dispersive_lod_write
                        .after(run_dispersive_sim)
                        .in_set(crate::water::WaterLodGraphSystems::Blend),
                );
        }
    }
}

/// Marks the [`BodyFootprintTarget`](crate::bodies::BodyFootprintTarget) kept over the patch.
#[derive(Component)]
struct DispersiveFootprint;

/// Keeps a footprint target over a placed patch with body maps, so floating bodies write into
/// them.
fn sync_dispersive_footprint(
    mut commands: Commands,
    controller: Option<Res<DispersiveController>>,
    mut existing: Query<
        (Entity, &mut crate::bodies::BodyFootprintTarget),
        With<DispersiveFootprint>,
    >,
) {
    let target = controller.as_ref().and_then(|c| {
        let placement = c.placement?;
        let maps = c.body_maps.as_ref()?;
        Some(crate::bodies::BodyFootprintTarget {
            center: placement.center,
            cells: UVec2::splat(c.n),
            cell_size: c.cell_size(),
            height_scale: 1.0,
            pressure_head: maps.pressure_head.clone(),
            obstruction: maps.obstruction.clone(),
        })
    });
    match (target, existing.single_mut()) {
        (Some(target), Ok((_, mut current))) => *current = target,
        (Some(target), Err(_)) => {
            commands.spawn((target, DispersiveFootprint));
        }
        (None, _) => {
            for (entity, _) in &existing {
                commands.entity(entity).despawn();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deep_mode_suits_water_deeper_than_half_the_wake() {
        // A boat at 6 m/s trails transverse waves about 23 m long.
        let wavelength = kelvin_wavelength(6.0, 9.81);
        assert!((wavelength - 23.06).abs() < 0.05, "{wavelength}");
        assert!(deep_mode_suits(12.0, wavelength));
        assert!(!deep_mode_suits(8.0, wavelength));
    }

    #[test]
    fn choose_mode_switches_with_hysteresis() {
        let mut images = Assets::<Image>::default();
        let mut world = World::new();
        let mut controller = DispersiveController::spawn(&mut world.commands(), &mut images, 64);
        let wavelength = 20.0;
        controller.choose_mode(100.0, wavelength);
        assert_eq!(controller.deep_water_depth, Some(100.0));
        // Between 0.4 and 0.6 wavelengths it stays deep, following the depth.
        controller.choose_mode(10.0, wavelength);
        assert_eq!(controller.deep_water_depth, Some(10.0));
        controller.choose_mode(7.0, wavelength);
        assert_eq!(controller.deep_water_depth, None);
        // And stays hybrid until the water is clearly deep again.
        controller.choose_mode(10.0, wavelength);
        assert_eq!(controller.deep_water_depth, None);
        controller.choose_mode(12.0, wavelength);
        assert_eq!(controller.deep_water_depth, Some(12.0));
    }
}
