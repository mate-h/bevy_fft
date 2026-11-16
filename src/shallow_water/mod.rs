//! Staggered shallow water on a heightfield (CMF10-style MacCormack on face velocities, upwind `h`,
//! wet–dry, PML, overshoot limiter). See `docs/shallow_water.md` and `src/shallow_water/simulator.wesl`.
//! Brush and presets mirror [webgpu-shallow-water](https://github.com/mate-h/webgpu-shallow-water) UX only.
//!
//! To drive the solver from another crate, add [`ShallowWaterSolverPlugin`] and build a
//! [`ShallowWaterSolver`] in the render world. It encodes single steps into a compute pass the
//! caller owns, with a bed and water sources that can change every frame.
//!
//! Add [`ShallowWaterPlugin`], insert [`ShallowWaterController`] as a resource with GPU images, and use
//! [`ShallowWaterSurfaceMaterial`] on an XZ plane. The compute pass runs on the root
//! [`RenderGraph`] schedule before the camera driver.
//!
//! To place the grid in a larger scene, set [`ShallowWaterController::placement`] and add
//! [`WaterPlugin`](crate::water::WaterPlugin), which [`OceanPlugin`](crate::ocean::OceanPlugin)
//! also brings in. The plugin then keeps a [`WaterHeightfield`] over the grid, which blends the
//! surface into the shared [`WaterLodData`](crate::water::WaterLodData) each frame, so the ocean surface material and
//! [`OceanQueries`](crate::ocean::OceanQueries) show it and it fades into the ocean at its edges.

mod nest;
mod render;
mod river;
mod shift;
mod solver;

use bevy::{
    asset::{Asset, Assets, Handle, RenderAssetUsages, embedded_asset},
    image::Image,
    pbr::{ExtendedMaterial, MaterialExtension, MeshMaterial3d, StandardMaterial},
    prelude::*,
    reflect::Reflect,
    render::{
        RenderApp,
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        render_resource::{
            AsBindGroup, Extent3d, ShaderType, TextureDimension, TextureFormat, TextureUsages,
        },
    },
    shader::ShaderRef,
};

use crate::water::{WaterHeightfield, WaterHeightfieldDry};

pub use nest::{
    NEST_REFINEMENT, ShallowWaterNest, ShallowWaterNestDescriptor, ShallowWaterNestPipelines,
};
pub use render::{
    ShallowWaterGpuResources, ShallowWaterSimLabel, ShallowWaterTimestamp,
    plug_shallow_water_render_app, prepare_shallow_water_gpu, run_shallow_water_sim,
};
pub use river::{GridEdge, RiverSpline, ShallowWaterBorderFace, ShallowWaterBorderProfile};
pub use shift::{
    ShallowWaterShift, ShallowWaterShiftParams, ShallowWaterShiftPipelines, ShallowWaterWorldBed,
};
pub use solver::{
    ShallowWaterInputs, ShallowWaterKernels, ShallowWaterPipelines, ShallowWaterSettings,
    ShallowWaterSolver, ShallowWaterSolverDescriptor, ShallowWaterTextures, round_particle_count,
};

/// What happens at one edge of the grid.
#[derive(Clone, Copy, Default, PartialEq, Eq, Reflect, Debug)]
pub enum ShallowWaterBorder {
    /// Closed. Water reflects.
    #[default]
    Wall,
    /// Fixed inflow.
    Source,
    /// Fixed outflow.
    Drain,
    /// Sine inflow and outflow, a simple wave maker for standalone tests.
    Waves,
    /// Open edge that follows the surrounding water. Over
    /// [`ShallowWaterController::ocean_band_cells`] cells inside the PML, the water height is
    /// blended toward the target surface (Chentanez, Müller and Kim 2015, eq. 18) and velocities
    /// are left to the solver. The target is the FFT ocean when
    /// [`ShallowWaterController::ocean_forcing`] is on, otherwise the flat
    /// [`ShallowWaterController::pml_eta_rest`].
    Ocean,
}

impl ShallowWaterBorder {
    pub(crate) fn bits(self) -> u32 {
        self as u32
    }
}

/// Where the grid sits in the world, for writing into the shared
/// [`WaterLodData`](crate::water::WaterLodData).
///
/// With a placement the solver runs in meters: [`Self::cell_size`] becomes its cell width, so
/// [`ShallowWaterController::gravity`] is in m/s² and bed and water heights are in meters.
#[derive(Clone, Copy, Debug, Reflect)]
pub struct ShallowWaterPlacement {
    /// World X and Z of the grid centre in meters.
    pub center: Vec2,
    /// Width of one cell in meters.
    pub cell_size: f32,
    /// Height of the solver's zero level above the surface mesh plane, in meters. A surface at
    /// solver height `η` lands at `datum + η`.
    pub datum: f32,
    /// Width in cells of the band along the grid edge where the solver fades into the data
    /// below it, such as the FFT ocean.
    pub blend_cells: f32,
}

impl Default for ShallowWaterPlacement {
    fn default() -> Self {
        Self {
            center: Vec2::ZERO,
            cell_size: 1.0,
            datum: 0.0,
            blend_cells: 16.0,
        }
    }
}

impl ShallowWaterPlacement {
    /// Grid coordinates in cells of world `xz`, with cell `i` spanning `i..i + 1`.
    pub fn world_to_cells(&self, xz: Vec2, cells: UVec2) -> Vec2 {
        (xz - self.center) / self.cell_size + 0.5 * cells.as_vec2()
    }
}

/// Main-world simulation parameters (extracted to the render world each frame).
#[derive(Resource, Clone, ExtractResource, Reflect)]
#[extract_app(RenderApp)]
pub struct ShallowWaterController {
    pub cells_x: u32,
    pub cells_y: u32,
    pub dt: f32,
    pub gravity: f32,
    pub friction: f32,
    pub paused: bool,
    pub particle_count: u32,
    pub left_border: ShallowWaterBorder,
    pub right_border: ShallowWaterBorder,
    pub bottom_border: ShallowWaterBorder,
    pub top_border: ShallowWaterBorder,
    /// 0 = none, 1–4 = bed/water brush, 5 = add velocity along stroke (reference `InteractionMode`).
    pub interaction_mode: u32,
    /// When false, the GPU interact pass skips brush and push-water (mode is forced to none).
    pub brush_input_active: bool,
    pub brush_radius: f32,
    pub brush_force: f32,
    /// Manning roughness n on top of [`Self::friction`]. Zero turns it off.
    pub manning_n: f32,
    /// Fraction of the water depth that evaporates per second.
    pub evaporation: f32,
    /// Optional `Rgba32Float` source texture, one texel per cell. `.x` adds depth per second and
    /// `.y` is a sink weight per step. See [`ShallowWaterInputs::source`].
    pub source: Option<Handle<Image>>,
    /// Optional `R32Float` pressure head per cell, such as [`shallow_r32_image`] makes. Floating
    /// bodies press the water down through it: [`FloatingBodyPlugin`](crate::bodies::FloatingBodyPlugin)
    /// writes their footprints into it, and the solver drives the flow with the gradient of the
    /// surface plus this head. See [`ShallowWaterInputs::pressure_head`].
    pub pressure_head: Option<Handle<Image>>,
    /// Optional per-face border overrides, such as a river's inflow and outflow, from
    /// [`ShallowWaterBorderProfile::image`]. Faces it leaves unset keep their edge's border.
    pub border_profile: Option<Handle<Image>>,
    pub pointer_sim: Vec2,
    pub pointer_prev_sim: Vec2,
    /// Terrain preset index for `loadPreset`. 0 to 3 are procedural, and
    /// [`SHALLOW_WATER_PRESET_BED_SOURCE`] reads [`Self::bed_source`].
    pub preset_index: u32,
    /// Bed height per cell in solver units (`R32Float`, `cells_x × cells_y`), loaded on reset when
    /// [`Self::preset_index`] is [`SHALLOW_WATER_PRESET_BED_SOURCE`]. Use it to start the grid from
    /// a terrain. Starts as a 1×1 placeholder.
    pub bed_source: Handle<Image>,
    /// Still water level that `loadPreset` fills to, in solver height units. Cells whose bed is
    /// above it start dry. Zero leaves every preset dry.
    pub fill_level: f32,
    pub bed_water: Handle<Image>,
    /// Cell-centered velocity (.xy) for particles and diagnostics; .zw unused.
    pub velocity: Handle<Image>,
    /// Staggered face velocity **u** at vertical faces: `(cells_x + 1) × cells_y`.
    pub flow_x: Handle<Image>,
    /// Staggered face velocity **w** at horizontal faces: `cells_x × (cells_y + 1)`.
    pub flow_y: Handle<Image>,
    /// MacCormack temps for u (same layout as `flow_x`): `.x` / `.y` = saved face value and reverse sample.
    pub mac_u_temps: Handle<Image>,
    /// Same for w (layout as `flow_y`).
    pub mac_w_temps: Handle<Image>,
    /// PML auxiliary field per cell, `Rgba32Float` (four channels used in the strip).
    pub pml_state: Handle<Image>,
    /// PML strip width in cells (0 = off).
    pub pml_width: u32,
    /// Target free-surface height **η** in the PML strip (same units as bed + water in the sim).
    pub pml_eta_rest: f32,
    /// Target water height per cell (`R32Float`, `cells_x × cells_y`) for [`ShallowWaterBorder::Ocean`]
    /// edges and the PML, written each frame by [`OceanPlugin`](crate::ocean::OceanPlugin) when
    /// [`Self::ocean_forcing`] is on. The edge strip covering the PML and the ocean band is written
    /// every frame. On reset the whole grid is written so the wet interior can be seeded from it.
    pub ocean_eta: Handle<Image>,
    /// Takes the target surface for ocean edges and the PML from the FFT ocean's long waves, and
    /// seeds the wet interior from those waves on each reset. Needs [`Self::placement`] and
    /// [`OceanPlugin`](crate::ocean::OceanPlugin).
    pub ocean_forcing: bool,
    /// Width in cells of the band inside the PML where [`ShallowWaterBorder::Ocean`] edges blend the
    /// water height toward the target.
    pub ocean_band_cells: f32,
    /// PML absorption ramp: **2** = quadratic (Joh08 / CMF10 default), **3** = cubic.
    pub pml_sigma_exponent: f32,
    /// **0** = uniform damping in each strip; **1** = full Joh08 style scaling by **|u|/|v|** (x strip) and **|w|/|v|** (y strip) from cell-centered velocity.
    pub pml_cosine_blend: f32,
    /// Increment to re-run `clearBuffers` + `loadPreset` on the GPU.
    pub sim_apply_serial: u32,
    /// World placement. When set and [`WaterPlugin`](crate::water::WaterPlugin) is present, the
    /// surface is blended into the shared water LOD data every frame.
    pub placement: Option<ShallowWaterPlacement>,
    /// A finer grid nested in this one that follows a point, such as the camera. Needs
    /// [`Self::placement`]. See [`ShallowWaterNestedGrid`].
    pub nested: Option<ShallowWaterNestedGrid>,
}

/// A world bed height image for the cells a nested grid moves over.
#[derive(Clone, Debug, Reflect)]
pub struct ShallowWaterWorldBedImage {
    /// `R32Float` bed heights in the solver's height units.
    pub image: Handle<Image>,
    /// World XZ of the image's corner in meters.
    pub origin: Vec2,
    /// Width of one texel in meters.
    pub texel_size: f32,
}

/// A grid with a third of the cell width, nested in a placed [`ShallowWaterController`] grid and
/// coupled to it both ways by a [`ShallowWaterNest`]. It moves by whole parent cells to stay
/// centred on [`Self::follow`] and takes the bed of the ground it moves over from
/// [`Self::world_bed`]. It blends into the water LOD data over its parent. Floating bodies, the
/// brush and the border profile act on the parent only.
#[derive(Clone, Debug, Reflect)]
pub struct ShallowWaterNestedGrid {
    /// Cells per edge, a multiple of 3. The grid covers a third as many parent cells.
    pub cells: u32,
    pub images: ShallowWaterImages,
    /// World XZ the grid centres on. It stays inside its parent.
    pub follow: Vec2,
    /// Width of the band along its edge that takes the parent's state, in parent cells.
    pub band: u32,
    /// World bed for the cells the grid moves over. Without one it interpolates the parent's
    /// [`ShallowWaterController::bed_source`].
    pub world_bed: Option<ShallowWaterWorldBedImage>,
    /// Width in its own cells of the band where it fades into its parent in the LOD data.
    pub blend_cells: f32,
}

impl ShallowWaterNestedGrid {
    pub fn new(images: &mut Assets<Image>, cells: u32) -> Self {
        assert!(
            cells.is_multiple_of(NEST_REFINEMENT),
            "nested grid cells must be a multiple of {NEST_REFINEMENT}"
        );
        Self {
            cells,
            images: ShallowWaterImages::new(images, cells, cells),
            follow: Vec2::ZERO,
            band: 1,
            world_bed: None,
            blend_cells: 12.0,
        }
    }
}

impl ShallowWaterController {
    pub fn packed_border_mask(&self) -> u32 {
        self.settings(0, Vec2::X).packed_border_mask()
    }

    /// Builds textures and returns a controller. `cells_*` should be multiples of 32.
    pub fn new(images: &mut Assets<Image>, cells_x: u32, cells_y: u32) -> Self {
        let grid = ShallowWaterImages::new(images, cells_x, cells_y);
        Self {
            cells_x,
            cells_y,
            bed_water: grid.bed_water,
            velocity: grid.velocity,
            flow_x: grid.flow_x,
            flow_y: grid.flow_y,
            mac_u_temps: grid.mac_u_temps,
            mac_w_temps: grid.mac_w_temps,
            pml_state: grid.pml_state,
            ocean_eta: grid.ocean_eta,
            bed_source: images.add(bed_source_image(1, 1, &[0.0])),
            ocean_forcing: false,
            ocean_band_cells: 5.0,
            pml_width: 0,
            pml_eta_rest: 2.0,
            pml_sigma_exponent: 2.0,
            pml_cosine_blend: 0.0,
            dt: 0.08,
            gravity: 10.0,
            friction: 0.0,
            paused: false,
            particle_count: round_particle_count(16 * 1024),
            left_border: ShallowWaterBorder::Wall,
            right_border: ShallowWaterBorder::Wall,
            bottom_border: ShallowWaterBorder::Wall,
            top_border: ShallowWaterBorder::Wall,
            interaction_mode: 3,
            brush_input_active: false,
            brush_radius: 16.0,
            brush_force: 0.75,
            manning_n: 0.0,
            evaporation: 0.0,
            source: None,
            pressure_head: None,
            border_profile: None,
            nested: None,
            pointer_sim: Vec2::ZERO,
            pointer_prev_sim: Vec2::ZERO,
            preset_index: 0,
            fill_level: 0.0,
            sim_apply_serial: 1,
            placement: None,
        }
    }

    /// Cell width the solver uses: [`ShallowWaterPlacement::cell_size`] when placed, otherwise 1.
    pub fn cell_size(&self) -> f32 {
        self.placement.map_or(1.0, |p| p.cell_size)
    }

    /// Recreate grid textures after a resolution change and schedule GPU re-init.
    pub fn rebuild_grid(&mut self, images: &mut Assets<Image>, cells_x: u32, cells_y: u32) {
        let grid = ShallowWaterImages::new(images, cells_x, cells_y);
        self.cells_x = cells_x;
        self.cells_y = cells_y;
        self.bed_water = grid.bed_water;
        self.velocity = grid.velocity;
        self.flow_x = grid.flow_x;
        self.flow_y = grid.flow_y;
        self.mac_u_temps = grid.mac_u_temps;
        self.mac_w_temps = grid.mac_w_temps;
        self.pml_state = grid.pml_state;
        self.ocean_eta = grid.ocean_eta;
        self.sim_apply_serial = self.sim_apply_serial.wrapping_add(1);
    }

    /// Handles of the grid textures.
    pub fn images(&self) -> ShallowWaterImages {
        ShallowWaterImages {
            bed_water: self.bed_water.clone(),
            velocity: self.velocity.clone(),
            flow_x: self.flow_x.clone(),
            flow_y: self.flow_y.clone(),
            mac_u_temps: self.mac_u_temps.clone(),
            mac_w_temps: self.mac_w_temps.clone(),
            pml_state: self.pml_state.clone(),
            ocean_eta: self.ocean_eta.clone(),
        }
    }

    /// Parent cell under the nested grid's cell (0, 0): the grid centred on its
    /// [`ShallowWaterNestedGrid::follow`] point, rounded to whole parent cells and kept inside.
    pub fn nest_offset(&self) -> Option<IVec2> {
        let (placement, nested) = (self.placement?, self.nested.as_ref()?);
        let cells = UVec2::new(self.cells_x, self.cells_y);
        let covered = (nested.cells / NEST_REFINEMENT) as i32;
        let at = placement.world_to_cells(nested.follow, cells) - 0.5 * covered as f32;
        Some(
            at.round()
                .as_ivec2()
                .clamp(IVec2::ZERO, cells.as_ivec2() - covered),
        )
    }

    /// World XZ of the nested grid's centre, from [`Self::nest_offset`].
    pub fn nest_center(&self) -> Option<Vec2> {
        let (placement, nested) = (self.placement?, self.nested.as_ref()?);
        let cells = UVec2::new(self.cells_x, self.cells_y).as_vec2();
        let covered = (nested.cells / NEST_REFINEMENT) as f32;
        let corner = self.nest_offset()?.as_vec2() + 0.5 * covered;
        Some(placement.center + (corner - 0.5 * cells) * placement.cell_size)
    }

    /// Step parameters for the solver at step `timestamp`.
    pub fn settings(&self, timestamp: u32, wind_dir: Vec2) -> ShallowWaterSettings {
        ShallowWaterSettings {
            dt: self.dt,
            cell_size: self.cell_size(),
            gravity: self.gravity,
            friction: self.friction,
            manning_n: self.manning_n,
            evaporation: self.evaporation,
            left_border: self.left_border,
            right_border: self.right_border,
            bottom_border: self.bottom_border,
            top_border: self.top_border,
            pml_width: self.pml_width,
            pml_eta_rest: self.pml_eta_rest,
            pml_sigma_exponent: self.pml_sigma_exponent,
            pml_cosine_blend: self.pml_cosine_blend,
            ocean_band_cells: self.ocean_band_cells,
            ocean_forcing: self.ocean_forcing,
            overshoot_limiter: true,
            second_order_depth: false,
            wind_dir,
            timestamp,
        }
    }
}

/// Main-world handles of the grid textures a [`ShallowWaterSolver`] steps. See
/// [`ShallowWaterTextures`] for the sizes and formats, and [`ShallowWaterTextures::from_images`]
/// for the render-world views.
#[derive(Clone, Debug, Reflect)]
pub struct ShallowWaterImages {
    pub bed_water: Handle<Image>,
    pub velocity: Handle<Image>,
    pub flow_x: Handle<Image>,
    pub flow_y: Handle<Image>,
    pub mac_u_temps: Handle<Image>,
    pub mac_w_temps: Handle<Image>,
    pub pml_state: Handle<Image>,
    pub ocean_eta: Handle<Image>,
}

impl ShallowWaterImages {
    /// Adds zeroed grid images for `cells_x × cells_y` cells. `ocean_eta` starts unset.
    pub fn new(images: &mut Assets<Image>, cells_x: u32, cells_y: u32) -> Self {
        Self {
            bed_water: images.add(shallow_rg32_image(cells_x, cells_y)),
            velocity: images.add(shallow_rg32_image(cells_x, cells_y)),
            flow_x: images.add(shallow_r32_image(cells_x + 1, cells_y)),
            flow_y: images.add(shallow_r32_image(cells_x, cells_y + 1)),
            mac_u_temps: images.add(shallow_rg32_image(cells_x + 1, cells_y)),
            mac_w_temps: images.add(shallow_rg32_image(cells_x, cells_y + 1)),
            pml_state: images.add(shallow_rg32_image(cells_x, cells_y)),
            ocean_eta: images.add(ocean_eta_image(cells_x, cells_y)),
        }
    }
}

/// Empty `Rgba32Float` image for GPU storage (channel meaning is per-field on [`ShallowWaterController`]).
pub fn shallow_rg32_image(width: u32, height: u32) -> Image {
    let mut image = Image::new_fill(
        Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &[0u8; 16],
        TextureFormat::Rgba32Float,
        RenderAssetUsages::default(),
    );
    image.texture_descriptor.usage = TextureUsages::STORAGE_BINDING
        | TextureUsages::TEXTURE_BINDING
        | TextureUsages::COPY_SRC
        | TextureUsages::COPY_DST;
    image
}

pub fn shallow_r32_image(width: u32, height: u32) -> Image {
    let mut image = Image::new_fill(
        Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &[0u8; 4],
        TextureFormat::R32Float,
        RenderAssetUsages::default(),
    );
    image.texture_descriptor.usage = TextureUsages::STORAGE_BINDING
        | TextureUsages::TEXTURE_BINDING
        | TextureUsages::COPY_SRC
        | TextureUsages::COPY_DST;
    image
}

/// Value the ocean target texture starts with. The solver treats cells below `-1e29` as not yet
/// written and falls back to [`ShallowWaterController::pml_eta_rest`].
const OCEAN_ETA_UNSET: f32 = -1.0e30;

/// [`ShallowWaterController::preset_index`] that loads the bed from
/// [`ShallowWaterController::bed_source`].
pub const SHALLOW_WATER_PRESET_BED_SOURCE: u32 = 4;

/// `R32Float` bed image for [`ShallowWaterController::bed_source`], from `heights` in row order
/// with `width` cells per row. Row `j` is grid row `j`, which runs along world +Z when placed.
pub fn bed_source_image(width: u32, height: u32, heights: &[f32]) -> Image {
    assert_eq!(heights.len(), (width * height) as usize);
    let mut image = shallow_r32_image(width, height);
    image.texture_descriptor.usage = TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST;
    image.data = Some(heights.iter().flat_map(|h| h.to_le_bytes()).collect());
    image
}

fn ocean_eta_image(width: u32, height: u32) -> Image {
    let mut image = shallow_r32_image(width, height);
    image.data = Some(
        OCEAN_ETA_UNSET
            .to_le_bytes()
            .repeat((width * height) as usize),
    );
    image
}

#[derive(Clone, Copy, Default, ShaderType)]
pub struct GpuInteractionUniform {
    pub mode: u32,
    pub radius: f32,
    pub force: f32,
    pub dt: f32,
    pub old_position: Vec2,
    pub position: Vec2,
    pub preset: u32,
    pub fill_level: f32,
}

#[derive(Clone, Copy, Default, ShaderType)]
pub struct GpuSimulationUniform {
    pub size: UVec2,
    pub dt: f32,
    pub dx: f32,
    pub gravity: f32,
    pub friction_factor: f32,
    pub timestamp: u32,
    pub border_mask: u32,
    pub pml_width: u32,
    /// Bit 0 enables overshoot reduction. Bit 1 reads the target surface from the ocean target
    /// texture instead of `pml_eta_rest`. Bit 2 applies the source texture.
    pub flags: u32,
    pub pml_eta_rest: f32,
    pub vel_clamp_alpha: f32,
    pub h_avgmax_beta: f32,
    pub eps_wet: f32,
    pub pml_lambda_decay: f32,
    pub pml_lambda_update: f32,
    pub pml_sigma_max: f32,
    pub overshoot_alpha: f32,
    pub overshoot_lambda_edge: f32,
    pub pml_sigma_exponent: f32,
    pub pml_cosine_blend: f32,
    pub ocean_band_cells: f32,
    /// Manning roughness squared. Zero turns Manning friction off.
    pub manning_n2: f32,
    /// Fraction of the water depth that evaporates per second.
    pub evaporation: f32,
    /// Unit wind direction on XZ. The reset seed points column velocity along it.
    pub wind_dir: Vec2,
}

#[derive(Clone, Copy, Default, ShaderType)]
pub struct GpuParticle {
    pub position: Vec2,
    pub lifetime: u32,
    pub alive: u32,
}

#[derive(Clone, Copy, Default, Reflect, ShaderType)]
pub struct ShallowWaterMaterialUniform {
    pub height_scale: f32,
    pub tile_world_size: f32,
    pub grid_size: f32,
    pub _pad0: f32,
}

#[derive(Asset, TypePath, AsBindGroup, Clone)]
pub struct ShallowWaterSurfaceExtension {
    #[uniform(100)]
    pub settings: ShallowWaterMaterialUniform,
    #[texture(101)]
    #[sampler(102)]
    pub bed_water: Handle<Image>,
}

impl Default for ShallowWaterSurfaceExtension {
    fn default() -> Self {
        Self {
            settings: ShallowWaterMaterialUniform {
                height_scale: 0.08,
                tile_world_size: 32.0,
                grid_size: 256.0,
                _pad0: 0.0,
            },
            bed_water: Handle::default(),
        }
    }
}

impl MaterialExtension for ShallowWaterSurfaceExtension {
    fn vertex_shader() -> ShaderRef {
        shaders::SHALLOW_WATER_SURFACE.into()
    }

    fn fragment_shader() -> ShaderRef {
        shaders::SHALLOW_WATER_SURFACE.into()
    }

    fn deferred_vertex_shader() -> ShaderRef {
        shaders::SHALLOW_WATER_SURFACE.into()
    }

    fn deferred_fragment_shader() -> ShaderRef {
        shaders::SHALLOW_WATER_SURFACE.into()
    }
}

pub type ShallowWaterSurfaceMaterial =
    ExtendedMaterial<StandardMaterial, ShallowWaterSurfaceExtension>;

#[derive(Component)]
pub struct ShallowWaterSurfaceTag;

pub mod shaders {
    pub const SHALLOW_WATER_SIMULATOR: &str = "embedded://bevy_fft/shallow_water/simulator.wesl";
    pub const SHALLOW_WATER_SURFACE: &str = "embedded://bevy_fft/shallow_water/surface.wesl";
    pub const SHALLOW_WATER_NEST: &str = "embedded://bevy_fft/shallow_water/nest.wesl";
    pub const SHALLOW_WATER_SHIFT: &str = "embedded://bevy_fft/shallow_water/shift.wesl";
}

fn sync_shallow_water_mesh_material(
    sim: Res<ShallowWaterController>,
    mut materials: ResMut<Assets<ShallowWaterSurfaceMaterial>>,
    q: Query<&MeshMaterial3d<ShallowWaterSurfaceMaterial>, With<ShallowWaterSurfaceTag>>,
) {
    let Ok(h) = q.single() else {
        return;
    };
    let Some(mut mat) = materials.get_mut(&h.0) else {
        return;
    };
    let g = sim.cells_x.max(sim.cells_y) as f32;
    mat.extension.settings.grid_size = g;
    mat.extension.bed_water = sim.bed_water.clone();
}

/// Marks the [`WaterHeightfield`] that [`ShallowWaterPlugin`] keeps over the controller's grid.
#[derive(Component)]
struct ShallowWaterHeightfield;

/// Keeps a [`WaterHeightfield`] over the grid while [`ShallowWaterController::placement`] is set.
/// The placed solver runs in meters, so heights and velocities need no scaling.
fn sync_shallow_water_heightfield(
    mut commands: Commands,
    sim: Res<ShallowWaterController>,
    mut existing: Query<
        (
            Entity,
            &mut WaterHeightfield,
            Has<ShallowWaterNestHeightfield>,
        ),
        With<ShallowWaterHeightfield>,
    >,
) {
    let Some(placement) = sim.placement else {
        for (entity, _, _) in &existing {
            commands.entity(entity).despawn();
        }
        return;
    };
    let heightfield = WaterHeightfield {
        pressure_head: sim.pressure_head.clone(),
        center: placement.center,
        cell_size: placement.cell_size,
        datum: placement.datum,
        blend_cells: placement.blend_cells,
        dry: WaterHeightfieldDry::Level(sim.fill_level),
        ..WaterHeightfield::new(
            sim.bed_water.clone(),
            sim.velocity.clone(),
            UVec2::new(sim.cells_x, sim.cells_y),
        )
    };
    match existing.iter_mut().find(|(_, _, nested)| !nested) {
        Some((_, mut current, _)) => *current = heightfield.clone(),
        None => {
            commands.spawn((heightfield.clone(), ShallowWaterHeightfield));
        }
    }
    let nested = sim
        .nested
        .as_ref()
        .zip(sim.nest_center())
        .map(|(nested, center)| WaterHeightfield {
            center,
            cell_size: placement.cell_size / NEST_REFINEMENT as f32,
            blend_cells: nested.blend_cells,
            pressure_head: None,
            order: heightfield.order + 1,
            dry: heightfield.dry,
            datum: heightfield.datum,
            ..WaterHeightfield::new(
                nested.images.bed_water.clone(),
                nested.images.velocity.clone(),
                UVec2::splat(nested.cells),
            )
        });
    let current = existing.iter_mut().find(|(_, _, nested)| *nested);
    match (nested, current) {
        (Some(nested), Some((_, mut current, _))) => *current = nested,
        (Some(nested), None) => {
            commands.spawn((nested, ShallowWaterHeightfield, ShallowWaterNestHeightfield));
        }
        (None, Some((entity, _, _))) => commands.entity(entity).despawn(),
        (None, None) => {}
    }
}

/// Marks the [`WaterHeightfield`] of the controller's nested grid.
#[derive(Component)]
struct ShallowWaterNestHeightfield;

/// Registers the solver shader and queues [`ShallowWaterPipelines`] in the render world, for
/// callers that build and step a [`ShallowWaterSolver`] themselves. [`ShallowWaterPlugin`] adds it.
pub struct ShallowWaterSolverPlugin;

impl Plugin for ShallowWaterSolverPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "simulator.wesl");
        embedded_asset!(app, "nest.wesl");
        embedded_asset!(app, "shift.wesl");
    }

    fn finish(&self, app: &mut App) {
        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .init_resource::<ShallowWaterPipelines>()
                .init_resource::<ShallowWaterNestPipelines>()
                .init_resource::<ShallowWaterShiftPipelines>();
        }
    }
}

/// The controller-driven simulation: presets, brush, particles, PML, ocean forcing, and the water
/// LOD blend, stepped once per frame from [`ShallowWaterController`].
pub struct ShallowWaterPlugin;

impl Plugin for ShallowWaterPlugin {
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<ShallowWaterSolverPlugin>() {
            app.add_plugins(ShallowWaterSolverPlugin);
        }
        embedded_asset!(app, "surface.wesl");

        app.register_type::<ShallowWaterController>()
            .register_type::<ShallowWaterBorder>()
            .register_type::<ShallowWaterPlacement>()
            .register_type::<ShallowWaterMaterialUniform>()
            .add_plugins(ExtractResourcePlugin::<ShallowWaterController>::default())
            .add_plugins(MaterialPlugin::<ShallowWaterSurfaceMaterial>::default())
            .add_systems(
                PostUpdate,
                (
                    sync_shallow_water_mesh_material,
                    sync_shallow_water_heightfield,
                ),
            );
    }

    fn finish(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        plug_shallow_water_render_app(render_app);
    }
}
