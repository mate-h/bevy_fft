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

mod banks;
mod nest;
mod render;
mod river;
mod settled;
mod shift;
mod solver;
mod waterfalls;

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

pub use banks::{
    BankLeak, ClampedFall, LowBank, RiverBankCheck, RiverBankReport, SettledGrid,
    check_river_banks, clamped_falls,
};
pub use nest::{
    NEST_REFINEMENT, ShallowWaterNest, ShallowWaterNestDescriptor, ShallowWaterNestPipelines,
};
pub use render::{
    ShallowWaterGpuResources, ShallowWaterSimLabel, ShallowWaterTimestamp,
    plug_shallow_water_render_app, prepare_shallow_water_gpu, run_shallow_water_sim,
};
pub use river::{GridEdge, RiverSpline, ShallowWaterBorderFace, ShallowWaterBorderProfile};
pub use settled::{ShallowWaterSettled, capture_shallow_water_settled};
pub use shift::{
    ShallowWaterShift, ShallowWaterShiftParams, ShallowWaterShiftPipelines, ShallowWaterWorldBed,
};
pub use solver::{
    ShallowWaterInputs, ShallowWaterKernels, ShallowWaterPipelines, ShallowWaterSettings,
    ShallowWaterSolver, ShallowWaterSolverDescriptor, ShallowWaterTextures, round_particle_count,
};
pub use waterfalls::ShallowWaterWaterfalls;

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
    /// Advects velocity in momentum-conserving form, after Stelling and Duinmeijer (2003), so
    /// bores and flood fronts move at the right speed. See
    /// [`ShallowWaterSettings::momentum_conserving`].
    pub momentum_conserving: bool,
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
    /// Finer grids nested in this one that follow a point, such as the camera, each nested in
    /// the one before it: the first in this grid, the second in the first, and so on. Needs
    /// [`Self::placement`]. See [`ShallowWaterNestedGrid`] and [`Self::nest_levels`].
    pub nested: Vec<ShallowWaterNestedGrid>,
    /// This frame's [`ShallowWaterMover`]s, gathered each frame. The first
    /// [`SHALLOW_WATER_MAX_MOVERS`] stir the water.
    #[reflect(ignore)]
    pub movers: Vec<ShallowWaterMoverSample>,
    /// A settled state to start from. Each reset loads it over the preset when it
    /// [`fits`](ShallowWaterSettled::fits) the grid, after the ocean seed, and the nested grids
    /// then fill from it.
    #[reflect(ignore)]
    pub settled: Option<std::sync::Arc<ShallowWaterSettled>>,
    /// Spray thrown off waterfall lips into the water particles. Needs a placement.
    pub waterfalls: Option<ShallowWaterWaterfalls>,
}

/// Most [`ShallowWaterMover`]s that stir the water in one frame.
pub const SHALLOW_WATER_MAX_MOVERS: usize = 16;

/// Stirs the shallow water with its entity's motion, only while it moves, as a wading character, a
/// paddle or a dragged log does. Within `radius` of the entity's position, each face velocity on
/// every grid, nested or not, is pulled toward the entity's horizontal velocity at `rate` per
/// second, where the water is slower in that direction. Never slowed, still water stays as it is
/// around a body at rest, so a mover is no obstacle: the bed and floating bodies' pressure head
/// block the flow. The stir needs the entity's lowest point, `depth` below its position, under the
/// water surface.
#[derive(Component, Clone, Debug, Reflect)]
#[reflect(Component)]
pub struct ShallowWaterMover {
    /// Meters.
    pub radius: f32,
    /// Share of the speed difference closed per second.
    pub rate: f32,
    /// Meters from the entity's position down to its lowest point.
    pub depth: f32,
    /// Slower motion, in m/s, doesn't stir.
    pub min_speed: f32,
    /// The entity's position last frame, for its velocity.
    #[reflect(ignore)]
    pub previous: Option<Vec3>,
}

impl Default for ShallowWaterMover {
    fn default() -> Self {
        Self {
            radius: 0.5,
            rate: 8.0,
            depth: 1.0,
            min_speed: 0.05,
            previous: None,
        }
    }
}

/// One [`ShallowWaterMover`] this frame, in world units.
#[derive(Clone, Copy, Debug, Default)]
pub struct ShallowWaterMoverSample {
    pub position: Vec2,
    pub velocity: Vec2,
    pub radius: f32,
    /// Share of the speed difference closed this frame, from [`ShallowWaterMover::rate`].
    pub blend: f32,
    /// World height of the mover's lowest point.
    pub bottom: f32,
}

impl ShallowWaterMoverSample {
    /// The two `vec4`s of `InteractionSettings::movers` for a grid with this `placement`, `cells`
    /// and `datum`: position and radius in cells and the blend, then velocity in m/s and the
    /// lowest point in solver height units.
    fn gpu(&self, placement: &ShallowWaterPlacement, cells: UVec2) -> [Vec4; 2] {
        let at = placement.world_to_cells(self.position, cells);
        [
            Vec4::new(at.x, at.y, self.radius / placement.cell_size, self.blend),
            Vec4::new(
                self.velocity.x,
                self.velocity.y,
                self.bottom - placement.datum,
                0.0,
            ),
        ]
    }
}

/// Packs `movers` for a grid into `InteractionSettings::mover_count` and `::movers`.
pub(crate) fn pack_movers(
    movers: &[ShallowWaterMoverSample],
    placement: Option<&ShallowWaterPlacement>,
    cells: UVec2,
) -> (u32, [Vec4; 2 * SHALLOW_WATER_MAX_MOVERS]) {
    let mut packed = [Vec4::ZERO; 2 * SHALLOW_WATER_MAX_MOVERS];
    let Some(placement) = placement else {
        return (0, packed);
    };
    let count = movers.len().min(SHALLOW_WATER_MAX_MOVERS);
    for (i, mover) in movers[..count].iter().enumerate() {
        let [a, b] = mover.gpu(placement, cells);
        packed[2 * i] = a;
        packed[2 * i + 1] = b;
    }
    (count as u32, packed)
}

/// Gathers the moving [`ShallowWaterMover`]s into [`ShallowWaterController::movers`].
fn gather_shallow_water_movers(
    time: Res<Time>,
    controller: Option<ResMut<ShallowWaterController>>,
    mut movers: Query<(&mut ShallowWaterMover, &GlobalTransform)>,
) {
    let Some(mut controller) = controller else {
        return;
    };
    let dt = time.delta_secs();
    controller.movers.clear();
    for (mut mover, transform) in &mut movers {
        let position = transform.translation();
        let previous = mover.previous.replace(position);
        let Some(previous) = previous.filter(|_| dt > 0.0) else {
            continue;
        };
        let velocity = (position - previous).xz() / dt;
        if velocity.length() < mover.min_speed {
            continue;
        }
        controller.movers.push(ShallowWaterMoverSample {
            position: position.xz(),
            velocity,
            radius: mover.radius,
            blend: 1.0 - (-mover.rate.max(0.0) * dt).exp(),
            bottom: position.y - mover.depth,
        });
    }
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

/// A grid with a third of the cell width, nested in a placed [`ShallowWaterController`] grid or
/// in the nested grid before it in [`ShallowWaterController::nested`], and coupled to that parent
/// both ways by a [`ShallowWaterNest`]. It moves by whole parent cells to stay centred on
/// [`Self::follow`] and takes the bed of the ground it moves over from [`Self::world_bed`]. It
/// blends into the water LOD data over its parent. The brush and the border profile act on the
/// controller's grid only. [`ShallowWaterMover`]s stir every grid, and floating bodies press on
/// every grid, a nested one through [`Self::pressure_head`].
#[derive(Clone, Debug, Reflect)]
pub struct ShallowWaterNestedGrid {
    /// Cells per edge, a multiple of 3. The grid covers a third as many parent cells.
    pub cells: u32,
    pub images: ShallowWaterImages,
    /// World XZ the grid centres on. It stays inside its parent, and inside its parent's band
    /// when the parent is nested too.
    pub follow: Vec2,
    /// Width of the band along its edge that takes the parent's state, in parent cells.
    pub band: u32,
    /// World bed for the cells the grid moves over. Without one it interpolates its parent's bed:
    /// [`ShallowWaterController::bed_source`] for the first nested grid, the bed in the previous
    /// nested grid's cells for the others.
    pub world_bed: Option<ShallowWaterWorldBedImage>,
    /// Steps per parent step. `None` takes as many as [`Self::stable_steps`] says its stability
    /// limit needs. [`Self::new`] sets 3, which refines time as space.
    pub steps: Option<u32>,
    /// Fastest flow, in m/s, that [`Self::stable_steps`] plans for.
    pub max_speed: f32,
    /// Deepest water, in meters, that [`Self::stable_steps`] plans for.
    pub max_depth: f32,
    /// Fastest fall, in m/s, that [`Self::stable_steps`] keeps under the face velocity clamp of
    /// 0.5 Δx/Δt. Water reaches 7.7 m/s after a 3 m drop.
    pub fall_speed: f32,
    /// Advects velocity in momentum-conserving form. `None` turns it on for cells under 0.5 m and
    /// otherwise takes the parent's setting. See [`ShallowWaterSettings::momentum_conserving`].
    pub momentum_conserving: Option<bool>,
    /// Width in its own cells of the band where it fades into its parent in the LOD data.
    pub blend_cells: f32,
    /// The grid's own pressure head, one texel per cell, which floating bodies write into as
    /// they do the parent's. It acts while the parent has a
    /// [`ShallowWaterController::pressure_head`].
    pub pressure_head: Handle<Image>,
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
            steps: Some(NEST_REFINEMENT),
            max_speed: 3.0,
            max_depth: 1.0,
            fall_speed: 7.7,
            momentum_conserving: None,
            blend_cells: 12.0,
            pressure_head: images.add(shallow_r32_image(cells, cells)),
        }
    }

    /// Steps per parent step of `parent_dt` that keep a grid of `cell_size` cells stable: each at
    /// most 0.5 Δx / (`max_speed` + √(g `max_depth`)), and short enough that `fall_speed` stays
    /// under the velocity clamp 0.5 Δx / Δt. At least 1.
    pub fn stable_steps(&self, parent_dt: f32, cell_size: f32, gravity: f32) -> u32 {
        let wave = self.max_speed + (gravity * self.max_depth).max(0.0).sqrt();
        let dt =
            (0.5 * cell_size / wave.max(1e-6)).min(0.5 * cell_size / self.fall_speed.max(1e-6));
        ((parent_dt / dt).ceil() as u32).max(1)
    }
}

/// Where one of [`ShallowWaterController::nested`] sits this frame and how it steps, from
/// [`ShallowWaterController::nest_levels`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShallowWaterNestLevel {
    /// Parent cell under the grid's cell (0, 0).
    pub offset: IVec2,
    /// The grid's cell (0, 0) counted from the controller grid's corner in the grid's own cells.
    /// It changes only when the grid moves over the world.
    pub corner_cells: IVec2,
    /// World XZ of the grid's corner.
    pub corner: Vec2,
    /// World XZ of the grid's centre.
    pub center: Vec2,
    pub cell_size: f32,
    /// Cells per edge.
    pub cells: u32,
    /// Steps per parent step.
    pub steps: u32,
    /// Time step, the parent's over [`Self::steps`].
    pub dt: f32,
    pub momentum_conserving: bool,
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
            momentum_conserving: false,
            evaporation: 0.0,
            source: None,
            pressure_head: None,
            border_profile: None,
            nested: Vec::new(),
            movers: Vec::new(),
            settled: None,
            waterfalls: None,
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

    /// Where each of [`Self::nested`] sits this frame and how it steps, in order. Each grid
    /// centres on its [`ShallowWaterNestedGrid::follow`] point, rounded to whole parent cells and
    /// kept inside its parent, and inside the parent's band when the parent is nested too. The
    /// list stops at the first grid that doesn't fit, and is empty without a placement.
    pub fn nest_levels(&self) -> Vec<ShallowWaterNestLevel> {
        let mut levels = Vec::new();
        let Some(placement) = self.placement else {
            return levels;
        };
        let root_cells = UVec2::new(self.cells_x, self.cells_y);
        let root_corner = placement.center - 0.5 * root_cells.as_vec2() * placement.cell_size;
        let rho = NEST_REFINEMENT as i32;
        let mut parent = NestParent {
            cells: root_cells.as_ivec2(),
            corner: root_corner,
            cell_size: placement.cell_size,
            margin: 0,
            corner_cells: IVec2::ZERO,
            dt: self.dt,
            momentum_conserving: self.momentum_conserving,
        };
        for nested in &self.nested {
            let covered = (nested.cells / NEST_REFINEMENT) as i32;
            let (lo, hi) = (
                IVec2::splat(parent.margin),
                parent.cells - parent.margin - covered,
            );
            if covered == 0 || hi.cmplt(lo).any() {
                break;
            }
            let at = (nested.follow - parent.corner) / parent.cell_size - 0.5 * covered as f32;
            let offset = at.round().as_ivec2().clamp(lo, hi);
            let cell_size = parent.cell_size / NEST_REFINEMENT as f32;
            let corner_cells = (parent.corner_cells + offset) * rho;
            let corner = root_corner + corner_cells.as_vec2() * cell_size;
            let steps = nested
                .steps
                .unwrap_or_else(|| nested.stable_steps(parent.dt, cell_size, self.gravity))
                .max(1);
            let level = ShallowWaterNestLevel {
                offset,
                corner_cells,
                corner,
                center: corner + 0.5 * nested.cells as f32 * cell_size,
                cell_size,
                cells: nested.cells,
                steps,
                dt: parent.dt / steps as f32,
                momentum_conserving: nested
                    .momentum_conserving
                    .unwrap_or(cell_size < 0.5 || parent.momentum_conserving),
            };
            levels.push(level);
            parent = NestParent {
                cells: IVec2::splat(nested.cells as i32),
                corner,
                cell_size,
                margin: nested.band as i32 * rho,
                corner_cells,
                dt: level.dt,
                momentum_conserving: level.momentum_conserving,
            };
        }
        levels
    }

    /// Parent cell under the first nested grid's cell (0, 0). See [`Self::nest_levels`].
    pub fn nest_offset(&self) -> Option<IVec2> {
        self.nest_levels().first().map(|l| l.offset)
    }

    /// World XZ of the first nested grid's centre. See [`Self::nest_levels`].
    pub fn nest_center(&self) -> Option<Vec2> {
        self.nest_levels().first().map(|l| l.center)
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
            momentum_conserving: self.momentum_conserving,
            wind_dir,
            timestamp,
        }
    }
}

/// The grid a nested grid sits in, for [`ShallowWaterController::nest_levels`].
struct NestParent {
    cells: IVec2,
    corner: Vec2,
    cell_size: f32,
    /// Cells along each edge a child keeps clear of: the parent's own band.
    margin: i32,
    /// The parent's corner from the controller grid's corner, in the parent's cells.
    corner_cells: IVec2,
    dt: f32,
    momentum_conserving: bool,
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
    /// [`ShallowWaterMover`]s in `movers`.
    pub mover_count: u32,
    /// Two per mover, from [`ShallowWaterMoverSample`].
    pub movers: [Vec4; 2 * SHALLOW_WATER_MAX_MOVERS],
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
    pub const SHALLOW_WATER_WATERFALLS: &str = "embedded://bevy_fft/shallow_water/waterfalls.wesl";
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
            Option<&ShallowWaterNestHeightfield>,
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
        dry: WaterHeightfieldDry::Extend(sim.fill_level),
        ..WaterHeightfield::new(
            sim.bed_water.clone(),
            sim.velocity.clone(),
            UVec2::new(sim.cells_x, sim.cells_y),
        )
    };
    match existing.iter_mut().find(|(_, _, level)| level.is_none()) {
        Some((_, mut current, _)) => *current = heightfield.clone(),
        None => {
            commands.spawn((heightfield.clone(), ShallowWaterHeightfield));
        }
    }
    let levels = sim.nest_levels();
    for (entity, _, level) in &existing {
        if level.is_some_and(|l| l.0 >= levels.len()) {
            commands.entity(entity).despawn();
        }
    }
    for (k, (nested, level)) in sim.nested.iter().zip(&levels).enumerate() {
        let field = WaterHeightfield {
            center: level.center,
            cell_size: level.cell_size,
            blend_cells: nested.blend_cells,
            // Bodies press on the nested grids wherever they press on the parent.
            pressure_head: sim
                .pressure_head
                .as_ref()
                .map(|_| nested.pressure_head.clone()),
            // Each grid blends over its parent.
            order: heightfield.order + 1 + k as i32,
            dry: heightfield.dry,
            datum: heightfield.datum,
            ..WaterHeightfield::new(
                nested.images.bed_water.clone(),
                nested.images.velocity.clone(),
                UVec2::splat(nested.cells),
            )
        };
        match existing
            .iter_mut()
            .find(|(_, _, level)| level.is_some_and(|l| l.0 == k))
        {
            Some((_, mut current, _)) => *current = field,
            None => {
                commands.spawn((
                    field,
                    ShallowWaterHeightfield,
                    ShallowWaterNestHeightfield(k),
                ));
            }
        }
    }
}

/// Marks the [`WaterHeightfield`] of one of the controller's nested grids, by its index in
/// [`ShallowWaterController::nested`].
#[derive(Component, Clone, Copy)]
struct ShallowWaterNestHeightfield(usize);

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
        embedded_asset!(app, "waterfalls.wesl");

        app.register_type::<ShallowWaterController>()
            .register_type::<ShallowWaterBorder>()
            .register_type::<ShallowWaterPlacement>()
            .register_type::<ShallowWaterMover>()
            .register_type::<ShallowWaterMaterialUniform>()
            .add_plugins(ExtractResourcePlugin::<ShallowWaterController>::default())
            .add_plugins(MaterialPlugin::<ShallowWaterSurfaceMaterial>::default())
            .add_systems(
                PostUpdate,
                (
                    sync_shallow_water_mesh_material,
                    sync_shallow_water_heightfield,
                    gather_shallow_water_movers.after(bevy::transform::TransformSystems::Propagate),
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
