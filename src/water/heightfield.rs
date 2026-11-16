//! Blends heightfield water, such as a shallow water grid, into the shared [`WaterLodData`] slices
//! each frame.
//!
//! Any solver that keeps a bed, a water depth, and a velocity per cell can show up on the
//! [`WaterSurfaceMaterial`](super::WaterSurfaceMaterial) this way. Spawn a [`WaterHeightfield`]
//! that points at its textures and says where the grid sits. The solver itself stays outside this
//! module: it only has to have written the textures before
//! [`WaterLodGraphSystems::Blend`](super::WaterLodGraphSystems::Blend) runs.

use bevy::{
    ecs::system::lifetimeless::Read,
    prelude::*,
    render::{
        RenderApp,
        diagnostic::RecordDiagnostics,
        extract_component::ExtractComponent,
        render_asset::RenderAssets,
        render_resource::{binding_types::*, *},
        renderer::{RenderContext, RenderDevice, RenderQueue},
        sync_component::SyncComponent,
        texture::GpuImage,
    },
};

use crate::{
    ocean::{OCEAN_MAX_CASCADES, OceanLodWriteFrame},
    water::{
        WATER_LOD_DRIFT_FORMAT, WATER_LOD_SURFACE_FORMAT, WaterLodData, WaterLodGpu,
        WaterLodScratch, WaterLodSlices,
    },
};

pub const WATER_HEIGHTFIELD_SHADER: &str = "embedded://bevy_fft/water/heightfield.wesl";

/// Where the surface of a dry cell sits.
#[derive(Clone, Copy, Debug, PartialEq, Reflect)]
pub enum WaterHeightfieldDry {
    /// Dry cells render at this still-water level, in the source's height units, so the mesh
    /// does not follow the bed up onto a beach.
    Level(f32),
    /// Dry cells render on the bed, so a river running down a slope meets its banks instead of
    /// dropping to one level. Set [`WaterSurfaceUniform::dry_depth`](super::WaterSurfaceUniform)
    /// so the material discards the dry ground.
    Bed,
}

/// A grid of water cells that the water LOD data blends in each frame.
///
/// `bed_water` holds the bed height in `.x` and the water depth in `.y`, and `velocity` holds the
/// depth-averaged velocity at cell centres in `.xy`, one texel per cell. Both are read with
/// loads, so any float format works. Row `j` of the grid runs along world +Z.
///
/// Inside the grid the heightfield replaces the data below it. Along the grid edge the
/// replacement fades into that data over [`Self::blend_cells`].
#[derive(Component, Clone, Debug, Reflect)]
pub struct WaterHeightfield {
    pub bed_water: Handle<Image>,
    pub velocity: Handle<Image>,
    /// Grid size in cells.
    pub cells: UVec2,
    /// World X and Z of the grid centre in meters.
    pub center: Vec2,
    /// Width of one cell in meters.
    pub cell_size: f32,
    /// Height of the source's zero level above the surface mesh plane, in meters.
    pub datum: f32,
    /// Width in cells of the band along the grid edge where the heightfield fades into the data
    /// below it.
    pub blend_cells: f32,
    /// Meters per unit of bed height and water depth.
    pub height_scale: f32,
    /// Meters per second per unit of velocity.
    pub velocity_scale: f32,
    pub dry: WaterHeightfieldDry,
}

impl WaterHeightfield {
    /// A grid of `cells` one-meter cells centred on the origin, in meters and meters per second,
    /// that renders dry cells on the bed.
    pub fn new(bed_water: Handle<Image>, velocity: Handle<Image>, cells: UVec2) -> Self {
        Self {
            bed_water,
            velocity,
            cells,
            center: Vec2::ZERO,
            cell_size: 1.0,
            datum: 0.0,
            blend_cells: 16.0,
            height_scale: 1.0,
            velocity_scale: 1.0,
            dry: WaterHeightfieldDry::Bed,
        }
    }
}

impl SyncComponent<RenderApp> for WaterHeightfield {
    type Target = Self;
}

impl ExtractComponent<RenderApp> for WaterHeightfield {
    type QueryData = Read<Self>;
    type QueryFilter = ();
    type Out = Self;

    fn extract_component(
        item: bevy::ecs::query::QueryItem<'_, '_, Self::QueryData>,
    ) -> Option<Self::Out> {
        Some(item.clone())
    }
}

/// Uniform for `heightfield.wesl`.
#[derive(Clone, Copy, Default, ShaderType)]
struct WaterHeightfieldParams {
    center: Vec2,
    cell_size: f32,
    datum: f32,
    cells: UVec2,
    blend_cells: f32,
    /// Still-water level in meters for [`WaterHeightfieldDry::Level`].
    dry_level: f32,
    cascade_tile_sizes: Vec4,
    amplitude: f32,
    choppiness: f32,
    cascade_count: u32,
    /// 0 for [`WaterHeightfieldDry::Level`], 1 for [`WaterHeightfieldDry::Bed`].
    dry_mode: u32,
    height_scale: f32,
    velocity_scale: f32,
}

#[derive(Resource)]
pub(super) struct WaterHeightfieldPipeline {
    layout: BindGroupLayoutDescriptor,
    pipeline: CachedComputePipelineId,
}

impl FromWorld for WaterHeightfieldPipeline {
    fn from_world(world: &mut World) -> Self {
        let pipeline_cache = world.resource::<PipelineCache>();
        let asset_server = world.resource::<AssetServer>();
        let unfiltered = TextureSampleType::Float { filterable: false };
        let entries = BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                uniform_buffer::<WaterHeightfieldParams>(false),
                uniform_buffer::<WaterLodSlices>(false),
                texture_storage_2d_array(WATER_LOD_SURFACE_FORMAT, StorageTextureAccess::WriteOnly),
                texture_storage_2d_array(WATER_LOD_SURFACE_FORMAT, StorageTextureAccess::WriteOnly),
                texture_storage_2d_array(
                    TextureFormat::Rgba32Float,
                    StorageTextureAccess::WriteOnly,
                ),
                texture_2d_array(unfiltered),
                texture_2d_array(unfiltered),
                texture_2d_array(unfiltered),
                texture_2d(unfiltered),
                texture_2d(unfiltered),
                texture_2d(unfiltered),
                texture_2d(unfiltered),
                texture_2d(unfiltered),
                texture_2d(unfiltered),
                texture_2d(unfiltered),
                texture_2d(unfiltered),
                texture_2d(unfiltered),
                texture_2d(unfiltered),
                texture_storage_2d_array(WATER_LOD_DRIFT_FORMAT, StorageTextureAccess::WriteOnly),
                texture_2d_array(unfiltered),
            ),
        );
        let layout = BindGroupLayoutDescriptor::new("water_heightfield_layout", &entries);
        let pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("water_heightfield".into()),
            layout: vec![layout.clone()],
            immediate_size: 0,
            shader: asset_server.load(WATER_HEIGHTFIELD_SHADER),
            shader_defs: vec![],
            entry_point: Some("water_heightfield".into()),
            zero_initialize_workgroup_memory: false,
            constants: vec![],
        });
        Self { layout, pipeline }
    }
}

struct HeightfieldPass {
    group: BindGroup,
    dispatch: UVec3,
}

/// One blend pass per [`WaterHeightfield`] whose textures are on the GPU this frame.
#[derive(Resource, Default)]
pub(super) struct WaterHeightfieldPasses {
    passes: Vec<HeightfieldPass>,
    params: Vec<UniformBuffer<WaterHeightfieldParams>>,
}

#[allow(clippy::too_many_arguments)]
pub(super) fn prepare_water_heightfields(
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    pipeline_cache: Res<PipelineCache>,
    pipeline: Res<WaterHeightfieldPipeline>,
    heightfields: Query<&WaterHeightfield>,
    lod: Option<Res<WaterLodData>>,
    lod_gpu: Res<WaterLodGpu>,
    scratch: Res<WaterLodScratch>,
    images: Res<RenderAssets<GpuImage>>,
    ocean: Option<Res<OceanLodWriteFrame>>,
    mut state: ResMut<WaterHeightfieldPasses>,
) {
    let state = &mut *state;
    state.passes.clear();
    let Some(lod) = lod else {
        return;
    };
    let (Some(disp_out), Some(foam_out), Some(flow_out), Some(drift_out)) = (
        images.get(&lod.displacement),
        images.get(&lod.foam),
        images.get(&lod.flow),
        images.get(&lod.drift),
    ) else {
        return;
    };
    let (Some(disp_in), Some(foam_in), Some(flow_in), Some(drift_in), Some(slices)) = (
        scratch.displacement(),
        scratch.foam(),
        scratch.flow(),
        scratch.drift(),
        lod_gpu.slices.binding(),
    ) else {
        return;
    };
    let cascades = ocean.as_ref().and_then(|frame| {
        if frame.params.cascade_count == 0 || frame.displacement.len() < OCEAN_MAX_CASCADES {
            return None;
        }
        let views = |handles: &[Handle<Image>]| {
            handles
                .iter()
                .take(OCEAN_MAX_CASCADES)
                .map(|handle| images.get(handle).map(|gpu| &gpu.texture_view))
                .collect::<Option<Vec<_>>>()
        };
        Some((
            frame.params,
            views(&frame.displacement)?,
            views(&frame.foam)?,
        ))
    });
    let layout = pipeline_cache.get_bind_group_layout(&pipeline.layout);
    let res = lod.slices.resolution;
    let dispatch = UVec3::new(res.div_ceil(8), res.div_ceil(8), lod.slices.count);
    for heightfield in &heightfields {
        let (Some(bed_water), Some(velocity)) = (
            images.get(&heightfield.bed_water),
            images.get(&heightfield.velocity),
        ) else {
            continue;
        };
        let (tiles, amplitude, choppiness, cascade_count, disp, foam) = match &cascades {
            Some((params, disp, foam)) => (
                params.cascade_tile_sizes,
                params.amplitude,
                params.choppiness,
                params.cascade_count,
                disp.clone(),
                foam.clone(),
            ),
            None => (
                Vec4::ONE,
                0.0,
                0.0,
                0,
                vec![&bed_water.texture_view; OCEAN_MAX_CASCADES],
                vec![&bed_water.texture_view; OCEAN_MAX_CASCADES],
            ),
        };
        let (dry_mode, dry_level) = match heightfield.dry {
            WaterHeightfieldDry::Level(level) => (0, level * heightfield.height_scale),
            WaterHeightfieldDry::Bed => (1, 0.0),
        };
        let index = state.passes.len();
        if state.params.len() <= index {
            state.params.push(default());
        }
        let params = &mut state.params[index];
        params.set(WaterHeightfieldParams {
            center: heightfield.center,
            cell_size: heightfield.cell_size,
            datum: heightfield.datum,
            cells: heightfield.cells,
            blend_cells: heightfield.blend_cells,
            dry_level,
            cascade_tile_sizes: tiles,
            amplitude,
            choppiness,
            cascade_count,
            dry_mode,
            height_scale: heightfield.height_scale,
            velocity_scale: heightfield.velocity_scale,
        });
        params.write_buffer(&device, &queue);
        let Some(params) = params.binding() else {
            continue;
        };
        let group = device.create_bind_group(
            "water_heightfield_bind_group",
            &layout,
            &BindGroupEntries::sequential((
                params,
                slices.clone(),
                &disp_out.texture_view,
                &foam_out.texture_view,
                &flow_out.texture_view,
                disp_in,
                foam_in,
                flow_in,
                &bed_water.texture_view,
                &velocity.texture_view,
                disp[0],
                disp[1],
                disp[2],
                disp[3],
                foam[0],
                foam[1],
                foam[2],
                foam[3],
                &drift_out.texture_view,
                drift_in,
            )),
        );
        state.passes.push(HeightfieldPass { group, dispatch });
    }
}

/// Runs each heightfield's blend in turn. Every pass reads a fresh scratch copy, so later
/// heightfields blend over earlier ones.
pub(super) fn run_water_heightfields(
    mut ctx: RenderContext,
    pipeline: Res<WaterHeightfieldPipeline>,
    cache: Res<PipelineCache>,
    state: Res<WaterHeightfieldPasses>,
    scratch: Res<WaterLodScratch>,
    lod: Option<Res<WaterLodData>>,
    images: Res<RenderAssets<GpuImage>>,
) {
    let Some(lod) = lod else {
        return;
    };
    if state.passes.is_empty() {
        return;
    }
    let Some(compute) = cache.get_compute_pipeline(pipeline.pipeline) else {
        return;
    };
    let diagnostics = ctx.diagnostic_recorder();
    let diagnostics = diagnostics.as_deref();
    let encoder = ctx.command_encoder();
    let time_span = diagnostics.time_span(encoder, "water_heightfield");
    for pass in &state.passes {
        if !scratch.copy_from_lod(encoder, &images, &lod) {
            break;
        }
        let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("water_heightfield_pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(compute);
        compute_pass.set_bind_group(0, &pass.group, &[]);
        compute_pass.dispatch_workgroups(pass.dispatch.x, pass.dispatch.y, pass.dispatch.z);
    }
    time_span.end(encoder);
}
