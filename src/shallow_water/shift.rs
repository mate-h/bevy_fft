//! Moving a shallow water grid over the world by whole cells, with the cells that come in filled
//! from world bathymetry.

use bevy::{
    ecs::world::{FromWorld, World},
    prelude::*,
    render::{
        render_resource::{
            binding_types::{texture_2d, texture_storage_2d, uniform_buffer},
            encase::internal::{WriteInto, Writer},
            *,
        },
        renderer::{RenderDevice, RenderQueue},
    },
};

use super::{ShallowWaterSolver, shaders};

const WORKGROUP: u32 = 16;

#[derive(Clone, Copy, Default, ShaderType)]
struct GpuShiftSettings {
    cells: UVec2,
    shift: IVec2,
    origin: Vec2,
    cell_size: f32,
    subsamples: u32,
    bed_origin: Vec2,
    bed_texel: f32,
    fill_level: f32,
    parent_cells: UVec2,
    parent_offset: IVec2,
    has_parent: u32,
    eps_wet: f32,
}

/// The shift kernel, queued once by [`ShallowWaterSolverPlugin`](super::ShallowWaterSolverPlugin).
#[derive(Resource)]
pub struct ShallowWaterShiftPipelines {
    target: BindGroupLayoutDescriptor,
    settings: BindGroupLayoutDescriptor,
    source: BindGroupLayoutDescriptor,
    shift: CachedComputePipelineId,
}

impl FromWorld for ShallowWaterShiftPipelines {
    fn from_world(world: &mut World) -> Self {
        let cache = world.resource::<PipelineCache>();
        let shader = world
            .resource::<AssetServer>()
            .load(shaders::SHALLOW_WATER_SHIFT);
        let target = BindGroupLayoutDescriptor::new(
            "shallow_water_shift_target_layout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (
                    texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::ReadWrite),
                    texture_storage_2d(TextureFormat::R32Float, StorageTextureAccess::ReadWrite),
                    texture_storage_2d(TextureFormat::R32Float, StorageTextureAccess::ReadWrite),
                ),
            ),
        );
        let settings = BindGroupLayoutDescriptor::new(
            "shallow_water_shift_settings_layout",
            &BindGroupLayoutEntries::single(
                ShaderStages::COMPUTE,
                uniform_buffer::<GpuShiftSettings>(false),
            ),
        );
        let t = || texture_2d(TextureSampleType::Float { filterable: false });
        let source = BindGroupLayoutDescriptor::new(
            "shallow_water_shift_source_layout",
            &BindGroupLayoutEntries::sequential(ShaderStages::COMPUTE, (t(), t(), t(), t(), t())),
        );
        let shift = cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("shallow_water_shift".into()),
            layout: vec![target.clone(), settings.clone(), source.clone()],
            immediate_size: 0,
            shader,
            shader_defs: vec![],
            entry_point: Some("shiftGrid".into()),
            zero_initialize_workgroup_memory: false,
            constants: vec![],
        });
        Self {
            target,
            settings,
            source,
            shift,
        }
    }
}

/// A world bed height texture, `R32Float`, for the cells a moving grid uncovers.
#[derive(Clone)]
pub struct ShallowWaterWorldBed {
    pub view: TextureView,
    /// World XZ of the texture's corner, texel (0, 0)'s outer corner, in meters.
    pub origin: Vec2,
    /// Width of one texel in meters.
    pub texel_size: f32,
}

/// Where a grid sits in the world after a shift, and how the cells that come in are filled.
#[derive(Clone, Copy, Debug)]
pub struct ShallowWaterShiftParams {
    /// The new cell `i` takes the old cell `i + shift`. A shift at least the grid's size refills
    /// every cell, which is how a grid starts from the world.
    pub shift: IVec2,
    /// World XZ of the grid's corner after the shift.
    pub origin: Vec2,
    pub cell_size: f32,
    /// Bed samples per cell along each axis. Give a grid 3 when it is a nest's parent, and its
    /// own parent 9, so a parent's bed is the mean of its child's.
    pub subsamples: u32,
    /// Still water level for cells that come in, when the grid has no parent.
    pub fill_level: f32,
    /// Parent cell under the grid's cell (0, 0) after the shift, when the grid is a nest's child.
    /// Cells that come in then take the parent's surface.
    pub parent_offset: Option<IVec2>,
}

/// Moves one [`ShallowWaterSolver`]'s grid by whole cells. See [`Self::encode`].
pub struct ShallowWaterShift {
    cells: UVec2,
    parent_cells: UVec2,
    pipeline: ComputePipeline,
    settings: Buffer,
    target: BindGroup,
    settings_group: BindGroup,
    source: BindGroup,
    /// Each grid texture and the scratch copy the kernel reads.
    textures: [(TextureView, Texture); 3],
}

impl ShallowWaterShift {
    /// Builds the shift for `solver`, reading the world bed from `bed` and, for a nest's child,
    /// the parent's surface from `parent`. Returns `None` while the kernel compiles.
    pub fn new(
        device: &RenderDevice,
        pipelines: &ShallowWaterShiftPipelines,
        cache: &PipelineCache,
        solver: &ShallowWaterSolver,
        bed: &ShallowWaterWorldBed,
        parent: Option<&ShallowWaterSolver>,
    ) -> Option<Self> {
        let pipeline = cache.get_compute_pipeline(pipelines.shift)?.clone();
        let cells = solver.cells();
        let scratch = |label: &'static str, size: UVec2, format: TextureFormat| {
            device.create_texture(&TextureDescriptor {
                label: Some(label),
                size: Extent3d {
                    width: size.x,
                    height: size.y,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format,
                usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
                view_formats: &[],
            })
        };
        let grid = solver.textures();
        let textures = [
            (
                grid.bed_water.clone(),
                scratch("shallow_water_shift_bed", cells, TextureFormat::Rgba32Float),
            ),
            (
                grid.flow_x.clone(),
                scratch(
                    "shallow_water_shift_u",
                    cells + UVec2::X,
                    TextureFormat::R32Float,
                ),
            ),
            (
                grid.flow_y.clone(),
                scratch(
                    "shallow_water_shift_w",
                    cells + UVec2::Y,
                    TextureFormat::R32Float,
                ),
            ),
        ];
        let view = |t: &Texture| t.create_view(&TextureViewDescriptor::default());
        let target = device.create_bind_group(
            "shallow_water_shift_target",
            &cache.get_bind_group_layout(&pipelines.target),
            &BindGroupEntries::sequential((&grid.bed_water, &grid.flow_x, &grid.flow_y)),
        );
        let settings = device.create_buffer(&BufferDescriptor {
            label: Some("shallow_water_shift_settings"),
            size: GpuShiftSettings::min_size().get(),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let settings_group = device.create_bind_group(
            "shallow_water_shift_settings",
            &cache.get_bind_group_layout(&pipelines.settings),
            &BindGroupEntries::single(settings.as_entire_buffer_binding()),
        );
        // Without a parent the kernel never reads its slot, so the old bed stands in.
        let old_bed = view(&textures[0].1);
        let parent_bed = parent.map_or(&old_bed, |p| &p.textures().bed_water);
        let source = device.create_bind_group(
            "shallow_water_shift_source",
            &cache.get_bind_group_layout(&pipelines.source),
            &BindGroupEntries::sequential((
                &old_bed,
                &view(&textures[1].1),
                &view(&textures[2].1),
                &bed.view,
                parent_bed,
            )),
        );
        Some(Self {
            cells,
            parent_cells: parent.map_or(UVec2::ONE, |p| p.cells()),
            pipeline,
            settings,
            target,
            settings_group,
            source,
            textures,
        })
    }

    /// Writes where the next [`Self::encode`] moves the grid. Write it once per frame.
    pub fn write_params(
        &self,
        queue: &RenderQueue,
        bed: &ShallowWaterWorldBed,
        params: &ShallowWaterShiftParams,
    ) {
        let uniform = GpuShiftSettings {
            cells: self.cells,
            shift: params.shift,
            origin: params.origin,
            cell_size: params.cell_size,
            subsamples: params.subsamples.max(1),
            bed_origin: bed.origin,
            bed_texel: bed.texel_size,
            fill_level: params.fill_level,
            parent_cells: self.parent_cells,
            parent_offset: params.parent_offset.unwrap_or_default(),
            has_parent: u32::from(params.parent_offset.is_some()),
            eps_wet: 1e-4 * params.cell_size,
        };
        let mut bytes = Vec::new();
        uniform.write_into(&mut Writer::new(&uniform, &mut bytes, 0).unwrap());
        queue.write_buffer(&self.settings, 0, &bytes);
    }

    /// Copies the grid aside and writes it back moved, filling the cells that come in. It
    /// encodes copies, so it takes the command encoder and runs its own compute pass. Velocities
    /// the solver derives each step, the cell-centred `velocity` and the MacCormack temporaries,
    /// are left for the next step to rebuild.
    pub fn encode(&self, encoder: &mut CommandEncoder) {
        for (from, to) in &self.textures {
            let from = from.texture();
            encoder.copy_texture_to_texture(from.as_image_copy(), to.as_image_copy(), from.size());
        }
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("shallow_water_shift"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.target, &[]);
        pass.set_bind_group(1, &self.settings_group, &[]);
        pass.set_bind_group(2, &self.source, &[]);
        let groups = (self.cells + UVec2::ONE).map(|n| n.div_ceil(WORKGROUP));
        pass.dispatch_workgroups(groups.x, groups.y, 1);
    }
}
