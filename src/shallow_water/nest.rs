//! Two-way nesting of a fine shallow water grid inside a coarse one.
//!
//! A [`ShallowWaterNest`] couples two [`ShallowWaterSolver`]s at a refinement of
//! [`NEST_REFINEMENT`] in space and time, after Debreu and Blayo (2008) and Debreu et al. (2012).
//! Each frame the parent takes one step and the child three, with a third of the parent's time
//! step on a third of its cell width. Before each child step the child's dynamic band, a strip
//! along its edge, takes the parent's surface and face velocities, blended in time across the
//! parent's step. After the child's steps, the parent cells over the child's interior take the
//! child's mean depth and face velocities, and the parent cells beside that region swap their
//! flux through the shared face for the child's (Berger and Colella 1989). The pair conserves the
//! parent's water.

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

/// Refinement between a parent grid and its child, in space and in time.
pub const NEST_REFINEMENT: u32 = 3;

const WORKGROUP: u32 = 16;

#[derive(Clone, Copy, Default, ShaderType)]
struct GpuNestSettings {
    child_cells: UVec2,
    parent_cells: UVec2,
    offset: IVec2,
    band: u32,
    time_blend: f32,
    parent_dx: f32,
    eps_wet: f32,
    _pad: Vec2,
}

/// Pipelines for the nesting kernels, queued once by
/// [`ShallowWaterSolverPlugin`](super::ShallowWaterSolverPlugin).
#[derive(Resource)]
pub struct ShallowWaterNestPipelines {
    target: BindGroupLayoutDescriptor,
    settings: BindGroupLayoutDescriptor,
    source: BindGroupLayoutDescriptor,
    snapshot: CachedComputePipelineId,
    prolong: CachedComputePipelineId,
    feedback: CachedComputePipelineId,
}

impl FromWorld for ShallowWaterNestPipelines {
    fn from_world(world: &mut World) -> Self {
        let cache = world.resource::<PipelineCache>();
        let shader = world
            .resource::<AssetServer>()
            .load(shaders::SHALLOW_WATER_NEST);
        let rgba =
            || texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::ReadWrite);
        let r = || texture_storage_2d(TextureFormat::R32Float, StorageTextureAccess::ReadWrite);
        let target = BindGroupLayoutDescriptor::new(
            "shallow_water_nest_target_layout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (rgba(), r(), r(), rgba(), r(), r()),
            ),
        );
        let settings = BindGroupLayoutDescriptor::new(
            "shallow_water_nest_settings_layout",
            &BindGroupLayoutEntries::single(
                ShaderStages::COMPUTE,
                uniform_buffer::<GpuNestSettings>(false),
            ),
        );
        let t = || texture_2d(TextureSampleType::Float { filterable: false });
        let source = BindGroupLayoutDescriptor::new(
            "shallow_water_nest_source_layout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (t(), t(), t(), t(), t(), t(), t(), t(), t(), t()),
            ),
        );
        let layout = vec![target.clone(), settings.clone(), source.clone()];
        let q = |label: &'static str, entry: &'static str| {
            cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some(label.into()),
                layout: layout.clone(),
                immediate_size: 0,
                shader: shader.clone(),
                shader_defs: vec![],
                entry_point: Some(entry.into()),
                zero_initialize_workgroup_memory: false,
                constants: vec![],
            })
        };
        Self {
            snapshot: q("shallow_water_nest_snapshot", "snapshotParent"),
            prolong: q("shallow_water_nest_prolong", "prolongBand"),
            feedback: q("shallow_water_nest_feedback", "feedback"),
            target,
            settings,
            source,
        }
    }
}

/// Where a child grid sits in its parent.
#[derive(Clone, Copy, Debug)]
pub struct ShallowWaterNestDescriptor {
    /// Parent cell under the child's cell (0, 0). The child covers
    /// `offset .. offset + child_cells / 3` in parent cells.
    pub offset: IVec2,
    /// Width of the child's dynamic band in parent cells, three times that in child cells. The
    /// band takes the parent's state each child step, and the child's interior inside it feeds
    /// back into the parent. 1 is enough for the solver's stencils.
    pub band: u32,
    /// Parent cell width. The child's is a third of it.
    pub parent_cell_size: f32,
    /// Child steps per parent step, each of the parent's step over this many. 3 refines time as
    /// space, after Debreu and Blayo (2008). A child whose own stability limit allows longer
    /// steps can take fewer, and one that needs shorter steps more; write its settings with the
    /// matching `dt`. At least 1.
    pub child_steps: u32,
}

/// A child [`ShallowWaterSolver`] nested in a parent one, coupled both ways. See
/// `docs/shallow_water_nesting.md`.
///
/// The child's cell counts must be multiples of [`NEST_REFINEMENT`], and it must lie inside the
/// parent. Write the child's settings with a third of the parent's `dt` and `cell_size`, and give
/// it walls on every edge: the band carries the parent's state in. Turn the overshoot limiter
/// and evaporation off on both for exact conservation.
pub struct ShallowWaterNest {
    parent_cells: UVec2,
    child_cells: UVec2,
    snapshot: ComputePipeline,
    prolong: ComputePipeline,
    feedback: ComputePipeline,
    /// Settings with `time_blend` at 1/3, 2/3 and 1.
    /// One per child step, with the parent's time at the end of that step.
    settings: Vec<(Buffer, BindGroup)>,
    child_target: BindGroup,
    child_source: BindGroup,
    parent_target: BindGroup,
    parent_source: BindGroup,
}

impl ShallowWaterNest {
    /// Builds the nest's bind groups, or returns `None` while its pipelines compile.
    pub fn new(
        device: &RenderDevice,
        pipelines: &ShallowWaterNestPipelines,
        cache: &PipelineCache,
        parent: &ShallowWaterSolver,
        child: &ShallowWaterSolver,
        descriptor: ShallowWaterNestDescriptor,
    ) -> Option<Self> {
        let get = |id| cache.get_compute_pipeline(id).cloned();
        let (snapshot, prolong, feedback) = (
            get(pipelines.snapshot)?,
            get(pipelines.prolong)?,
            get(pipelines.feedback)?,
        );
        let (parent_cells, child_cells) = (parent.cells(), child.cells());
        let covered = child_cells / NEST_REFINEMENT;
        assert_eq!(
            covered * NEST_REFINEMENT,
            child_cells,
            "child cells must be multiples of {NEST_REFINEMENT}"
        );
        assert!(
            descriptor.offset.cmpge(IVec2::ZERO).all()
                && (descriptor.offset + covered.as_ivec2())
                    .cmple(parent_cells.as_ivec2())
                    .all(),
            "the child must lie inside the parent"
        );
        assert!(
            descriptor.band >= 1 && 2 * descriptor.band < covered.min_element(),
            "the band must leave the child an interior"
        );

        let texture = |label: &'static str, size: UVec2, format: TextureFormat| {
            device
                .create_texture(&TextureDescriptor {
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
                    usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                })
                .create_view(&TextureViewDescriptor::default())
        };
        let prev_bed = texture(
            "shallow_water_nest_prev_bed",
            parent_cells,
            TextureFormat::Rgba32Float,
        );
        let prev_u = texture(
            "shallow_water_nest_prev_u",
            parent_cells + UVec2::X,
            TextureFormat::R32Float,
        );
        let prev_w = texture(
            "shallow_water_nest_prev_w",
            parent_cells + UVec2::Y,
            TextureFormat::R32Float,
        );
        // The child has no snapshot. Its slots take 1×1 placeholders.
        let none_rgba = texture(
            "shallow_water_nest_none",
            UVec2::ONE,
            TextureFormat::Rgba32Float,
        );
        let none_r = texture(
            "shallow_water_nest_none_r",
            UVec2::ONE,
            TextureFormat::R32Float,
        );

        let target_layout = cache.get_bind_group_layout(&pipelines.target);
        let source_layout = cache.get_bind_group_layout(&pipelines.source);
        let settings_layout = cache.get_bind_group_layout(&pipelines.settings);
        let (p, c) = (parent.textures(), child.textures());
        let child_target = device.create_bind_group(
            "shallow_water_nest_child_target",
            &target_layout,
            &BindGroupEntries::sequential((
                &c.bed_water,
                &c.flow_x,
                &c.flow_y,
                &none_rgba,
                &none_r,
                &none_r,
            )),
        );
        let child_source = device.create_bind_group(
            "shallow_water_nest_child_source",
            &source_layout,
            &BindGroupEntries::sequential((
                &p.bed_water,
                &p.flow_x,
                &p.flow_y,
                &prev_bed,
                &prev_u,
                &prev_w,
                &p.mac_u_temps,
                &p.mac_w_temps,
                &c.mac_u_temps,
                &c.mac_w_temps,
            )),
        );
        let parent_target = device.create_bind_group(
            "shallow_water_nest_parent_target",
            &target_layout,
            &BindGroupEntries::sequential((
                &p.bed_water,
                &p.flow_x,
                &p.flow_y,
                &prev_bed,
                &prev_u,
                &prev_w,
            )),
        );
        // The feedback never reads a snapshot of the child, so its slots repeat the child's
        // current fields.
        let parent_source = device.create_bind_group(
            "shallow_water_nest_parent_source",
            &source_layout,
            &BindGroupEntries::sequential((
                &c.bed_water,
                &c.flow_x,
                &c.flow_y,
                &c.bed_water,
                &c.flow_x,
                &c.flow_y,
                &c.mac_u_temps,
                &c.mac_w_temps,
                &p.mac_u_temps,
                &p.mac_w_temps,
            )),
        );

        let steps = descriptor.child_steps.max(1);
        let settings: Vec<_> = (1..=steps)
            .map(|m| {
                let uniform = GpuNestSettings {
                    child_cells,
                    parent_cells,
                    offset: descriptor.offset,
                    band: descriptor.band,
                    time_blend: m as f32 / steps as f32,
                    parent_dx: descriptor.parent_cell_size,
                    eps_wet: 1e-4 * descriptor.parent_cell_size,
                    _pad: Vec2::ZERO,
                };
                let mut bytes = Vec::new();
                uniform.write_into(&mut Writer::new(&uniform, &mut bytes, 0).unwrap());
                let buffer = device.create_buffer_with_data(&BufferInitDescriptor {
                    label: Some("shallow_water_nest_settings"),
                    contents: &bytes,
                    usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
                });
                let group = device.create_bind_group(
                    "shallow_water_nest_settings",
                    &settings_layout,
                    &BindGroupEntries::single(buffer.as_entire_buffer_binding()),
                );
                (buffer, group)
            })
            .collect();

        Some(Self {
            parent_cells,
            child_cells,
            snapshot,
            prolong,
            feedback,
            settings,
            child_target,
            child_source,
            parent_target,
            parent_source,
        })
    }

    /// Moves the child to `offset` in the parent. The grids' contents stay as they are.
    /// The child step count stays the one it was built with.
    pub fn set_offset(&self, queue: &RenderQueue, descriptor: ShallowWaterNestDescriptor) {
        let steps = self.settings.len();
        for (m, (buffer, _)) in self.settings.iter().enumerate() {
            let uniform = GpuNestSettings {
                child_cells: self.child_cells,
                parent_cells: self.parent_cells,
                offset: descriptor.offset,
                band: descriptor.band,
                time_blend: (m + 1) as f32 / steps as f32,
                parent_dx: descriptor.parent_cell_size,
                eps_wet: 1e-4 * descriptor.parent_cell_size,
                _pad: Vec2::ZERO,
            };
            let mut bytes = Vec::new();
            uniform.write_into(&mut Writer::new(&uniform, &mut bytes, 0).unwrap());
            queue.write_buffer(buffer, 0, &bytes);
        }
    }

    fn groups(size: UVec2) -> UVec2 {
        (size + UVec2::ONE).map(|n| n.div_ceil(WORKGROUP))
    }

    /// One parent step and [`ShallowWaterNestDescriptor::child_steps`] child steps, coupled both
    /// ways. `parent` and `child` must be
    /// the solvers the nest was built for.
    pub fn encode_frame(
        &self,
        pass: &mut ComputePass,
        parent: &ShallowWaterSolver,
        child: &ShallowWaterSolver,
    ) {
        parent.encode_clear_face_volumes(pass);
        child.encode_clear_face_volumes(pass);

        let parent_groups = Self::groups(self.parent_cells);
        pass.set_bind_group(0, &self.parent_target, &[]);
        pass.set_bind_group(1, &self.settings[self.settings.len() - 1].1, &[]);
        pass.set_bind_group(2, &self.parent_source, &[]);
        pass.set_pipeline(&self.snapshot);
        pass.dispatch_workgroups(parent_groups.x, parent_groups.y, 1);

        parent.encode_step_with_face_volumes(pass);

        let child_groups = Self::groups(self.child_cells);
        for (_, settings) in &self.settings {
            child.encode_step_with_face_volumes(pass);
            pass.set_bind_group(0, &self.child_target, &[]);
            pass.set_bind_group(1, settings, &[]);
            pass.set_bind_group(2, &self.child_source, &[]);
            pass.set_pipeline(&self.prolong);
            pass.dispatch_workgroups(child_groups.x, child_groups.y, 1);
        }

        pass.set_bind_group(0, &self.parent_target, &[]);
        pass.set_bind_group(1, &self.settings[self.settings.len() - 1].1, &[]);
        pass.set_bind_group(2, &self.parent_source, &[]);
        pass.set_pipeline(&self.feedback);
        pass.dispatch_workgroups(parent_groups.x, parent_groups.y, 1);
    }
}
