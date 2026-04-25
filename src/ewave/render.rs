//! GPU bind groups and the eWave render-graph node (FFT + k-space + extract).

use std::sync::atomic::{AtomicU32, Ordering};

use bevy::app::SubApp;
use bevy::ecs::query::QueryState;
use bevy::ecs::world::FromWorld;
use bevy::log::warn;
use bevy::render::graph::CameraDriverLabel;
use bevy::render::render_graph::{
    Node, NodeRunError, RenderGraph, RenderGraphContext, RenderLabel,
};
use bevy::render::render_resource::binding_types::{texture_storage_2d, uniform_buffer};
use bevy::render::render_resource::encase::internal::{WriteInto, Writer};
use bevy::render::render_resource::{
    BindGroup, BindGroupEntries, BindGroupLayoutDescriptor, BindGroupLayoutEntries, Buffer,
    BufferDescriptor, BufferUsages, CachedComputePipelineId, ComputePassDescriptor,
    ComputePipelineDescriptor, PipelineCache, ShaderStages, ShaderType, StorageTextureAccess,
    TextureFormat,
};
use bevy::render::renderer::{RenderContext, RenderDevice, RenderQueue};
use bevy::{
    prelude::*,
    render::{
        Render, RenderSystems, globals::GlobalsBuffer, render_asset::RenderAssets,
        texture::GpuImage,
    },
};

use crate::ewave::{EwaveController, EwaveGridImages, EwaveSimRoot};
use crate::fft::resources::{FftBindGroupLayouts, FftPipelines, FftTextures};
use crate::fft::{FftNode, FftSettings, run_forward_fft, run_inverse_fft};

#[repr(C)]
#[derive(Copy, Clone, Default, ShaderType)]
pub struct EwaveSimUniform {
    pub n: u32,
    pub _pad0: u32,
    pub g: f32,
    pub dt: f32,
    pub tile_world: f32,
    pub timestamp: u32,
    pub brush_on: u32,
    pub brush_radius: f32,
    pub brush_strength: f32,
    pub pointer_x: f32,
    pub pointer_y: f32,
    pub pointer_ox: f32,
    pub pointer_oy: f32,
}

fn write_uniform<T: ShaderType + WriteInto>(queue: &RenderQueue, buffer: &Buffer, v: &T) {
    let mut dest = Vec::new();
    v.write_into(&mut Writer::new(v, &mut dest, 0).unwrap());
    queue.write_buffer(buffer, 0, &dest);
}

#[derive(Resource)]
pub struct EwavePipelines {
    pub data_layout: BindGroupLayoutDescriptor,
    pub clear_spatial: CachedComputePipelineId,
    pub pack_h: CachedComputePipelineId,
    pub pack_phi: CachedComputePipelineId,
    pub copy_c_h: CachedComputePipelineId,
    pub copy_c_p: CachedComputePipelineId,
    pub ewave_k: CachedComputePipelineId,
    pub copy_h_c: CachedComputePipelineId,
    pub copy_p_c: CachedComputePipelineId,
    pub extract_b_h: CachedComputePipelineId,
    pub extract_b_phi: CachedComputePipelineId,
    pub brush: CachedComputePipelineId,
}

impl FromWorld for EwavePipelines {
    fn from_world(world: &mut World) -> Self {
        let cache = world.resource::<PipelineCache>();
        let assets = world.resource::<AssetServer>();
        let layouts = world.resource::<FftBindGroupLayouts>();
        let data_entries = BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                uniform_buffer::<EwaveSimUniform>(false),
                texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::ReadWrite),
                texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::ReadWrite),
                texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::ReadWrite),
                texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::ReadWrite),
                texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::ReadWrite),
            ),
        );
        let data_layout = BindGroupLayoutDescriptor::new("ewave_data_layout", &data_entries);
        let layout_two = vec![layouts.common.clone(), data_layout.clone()];
        let shader = assets.load("ewave/ewave.wgsl");
        let ewave_shader_defs = vec![bevy::shader::ShaderDefVal::UInt("CHANNELS".into(), 4)];
        let q = |label: &'static str, entry: &'static str| {
            cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some(label.into()),
                layout: layout_two.clone(),
                push_constant_ranges: vec![],
                shader: shader.clone(),
                shader_defs: ewave_shader_defs.clone(),
                entry_point: Some(entry.into()),
                zero_initialize_workgroup_memory: false,
            })
        };
        Self {
            data_layout: data_layout.clone(),
            clear_spatial: q("ewave_clear", "clear_spatial_to_flat"),
            pack_h: q("ewave_pack_h", "pack_h_to_a"),
            pack_phi: q("ewave_pack_phi", "pack_phi_to_a"),
            copy_c_h: q("ewave_c_h", "copy_c_to_spectrum_h"),
            copy_c_p: q("ewave_c_p", "copy_c_to_spectrum_p"),
            ewave_k: q("ewave_k", "ewave_k_step"),
            copy_h_c: q("ewave_h_c", "copy_spectrum_h_to_c"),
            copy_p_c: q("ewave_p_c", "copy_spectrum_p_to_c"),
            extract_b_h: q("ewave_ex_h", "extract_b_to_h"),
            extract_b_phi: q("ewave_ex_p", "extract_b_to_phi"),
            brush: q("ewave_brush", "brush_stroke"),
        }
    }
}

#[derive(Resource, Default)]
pub struct EwaveTimestamp(pub AtomicU32);

#[derive(Resource)]
pub struct EwaveGpuResources {
    pub fft_settings_buffer: Option<Buffer>,
    pub ewave_sim_buffer: Option<Buffer>,
    pub fft_bind_group: Option<BindGroup>,
    pub ewave_bind_group: Option<BindGroup>,
    pub last_n: u32,
    pub last_apply_serial: u32,
    /// Matches [`EwaveController::sim_apply_serial`] after a `clear_spatial` dispatch.
    pub last_cleared_apply_serial: AtomicU32,
}

impl Default for EwaveGpuResources {
    fn default() -> Self {
        Self {
            fft_settings_buffer: None,
            ewave_sim_buffer: None,
            fft_bind_group: None,
            ewave_bind_group: None,
            last_n: 0,
            last_apply_serial: 0,
            last_cleared_apply_serial: AtomicU32::new(0),
        }
    }
}

impl EwaveGpuResources {
    fn ensure_uniform_buffers(
        &mut self,
        device: &RenderDevice,
        n: u32,
        n_changed: bool,
        force: bool,
    ) {
        if !force
            && self.fft_settings_buffer.is_some()
            && self.ewave_sim_buffer.is_some()
            && self.last_n == n
            && !n_changed
        {
            return;
        }
        self.fft_bind_group = None;
        self.ewave_bind_group = None;
        self.last_n = n;
        self.fft_settings_buffer = Some(device.create_buffer(&BufferDescriptor {
            label: Some("ewave_fft_settings"),
            size: FftSettings::min_size().get() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        self.ewave_sim_buffer = Some(device.create_buffer(&BufferDescriptor {
            label: Some("ewave_sim_uniform"),
            size: <EwaveSimUniform as ShaderType>::min_size().get() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
    }
}

#[allow(clippy::too_many_arguments)]
pub fn prepare_ewave_gpu(
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    mut gpu: ResMut<EwaveGpuResources>,
    controller: Res<EwaveController>,
    timestamp: Res<EwaveTimestamp>,
    pipelines: Res<EwavePipelines>,
    pipeline_cache: Res<PipelineCache>,
    layouts: Res<FftBindGroupLayouts>,
    globals: Res<GlobalsBuffer>,
    roots: Res<crate::fft::resources::FftRootsBuffer>,
    gpu_imgs: Res<RenderAssets<GpuImage>>,
    grid_query: Query<(&FftTextures, &EwaveGridImages, &FftSettings), With<EwaveSimRoot>>,
) {
    let n = controller.n;
    let n_changed = gpu.last_n != n;
    let apply_changed = controller.sim_apply_serial != gpu.last_apply_serial;
    if apply_changed {
        gpu.last_apply_serial = controller.sim_apply_serial;
        gpu.fft_bind_group = None;
        gpu.ewave_bind_group = None;
    }
    let need_force_buffers = n_changed | apply_changed | gpu.fft_settings_buffer.is_none();
    gpu.ensure_uniform_buffers(&render_device, n, n_changed, need_force_buffers);

    let Some(fs) = gpu.fft_settings_buffer.as_ref() else {
        return;
    };
    let Some(ew) = gpu.ewave_sim_buffer.as_ref() else {
        return;
    };

    let Ok((textures, t, fs_comp)) = grid_query.single() else {
        return;
    };

    write_uniform(&render_queue, fs, fs_comp);
    let u = EwaveSimUniform {
        n: controller.n,
        _pad0: 0,
        g: controller.g,
        dt: controller.dt,
        tile_world: controller.tile_world,
        timestamp: timestamp.0.load(Ordering::Relaxed),
        brush_on: if controller.brush_active { 1 } else { 0 },
        brush_radius: controller.brush_radius,
        brush_strength: controller.brush_strength,
        pointer_x: controller.pointer.x,
        pointer_y: controller.pointer.y,
        pointer_ox: controller.pointer_prev.x,
        pointer_oy: controller.pointer_prev.y,
    };
    write_uniform(&render_queue, ew, &u);

    let Some(ga) = globals.buffer.binding() else {
        return;
    };
    let Some(rbind) = roots.buffer.binding() else {
        return;
    };
    let Some(a_re) = gpu_imgs.get(&textures.buffer_a_re) else {
        gpu.fft_bind_group = None;
        gpu.ewave_bind_group = None;
        return;
    };
    let Some(a_im) = gpu_imgs.get(&textures.buffer_a_im) else {
        return;
    };
    let Some(b_re) = gpu_imgs.get(&textures.buffer_b_re) else {
        return;
    };
    let Some(b_im) = gpu_imgs.get(&textures.buffer_b_im) else {
        return;
    };
    let Some(c_re) = gpu_imgs.get(&textures.buffer_c_re) else {
        return;
    };
    let Some(c_im) = gpu_imgs.get(&textures.buffer_c_im) else {
        return;
    };
    let Some(d_re) = gpu_imgs.get(&textures.buffer_d_re) else {
        return;
    };
    let Some(d_im) = gpu_imgs.get(&textures.buffer_d_im) else {
        return;
    };
    let Some(hp) = gpu_imgs.get(&t.h_phi) else {
        return;
    };
    let Some(hr) = gpu_imgs.get(&t.h_hat_re) else {
        return;
    };
    let Some(hi) = gpu_imgs.get(&t.h_hat_im) else {
        return;
    };
    let Some(pr) = gpu_imgs.get(&t.p_hat_re) else {
        return;
    };
    let Some(pi) = gpu_imgs.get(&t.p_hat_im) else {
        return;
    };

    let common_layout = pipeline_cache.get_bind_group_layout(&layouts.common);
    let ewave_layout = pipeline_cache.get_bind_group_layout(&pipelines.data_layout);
    let fft_bg = render_device.create_bind_group(
        "ewave_fft",
        &common_layout,
        &BindGroupEntries::sequential((
            ga,
            fs.as_entire_buffer_binding(),
            rbind,
            &a_re.texture_view,
            &a_im.texture_view,
            &b_re.texture_view,
            &b_im.texture_view,
            &c_re.texture_view,
            &c_im.texture_view,
            &d_re.texture_view,
            &d_im.texture_view,
        )),
    );
    let ed = render_device.create_bind_group(
        "ewave_data",
        &ewave_layout,
        &BindGroupEntries::sequential((
            ew.as_entire_buffer_binding(),
            &hp.texture_view,
            &hr.texture_view,
            &hi.texture_view,
            &pr.texture_view,
            &pi.texture_view,
        )),
    );
    gpu.fft_bind_group = Some(fft_bg);
    gpu.ewave_bind_group = Some(ed);
}

const WG: u32 = 16;

fn wg2(n: u32) -> (u32, u32) {
    (n.div_ceil(WG), n.div_ceil(WG))
}

fn dispatch_ewave(
    pass: &mut bevy::render::render_resource::ComputePass,
    cache: &PipelineCache,
    id: CachedComputePipelineId,
    fft: &BindGroup,
    ew: &BindGroup,
    wg: (u32, u32),
) {
    if let Some(p) = cache.get_compute_pipeline(id) {
        pass.set_pipeline(p);
        pass.set_bind_group(0, fft, &[]);
        pass.set_bind_group(1, ew, &[]);
        pass.dispatch_workgroups(wg.0, wg.1, 1);
    }
}

#[derive(PartialEq, Eq, Debug, Copy, Clone, Hash, RenderLabel)]
pub struct EwaveSimLabel;

pub struct EwaveSimNode {
    fft_settings: QueryState<&'static FftSettings, With<EwaveSimRoot>>,
}

impl FromWorld for EwaveSimNode {
    fn from_world(world: &mut World) -> Self {
        Self {
            fft_settings: QueryState::new(world),
        }
    }
}

impl Node for EwaveSimNode {
    fn update(&mut self, world: &mut World) {
        self.fft_settings.update_archetypes(world);
    }

    fn run(
        &self,
        _ctx: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), NodeRunError> {
        let pl_ew = world.resource::<EwavePipelines>();
        let pl_fft = world.resource::<FftPipelines>();
        let cache = world.resource::<PipelineCache>();
        let controller = world.resource::<EwaveController>();
        let gpu_res = world.resource::<EwaveGpuResources>();
        let ts = world.resource::<EwaveTimestamp>();

        let Some(settings) = self.fft_settings.iter_manual(world).next() else {
            return Ok(());
        };

        let Some(fft) = gpu_res.fft_bind_group.as_ref() else {
            return Ok(());
        };
        let Some(ew) = gpu_res.ewave_bind_group.as_ref() else {
            return Ok(());
        };
        let n = controller.n;
        let wgn = wg2(n);
        let apply_clear = controller.sim_apply_serial
            != gpu_res.last_cleared_apply_serial.load(Ordering::Relaxed);

        let enc = render_context.command_encoder();
        {
            let mut pass = enc.begin_compute_pass(&ComputePassDescriptor {
                label: Some("ewave_sim"),
                ..default()
            });
            if apply_clear {
                dispatch_ewave(&mut pass, &cache, pl_ew.clear_spatial, fft, ew, wgn);
                gpu_res
                    .last_cleared_apply_serial
                    .store(controller.sim_apply_serial, Ordering::Relaxed);
            }

            if !controller.paused {
                dispatch_ewave(&mut pass, &cache, pl_ew.pack_h, fft, ew, wgn);
                run_forward_fft(pl_fft, &cache, &mut pass, fft, settings);
                dispatch_ewave(&mut pass, &cache, pl_ew.copy_c_h, fft, ew, wgn);
                dispatch_ewave(&mut pass, &cache, pl_ew.pack_phi, fft, ew, wgn);
                run_forward_fft(pl_fft, &cache, &mut pass, fft, settings);
                dispatch_ewave(&mut pass, &cache, pl_ew.copy_c_p, fft, ew, wgn);
                dispatch_ewave(&mut pass, &cache, pl_ew.ewave_k, fft, ew, wgn);
                dispatch_ewave(&mut pass, &cache, pl_ew.copy_h_c, fft, ew, wgn);
                run_inverse_fft(pl_fft, &cache, &mut pass, fft, settings);
                dispatch_ewave(&mut pass, &cache, pl_ew.extract_b_h, fft, ew, wgn);
                dispatch_ewave(&mut pass, &cache, pl_ew.copy_p_c, fft, ew, wgn);
                run_inverse_fft(pl_fft, &cache, &mut pass, fft, settings);
                dispatch_ewave(&mut pass, &cache, pl_ew.extract_b_phi, fft, ew, wgn);
            }
            dispatch_ewave(&mut pass, &cache, pl_ew.brush, fft, ew, wgn);
        }
        if !controller.paused {
            ts.0.fetch_add(1, Ordering::Relaxed);
        }
        Ok(())
    }
}

/// Inserts the eWave pass between the stock FFT `ResolveOutputs` node and the camera so height
/// reads pick up the latest `h_phi` for the 3D pass.
pub fn splice_ewave_before_camera(world: &mut World) {
    let Some(mut graph) = world.get_resource_mut::<RenderGraph>() else {
        return;
    };
    if graph
        .remove_node_edge(FftNode::ResolveOutputs, CameraDriverLabel)
        .is_err()
    {
        warn!(
            "splice_ewave_before_camera: could not remove ResolveOutputs → CameraDriver edge. Register FftPlugin before EwavePlugin::finish."
        );
        return;
    }
    let user = EwaveSimLabel.intern();
    graph.add_node_edge(FftNode::ResolveOutputs, user);
    graph.add_node_edge(user, CameraDriverLabel);
}

pub fn plug_ewave_render_app(render_app: &mut SubApp) {
    render_app
        .init_resource::<EwavePipelines>()
        .init_resource::<EwaveTimestamp>()
        .init_resource::<EwaveGpuResources>()
        .add_systems(
            Render,
            prepare_ewave_gpu
                .in_set(RenderSystems::PrepareBindGroups)
                .after(crate::fft::resources::prepare_fft_resolve_bind_groups),
        );
    render_app
        .world_mut()
        .resource_scope(|world, mut graph: Mut<RenderGraph>| {
            graph.add_node(EwaveSimLabel, EwaveSimNode::from_world(world));
        });
    splice_ewave_before_camera(render_app.world_mut());
}
