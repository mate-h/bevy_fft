//! Compute pipelines, GPU buffers, and render-graph schedule for the hybrid dispersive sim.

use std::num::NonZeroU64;
use std::sync::atomic::{AtomicU32, Ordering};

use bevy::app::SubApp;
use bevy::ecs::world::FromWorld;
use bevy::render::render_resource::binding_types::{texture_storage_2d, uniform_buffer};
use bevy::render::render_resource::encase::internal::{WriteInto, Writer};
use bevy::render::render_resource::{
    BindGroup, BindGroupEntries, BindGroupLayoutDescriptor, BindGroupLayoutEntries,
    BindingResource, Buffer, BufferBinding, BufferDescriptor, BufferUsages,
    CachedComputePipelineId, ComputePassDescriptor, ComputePipelineDescriptor, PipelineCache,
    ShaderStages, ShaderType, StorageTextureAccess, TextureFormat,
};
use bevy::render::renderer::{RenderContext, RenderDevice, RenderQueue};
use bevy::{
    prelude::*,
    render::{
        Render, RenderSystems, globals::GlobalsBuffer, render_asset::RenderAssets,
        texture::GpuImage,
    },
};

use super::{DispersiveController, DispersiveGridImages, DispersiveSimRoot};
use crate::fft::resources::{FftBindGroupLayouts, FftPipelines, FftTextures};
use crate::fft::{FftSettings, run_fft_resolve_outputs, run_forward_fft, run_inverse_fft};

#[repr(C)]
#[derive(Copy, Clone, Default, ShaderType)]
pub struct DispersiveSimUniform {
    pub n: u32,
    pub diffusion_iters: u32,
    pub g: f32,
    pub dt: f32,
    pub dx: f32,
    pub tile_world: f32,
    pub gamma_surf: f32,
    pub d_grad_penalty: f32,
    pub timestamp: u32,
    pub spectral_flags: u32,
    pub h_bar_omega: f32,
    pub spectral_fixed_depth: f32,
    pub airy_depth_0: f32,
    pub airy_depth_1: f32,
    pub airy_depth_2: f32,
    pub airy_depth_3: f32,
    pub airy_stack_channel: u32,
    pub h_avgmax_beta: f32,
    pub vel_clamp_alpha: f32,
    pub brush_on: u32,
    pub brush_radius: f32,
    pub brush_strength: f32,
    pub pointer_x: f32,
    pub pointer_y: f32,
    pub pointer_ox: f32,
    pub pointer_oy: f32,
    pub _pad0: f32,
    pub _pad1: f32,
}

fn write_uniform<T: ShaderType + WriteInto>(queue: &RenderQueue, buffer: &Buffer, v: &T) {
    write_uniform_at(queue, buffer, 0, v);
}

fn write_uniform_at<T: ShaderType + WriteInto>(
    queue: &RenderQueue,
    buffer: &Buffer,
    offset: u64,
    v: &T,
) {
    let mut dest = Vec::new();
    v.write_into(&mut Writer::new(v, &mut dest, 0).unwrap());
    queue.write_buffer(buffer, offset, &dest);
}

/// Matches `SPECTRAL_FIXED_DEPTH` in `dispersive.wgsl`.
const SPECTRAL_FIXED_DEPTH_FLAG: u32 = 1;

/// wgpu uniform dynamic-offset alignment. Slot 0 is the base sim uniform; slots 1..4 are
/// the four Airy reference-depth uniforms. Mid-encoder `queue.write_buffer` to one offset
/// is unsafe: uploads flush before the whole command buffer, so only the last write wins.
const SIM_UNIFORM_SLOT_ALIGN: u64 = 256;
const SIM_UNIFORM_SLOT_COUNT: u64 = 5;

fn sim_uniform_slot_stride() -> u64 {
    let raw = <DispersiveSimUniform as ShaderType>::min_size().get();
    raw.div_ceil(SIM_UNIFORM_SLOT_ALIGN) * SIM_UNIFORM_SLOT_ALIGN
}

fn sim_uniform_binding<'a>(buffer: &'a Buffer, slot: u64) -> BindingResource<'a> {
    let size = NonZeroU64::new(<DispersiveSimUniform as ShaderType>::min_size().get())
        .expect("DispersiveSimUniform size");
    BindingResource::Buffer(BufferBinding {
        buffer,
        offset: slot * sim_uniform_slot_stride(),
        size: Some(size),
    })
}

fn sorted_airy_depths(controller: &DispersiveController) -> [f32; 4] {
    let mut sd = controller.airy_reference_depths;
    sd.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    sd
}

fn build_dispersive_sim_uniform(
    controller: &DispersiveController,
    timestamp: u32,
    spectral_flags: u32,
    spectral_fixed_depth: f32,
    airy_stack_channel: u32,
) -> DispersiveSimUniform {
    let n = controller.n;
    let dx = if controller.tile_world > 0.0 {
        controller.tile_world / n as f32
    } else {
        1.0
    };
    let sd = sorted_airy_depths(controller);
    DispersiveSimUniform {
        n,
        diffusion_iters: controller.diffusion_iters,
        g: controller.g,
        dt: controller.dt,
        dx,
        tile_world: controller.tile_world,
        gamma_surf: controller.gamma_surf,
        d_grad_penalty: 0.01,
        timestamp,
        spectral_flags,
        h_bar_omega: controller.h_bar_omega,
        spectral_fixed_depth,
        airy_depth_0: sd[0],
        airy_depth_1: sd[1],
        airy_depth_2: sd[2],
        airy_depth_3: sd[3],
        airy_stack_channel,
        h_avgmax_beta: controller.h_avgmax_beta,
        vel_clamp_alpha: controller.vel_clamp_alpha,
        brush_on: if controller.brush_active { 1 } else { 0 },
        brush_radius: controller.brush_radius,
        brush_strength: controller.brush_strength,
        pointer_x: controller.pointer.x,
        pointer_y: controller.pointer.y,
        pointer_ox: controller.pointer_prev.x,
        pointer_oy: controller.pointer_prev.y,
        _pad0: 0.0,
        _pad1: 0.0,
    }
}

#[derive(Resource)]
pub struct DispersivePipelines {
    pub data_layout: BindGroupLayoutDescriptor,
    pub decompose_init: CachedComputePipelineId,
    pub diffuse_eta_step: CachedComputePipelineId,
    pub decompose_split: CachedComputePipelineId,
    pub bar_sync_u_faces: CachedComputePipelineId,
    pub bar_sync_w_faces: CachedComputePipelineId,
    pub bar_mac_u_copy: CachedComputePipelineId,
    pub bar_mac_u_sl_forward: CachedComputePipelineId,
    pub bar_mac_u_sl_reverse: CachedComputePipelineId,
    pub bar_mac_u_combine: CachedComputePipelineId,
    pub bar_mac_w_copy: CachedComputePipelineId,
    pub bar_mac_w_sl_forward: CachedComputePipelineId,
    pub bar_mac_w_sl_reverse: CachedComputePipelineId,
    pub bar_mac_w_combine: CachedComputePipelineId,
    pub bar_cmf10_integrate_height: CachedComputePipelineId,
    pub bar_cmf10_vel_u: CachedComputePipelineId,
    pub bar_cmf10_vel_w: CachedComputePipelineId,
    pub bar_gather_cell_q: CachedComputePipelineId,
    pub pack_h_t_to_a: CachedComputePipelineId,
    pub copy_c_to_hspec: CachedComputePipelineId,
    pub unpack_b_re_to_tilde_h: CachedComputePipelineId,
    pub pack_qx_t_to_a: CachedComputePipelineId,
    pub qspec_save_c_to_backup: CachedComputePipelineId,
    pub qspec_restore_backup_to_c: CachedComputePipelineId,
    pub airy_stack_clear_qx: CachedComputePipelineId,
    pub airy_stack_clear_qy: CachedComputePipelineId,
    pub hybrid_k_qx: CachedComputePipelineId,
    pub airy_stack_write_qx_from_b: CachedComputePipelineId,
    pub blend_airy_qx_to_tilde: CachedComputePipelineId,
    pub pack_qy_t_to_a: CachedComputePipelineId,
    pub hybrid_k_qy: CachedComputePipelineId,
    pub airy_stack_write_qy_from_b: CachedComputePipelineId,
    pub blend_airy_qy_to_tilde: CachedComputePipelineId,
    pub transport_copy: CachedComputePipelineId,
    pub transport_advect: CachedComputePipelineId,
    pub merge_to_state: CachedComputePipelineId,
    pub pack_state_eta_to_a: CachedComputePipelineId,
    pub spectral_damp_c: CachedComputePipelineId,
    pub unpack_b_to_state_h: CachedComputePipelineId,
    pub pack_state_qx_to_a: CachedComputePipelineId,
    pub unpack_b_to_state_qx: CachedComputePipelineId,
    pub pack_state_qy_to_a: CachedComputePipelineId,
    pub unpack_b_to_state_qy_clamp: CachedComputePipelineId,
    pub store_bar_vel_prev: CachedComputePipelineId,
    pub init_basin: CachedComputePipelineId,
    pub brush_flow_impulse: CachedComputePipelineId,
    pub brush_tilde_wake: CachedComputePipelineId,
}

impl FromWorld for DispersivePipelines {
    fn from_world(world: &mut World) -> Self {
        let cache = world.resource::<PipelineCache>();
        let assets = world.resource::<AssetServer>();
        let layouts = world.resource::<FftBindGroupLayouts>();
        let data_entries = BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                uniform_buffer::<DispersiveSimUniform>(false),
                texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::ReadWrite),
                texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::ReadWrite),
                texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::ReadWrite),
                texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::ReadWrite),
                texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::ReadWrite),
                texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::ReadWrite),
                texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::ReadWrite),
                texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::ReadWrite),
                texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::ReadWrite),
                texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::ReadWrite),
                texture_storage_2d(TextureFormat::R32Float, StorageTextureAccess::ReadWrite),
                texture_storage_2d(TextureFormat::R32Float, StorageTextureAccess::ReadWrite),
                texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::ReadWrite),
                texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::ReadWrite),
                texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::ReadWrite),
            ),
        );
        let data_layout = BindGroupLayoutDescriptor::new("dispersive_data_layout", &data_entries);
        let layout_two = vec![layouts.common.clone(), data_layout.clone()];
        let shader = assets.load("dispersive/dispersive.wgsl");
        let def = vec![bevy::shader::ShaderDefVal::UInt("CHANNELS".into(), 4)];
        let q = |label: &'static str, entry: &'static str| {
            cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some(label.into()),
                layout: layout_two.clone(),
                immediate_size: 0,
                shader: shader.clone(),
                shader_defs: def.clone(),
                entry_point: Some(entry.into()),
                zero_initialize_workgroup_memory: false,
            })
        };
        Self {
            data_layout: data_layout.clone(),
            decompose_init: q("dis_decompose", "decompose_init"),
            diffuse_eta_step: q("dis_diffuse", "diffuse_eta_step"),
            decompose_split: q("dis_split", "decompose_split"),
            bar_sync_u_faces: q("dis_bsu", "bar_sync_u_faces"),
            bar_sync_w_faces: q("dis_bsw", "bar_sync_w_faces"),
            bar_mac_u_copy: q("dis_bmuc", "bar_mac_u_copy"),
            bar_mac_u_sl_forward: q("dis_bmuf", "bar_mac_u_sl_forward"),
            bar_mac_u_sl_reverse: q("dis_bmur", "bar_mac_u_sl_reverse"),
            bar_mac_u_combine: q("dis_bmuk", "bar_mac_u_combine"),
            bar_mac_w_copy: q("dis_bmwc", "bar_mac_w_copy"),
            bar_mac_w_sl_forward: q("dis_bmwf", "bar_mac_w_sl_forward"),
            bar_mac_w_sl_reverse: q("dis_bmwr", "bar_mac_w_sl_reverse"),
            bar_mac_w_combine: q("dis_bmwk", "bar_mac_w_combine"),
            bar_cmf10_integrate_height: q("dis_bh", "bar_cmf10_integrate_height"),
            bar_cmf10_vel_u: q("dis_bvu", "bar_cmf10_vel_u"),
            bar_cmf10_vel_w: q("dis_bvw", "bar_cmf10_vel_w"),
            bar_gather_cell_q: q("dis_bgq", "bar_gather_cell_q"),
            pack_h_t_to_a: q("dis_ph", "pack_h_t_to_a"),
            copy_c_to_hspec: q("dis_ch", "copy_c_to_hspec"),
            unpack_b_re_to_tilde_h: q("dis_uh", "unpack_b_re_to_tilde_h"),
            pack_qx_t_to_a: q("dis_pqx", "pack_qx_t_to_a"),
            qspec_save_c_to_backup: q("dis_qss", "qspec_save_c_to_backup"),
            qspec_restore_backup_to_c: q("dis_qsr", "qspec_restore_backup_to_c"),
            airy_stack_clear_qx: q("dis_asx", "airy_stack_clear_qx"),
            airy_stack_clear_qy: q("dis_asy", "airy_stack_clear_qy"),
            hybrid_k_qx: q("dis_hkx", "hybrid_k_qx"),
            airy_stack_write_qx_from_b: q("dis_awx", "airy_stack_write_qx_from_b"),
            blend_airy_qx_to_tilde: q("dis_blx", "blend_airy_qx_to_tilde"),
            pack_qy_t_to_a: q("dis_pqy", "pack_qy_t_to_a"),
            hybrid_k_qy: q("dis_hky", "hybrid_k_qy"),
            airy_stack_write_qy_from_b: q("dis_awy", "airy_stack_write_qy_from_b"),
            blend_airy_qy_to_tilde: q("dis_bly", "blend_airy_qy_to_tilde"),
            transport_copy: q("dis_tc", "transport_copy_tilde_to_scratch"),
            transport_advect: q("dis_ta", "transport_advect_tilde"),
            merge_to_state: q("dis_merge", "merge_to_state"),
            pack_state_eta_to_a: q("dis_pse", "pack_state_eta_to_a"),
            spectral_damp_c: q("dis_sdc", "spectral_damp_c"),
            unpack_b_to_state_h: q("dis_ush", "unpack_b_to_state_h"),
            pack_state_qx_to_a: q("dis_psx", "pack_state_qx_to_a"),
            unpack_b_to_state_qx: q("dis_usx", "unpack_b_to_state_qx"),
            pack_state_qy_to_a: q("dis_psy", "pack_state_qy_to_a"),
            unpack_b_to_state_qy_clamp: q("dis_usy", "unpack_b_to_state_qy_clamp"),
            store_bar_vel_prev: q("dis_svp", "store_bar_vel_prev"),
            init_basin: q("dis_init", "init_basin"),
            brush_flow_impulse: q("dis_brush", "brush_flow_impulse"),
            brush_tilde_wake: q("dis_brush_t", "brush_tilde_wake"),
        }
    }
}

#[derive(Resource, Default)]
pub struct DispersiveTimestamp(pub AtomicU32);

#[derive(Resource)]
pub struct DispersiveGpuResources {
    pub fft_settings_buffer: Option<Buffer>,
    pub dispersive_sim_buffer: Option<Buffer>,
    pub fft_bind_group: Option<BindGroup>,
    pub dispersive_bind_group: Option<BindGroup>,
    /// Bind groups that point at Airy depth uniform slots 1..4 (same textures as base).
    pub dispersive_bind_group_airy: [Option<BindGroup>; 4],
    pub last_n: u32,
    /// Last applied [`DispersiveController::init_serial`] after a successful `init_basin` dispatch.
    pub last_init_serial: AtomicU32,
    /// Last applied [`DispersiveController::step_serial`] after a successful paused step.
    pub last_step_serial: AtomicU32,
    pub last_apply_serial: u32,
}

impl Default for DispersiveGpuResources {
    fn default() -> Self {
        Self {
            fft_settings_buffer: None,
            dispersive_sim_buffer: None,
            fft_bind_group: None,
            dispersive_bind_group: None,
            dispersive_bind_group_airy: [None, None, None, None],
            last_n: 0,
            last_init_serial: AtomicU32::new(0),
            last_step_serial: AtomicU32::new(0),
            last_apply_serial: 0,
        }
    }
}

impl DispersiveGpuResources {
    fn clear_bind_groups(&mut self) {
        self.fft_bind_group = None;
        self.dispersive_bind_group = None;
        self.dispersive_bind_group_airy = [None, None, None, None];
    }

    fn ensure_uniform_buffers(
        &mut self,
        device: &RenderDevice,
        n: u32,
        n_changed: bool,
        force: bool,
    ) {
        if !force
            && self.fft_settings_buffer.is_some()
            && self.dispersive_sim_buffer.is_some()
            && self.last_n == n
            && !n_changed
        {
            return;
        }
        self.clear_bind_groups();
        self.last_n = n;
        self.fft_settings_buffer = Some(device.create_buffer(&BufferDescriptor {
            label: Some("dispersive_fft_settings"),
            size: FftSettings::min_size().get(),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        self.dispersive_sim_buffer = Some(device.create_buffer(&BufferDescriptor {
            label: Some("dispersive_sim"),
            size: sim_uniform_slot_stride() * SIM_UNIFORM_SLOT_COUNT,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
    }
}

#[allow(clippy::too_many_arguments)]
pub fn prepare_dispersive_gpu(
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    mut gpu: ResMut<DispersiveGpuResources>,
    controller: Res<DispersiveController>,
    timestamp: Res<DispersiveTimestamp>,
    pipelines: Res<DispersivePipelines>,
    pipeline_cache: Res<PipelineCache>,
    layouts: Res<FftBindGroupLayouts>,
    globals: Res<GlobalsBuffer>,
    roots: Res<crate::fft::resources::FftRootsBuffer>,
    gpu_imgs: Res<RenderAssets<GpuImage>>,
    grid_query: Query<(&FftTextures, &DispersiveGridImages, &FftSettings), With<DispersiveSimRoot>>,
) {
    let n = controller.n;
    let n_changed = gpu.last_n != n;
    let apply_changed = controller.sim_apply_serial != gpu.last_apply_serial;
    if apply_changed {
        gpu.last_apply_serial = controller.sim_apply_serial;
        gpu.clear_bind_groups();
    }
    let need = n_changed
        | apply_changed
        | gpu.fft_settings_buffer.is_none()
        | gpu.dispersive_sim_buffer.is_none();
    gpu.ensure_uniform_buffers(&render_device, n, n_changed, need);

    let Some(fs) = gpu.fft_settings_buffer.as_ref() else {
        return;
    };
    let Some(dsb) = gpu.dispersive_sim_buffer.as_ref() else {
        return;
    };

    let Ok((textures, dimg, fs_comp)) = grid_query.single() else {
        return;
    };

    write_uniform(&render_queue, fs, fs_comp);
    let ts_now = timestamp.0.load(Ordering::Relaxed);
    let base_u = build_dispersive_sim_uniform(&controller, ts_now, 0, 0.0, 0);
    write_uniform_at(&render_queue, dsb, 0, &base_u);
    let sd = sorted_airy_depths(&controller);
    for depth_idx in 0u32..4u32 {
        let mut u = base_u;
        u.spectral_flags = SPECTRAL_FIXED_DEPTH_FLAG;
        u.spectral_fixed_depth = sd[depth_idx as usize];
        u.airy_stack_channel = depth_idx;
        let offset = (depth_idx as u64 + 1) * sim_uniform_slot_stride();
        write_uniform_at(&render_queue, dsb, offset, &u);
    }

    let Some(ga) = globals.buffer.binding() else {
        return;
    };
    let Some(rbind) = roots.buffer.binding() else {
        return;
    };
    let Some(a_re) = gpu_imgs.get(&textures.buffer_a_re) else {
        gpu.clear_bind_groups();
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
    let Some(st) = gpu_imgs.get(&dimg.state) else {
        return;
    };
    let Some(br) = gpu_imgs.get(&dimg.bar) else {
        return;
    };
    let Some(td) = gpu_imgs.get(&dimg.tilde) else {
        return;
    };
    let Some(bed) = gpu_imgs.get(&dimg.bed) else {
        return;
    };
    let Some(sc) = gpu_imgs.get(&dimg.scratch) else {
        return;
    };
    let Some(hsr) = gpu_imgs.get(&dimg.h_spec_re) else {
        return;
    };
    let Some(hsi) = gpu_imgs.get(&dimg.h_spec_im) else {
        return;
    };
    let Some(qbk) = gpu_imgs.get(&dimg.q_spec_backup) else {
        return;
    };
    let Some(asx) = gpu_imgs.get(&dimg.airy_stack_qx) else {
        return;
    };
    let Some(asy) = gpu_imgs.get(&dimg.airy_stack_qy) else {
        return;
    };
    let Some(bfu) = gpu_imgs.get(&dimg.bar_face_u) else {
        return;
    };
    let Some(bfw) = gpu_imgs.get(&dimg.bar_face_w) else {
        return;
    };
    let Some(bmu) = gpu_imgs.get(&dimg.bar_mac_u) else {
        return;
    };
    let Some(bmw) = gpu_imgs.get(&dimg.bar_mac_w) else {
        return;
    };
    let Some(bvp) = gpu_imgs.get(&dimg.bar_vel_prev) else {
        return;
    };

    let common_layout = pipeline_cache.get_bind_group_layout(&layouts.common);
    let dis_layout = pipeline_cache.get_bind_group_layout(&pipelines.data_layout);
    let fft_bg = render_device.create_bind_group(
        "dispersive_fft",
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
    let texture_views = (
        &st.texture_view,
        &br.texture_view,
        &td.texture_view,
        &bed.texture_view,
        &sc.texture_view,
        &hsr.texture_view,
        &hsi.texture_view,
        &qbk.texture_view,
        &asx.texture_view,
        &asy.texture_view,
        &bfu.texture_view,
        &bfw.texture_view,
        &bmu.texture_view,
        &bmw.texture_view,
        &bvp.texture_view,
    );
    let make_data_bg = |label: &str, slot: u64| {
        render_device.create_bind_group(
            label,
            &dis_layout,
            &BindGroupEntries::sequential((
                sim_uniform_binding(dsb, slot),
                texture_views.0,
                texture_views.1,
                texture_views.2,
                texture_views.3,
                texture_views.4,
                texture_views.5,
                texture_views.6,
                texture_views.7,
                texture_views.8,
                texture_views.9,
                texture_views.10,
                texture_views.11,
                texture_views.12,
                texture_views.13,
                texture_views.14,
            )),
        )
    };
    let dbg = make_data_bg("dispersive_data", 0);
    let airy_bgs = [
        make_data_bg("dispersive_data_airy0", 1),
        make_data_bg("dispersive_data_airy1", 2),
        make_data_bg("dispersive_data_airy2", 3),
        make_data_bg("dispersive_data_airy3", 4),
    ];
    gpu.fft_bind_group = Some(fft_bg);
    gpu.dispersive_bind_group = Some(dbg);
    gpu.dispersive_bind_group_airy = airy_bgs.map(Some);
}

const WG: u32 = 16;

fn wg2(n: u32) -> (u32, u32) {
    (n.div_ceil(WG), n.div_ceil(WG))
}

fn wg_uface(n: u32) -> (u32, u32) {
    ((n + 1).div_ceil(WG), n.div_ceil(WG))
}

fn wg_wface(n: u32) -> (u32, u32) {
    (n.div_ceil(WG), (n + 1).div_ceil(WG))
}

fn dispatch(
    pass: &mut bevy::render::render_resource::ComputePass,
    cache: &PipelineCache,
    id: CachedComputePipelineId,
    fft: &BindGroup,
    d: &BindGroup,
    wg: (u32, u32),
) {
    let Some(p) = cache.get_compute_pipeline(id) else {
        return;
    };
    pass.set_pipeline(p);
    pass.set_bind_group(0, fft, &[]);
    pass.set_bind_group(1, d, &[]);
    pass.dispatch_workgroups(wg.0, wg.1, 1);
}

#[derive(PartialEq, Eq, Debug, Copy, Clone, Hash)]
pub struct DispersiveSimLabel;

#[allow(clippy::too_many_arguments)]
pub fn run_dispersive_sim(
    mut ctx: RenderContext,
    pl: Res<DispersivePipelines>,
    pl_fft: Res<FftPipelines>,
    cache: Res<PipelineCache>,
    controller: Res<DispersiveController>,
    gpu: Res<DispersiveGpuResources>,
    ts: Res<DispersiveTimestamp>,
    fft_settings: Query<&FftSettings, With<DispersiveSimRoot>>,
) {
    let Some(settings) = fft_settings.iter().next() else {
        return;
    };
    let Some(fft) = gpu.fft_bind_group.as_ref() else {
        return;
    };
    let Some(dbg) = gpu.dispersive_bind_group.as_ref() else {
        return;
    };
    let Some(airy0) = gpu.dispersive_bind_group_airy[0].as_ref() else {
        return;
    };
    let Some(airy1) = gpu.dispersive_bind_group_airy[1].as_ref() else {
        return;
    };
    let Some(airy2) = gpu.dispersive_bind_group_airy[2].as_ref() else {
        return;
    };
    let Some(airy3) = gpu.dispersive_bind_group_airy[3].as_ref() else {
        return;
    };
    let airy_dbg = [airy0, airy1, airy2, airy3];

    let n = controller.n;
    let wgn = wg2(n);
    let wu = wg_uface(n);
    let ww = wg_wface(n);

    let step_now =
        controller.paused && controller.step_serial != gpu.last_step_serial.load(Ordering::Relaxed);
    let run_hybrid = !controller.paused || step_now;

    let finish_step = || {
        if step_now {
            gpu.last_step_serial
                .store(controller.step_serial, Ordering::Relaxed);
        }
    };

    let enc = ctx.command_encoder();
    {
        let mut pass = enc.begin_compute_pass(&ComputePassDescriptor {
            label: Some("dispersive_sim"),
            ..default()
        });
        if controller.init_serial != gpu.last_init_serial.load(Ordering::Relaxed) {
            // Only advance the serial after a real dispatch. If the pipeline is still compiling,
            // leave the serial unmatched so init retries next frame (otherwise the field stays zero).
            if cache.get_compute_pipeline(pl.init_basin).is_some() {
                dispatch(&mut pass, &cache, pl.init_basin, fft, dbg, wgn);
                gpu.last_init_serial
                    .store(controller.init_serial, Ordering::Relaxed);
                once!(info!("dispersive: ran init_basin"));
            } else {
                once!(warn!(
                    "dispersive: init_basin pipeline not ready; will retry"
                ));
            }
        }

        // Brush before the hybrid step so the impulse is decomposed and evolved (paper Sec. 4.6).
        dispatch(&mut pass, &cache, pl.brush_flow_impulse, fft, dbg, wgn);

        if !run_hybrid {
            drop(pass);
            finish_step();
            return;
        }

        dispatch(&mut pass, &cache, pl.decompose_init, fft, dbg, wgn);
        for _ in 0..controller.diffusion_iters {
            dispatch(&mut pass, &cache, pl.diffuse_eta_step, fft, dbg, wgn);
        }
        dispatch(&mut pass, &cache, pl.decompose_split, fft, dbg, wgn);
        dispatch(&mut pass, &cache, pl.brush_tilde_wake, fft, dbg, wgn);

        dispatch(&mut pass, &cache, pl.bar_sync_u_faces, fft, dbg, wu);
        dispatch(&mut pass, &cache, pl.bar_sync_w_faces, fft, dbg, ww);
        dispatch(&mut pass, &cache, pl.bar_mac_u_copy, fft, dbg, wu);
        dispatch(&mut pass, &cache, pl.bar_mac_u_sl_forward, fft, dbg, wu);
        dispatch(&mut pass, &cache, pl.bar_mac_u_sl_reverse, fft, dbg, wu);
        dispatch(&mut pass, &cache, pl.bar_mac_u_combine, fft, dbg, wu);
        dispatch(&mut pass, &cache, pl.bar_mac_w_copy, fft, dbg, ww);
        dispatch(&mut pass, &cache, pl.bar_mac_w_sl_forward, fft, dbg, ww);
        dispatch(&mut pass, &cache, pl.bar_mac_w_sl_reverse, fft, dbg, ww);
        dispatch(&mut pass, &cache, pl.bar_mac_w_combine, fft, dbg, ww);
        dispatch(
            &mut pass,
            &cache,
            pl.bar_cmf10_integrate_height,
            fft,
            dbg,
            wgn,
        );
        dispatch(&mut pass, &cache, pl.bar_cmf10_vel_u, fft, dbg, wu);
        dispatch(&mut pass, &cache, pl.bar_cmf10_vel_w, fft, dbg, ww);
        dispatch(&mut pass, &cache, pl.bar_gather_cell_q, fft, dbg, wgn);

        dispatch(&mut pass, &cache, pl.pack_h_t_to_a, fft, dbg, wgn);
        run_forward_fft(&pl_fft, &cache, &mut pass, fft, settings);
        dispatch(&mut pass, &cache, pl.copy_c_to_hspec, fft, dbg, wgn);
        // Band-limit ˜h in place so merge/transport do not keep feeding spatial lattices.
        run_inverse_fft(&pl_fft, &cache, &mut pass, fft, settings);
        dispatch(&mut pass, &cache, pl.unpack_b_re_to_tilde_h, fft, dbg, wgn);
        dispatch(&mut pass, &cache, pl.pack_qx_t_to_a, fft, dbg, wgn);
        run_forward_fft(&pl_fft, &cache, &mut pass, fft, settings);
        dispatch(&mut pass, &cache, pl.qspec_save_c_to_backup, fft, dbg, wgn);
        dispatch(&mut pass, &cache, pl.airy_stack_clear_qx, fft, dbg, wgn);
    }

    for depth_idx in 0usize..4 {
        let mut pass = enc.begin_compute_pass(&ComputePassDescriptor {
            label: Some("dispersive_airy_qx_depth"),
            ..default()
        });
        let dbg_d = airy_dbg[depth_idx];
        dispatch(
            &mut pass,
            &cache,
            pl.qspec_restore_backup_to_c,
            fft,
            dbg_d,
            wgn,
        );
        dispatch(&mut pass, &cache, pl.hybrid_k_qx, fft, dbg_d, wgn);
        run_inverse_fft(&pl_fft, &cache, &mut pass, fft, settings);
        dispatch(
            &mut pass,
            &cache,
            pl.airy_stack_write_qx_from_b,
            fft,
            dbg_d,
            wgn,
        );
    }

    {
        let mut pass = enc.begin_compute_pass(&ComputePassDescriptor {
            label: Some("dispersive_airy_qy_prep"),
            ..default()
        });
        dispatch(&mut pass, &cache, pl.blend_airy_qx_to_tilde, fft, dbg, wgn);
        dispatch(&mut pass, &cache, pl.pack_qy_t_to_a, fft, dbg, wgn);
        run_forward_fft(&pl_fft, &cache, &mut pass, fft, settings);
        dispatch(&mut pass, &cache, pl.qspec_save_c_to_backup, fft, dbg, wgn);
        dispatch(&mut pass, &cache, pl.airy_stack_clear_qy, fft, dbg, wgn);
    }

    for depth_idx in 0usize..4 {
        let mut pass = enc.begin_compute_pass(&ComputePassDescriptor {
            label: Some("dispersive_airy_qy_depth"),
            ..default()
        });
        let dbg_d = airy_dbg[depth_idx];
        dispatch(
            &mut pass,
            &cache,
            pl.qspec_restore_backup_to_c,
            fft,
            dbg_d,
            wgn,
        );
        dispatch(&mut pass, &cache, pl.hybrid_k_qy, fft, dbg_d, wgn);
        run_inverse_fft(&pl_fft, &cache, &mut pass, fft, settings);
        dispatch(
            &mut pass,
            &cache,
            pl.airy_stack_write_qy_from_b,
            fft,
            dbg_d,
            wgn,
        );
    }

    {
        let mut pass = enc.begin_compute_pass(&ComputePassDescriptor {
            label: Some("dispersive_transport_merge"),
            ..default()
        });
        dispatch(&mut pass, &cache, pl.blend_airy_qy_to_tilde, fft, dbg, wgn);
        dispatch(&mut pass, &cache, pl.transport_copy, fft, dbg, wgn);
        dispatch(&mut pass, &cache, pl.transport_advect, fft, dbg, wgn);
        dispatch(&mut pass, &cache, pl.merge_to_state, fft, dbg, wgn);
        // Post-merge soft high-k damp on η and q (NumPy damp_state_fields).
        dispatch(&mut pass, &cache, pl.pack_state_eta_to_a, fft, dbg, wgn);
        run_forward_fft(&pl_fft, &cache, &mut pass, fft, settings);
        dispatch(&mut pass, &cache, pl.spectral_damp_c, fft, dbg, wgn);
        run_inverse_fft(&pl_fft, &cache, &mut pass, fft, settings);
        dispatch(&mut pass, &cache, pl.unpack_b_to_state_h, fft, dbg, wgn);
        dispatch(&mut pass, &cache, pl.pack_state_qx_to_a, fft, dbg, wgn);
        run_forward_fft(&pl_fft, &cache, &mut pass, fft, settings);
        dispatch(&mut pass, &cache, pl.spectral_damp_c, fft, dbg, wgn);
        run_inverse_fft(&pl_fft, &cache, &mut pass, fft, settings);
        dispatch(&mut pass, &cache, pl.unpack_b_to_state_qx, fft, dbg, wgn);
        dispatch(&mut pass, &cache, pl.pack_state_qy_to_a, fft, dbg, wgn);
        run_forward_fft(&pl_fft, &cache, &mut pass, fft, settings);
        dispatch(&mut pass, &cache, pl.spectral_damp_c, fft, dbg, wgn);
        run_inverse_fft(&pl_fft, &cache, &mut pass, fft, settings);
        dispatch(
            &mut pass,
            &cache,
            pl.unpack_b_to_state_qy_clamp,
            fft,
            dbg,
            wgn,
        );
        dispatch(&mut pass, &cache, pl.store_bar_vel_prev, fft, dbg, wgn);
    }
    ts.0.fetch_add(1, Ordering::Relaxed);
    finish_step();
}

pub fn plug_dispersive_render_app(render_app: &mut SubApp) {
    render_app
        .init_resource::<DispersivePipelines>()
        .init_resource::<DispersiveTimestamp>()
        .init_resource::<DispersiveGpuResources>()
        .add_systems(
            Render,
            prepare_dispersive_gpu
                .in_set(RenderSystems::PrepareBindGroups)
                .after(crate::fft::resources::prepare_fft_resolve_bind_groups),
        );
    render_app.add_systems(
        bevy::render::renderer::RenderGraph,
        run_dispersive_sim
            .after(run_fft_resolve_outputs)
            .before(bevy::core_pipeline::schedule::camera_driver)
            .in_set(bevy::render::renderer::RenderGraphSystems::Render),
    );
}

#[cfg(test)]
mod layout_tests {
    use super::DispersiveSimUniform;
    use bevy::render::render_resource::ShaderType;

    #[test]
    fn uniform_size_is_16_aligned() {
        let n = DispersiveSimUniform::min_size().get();
        assert_eq!(n % 16, 0, "uniform size {n} must be 16-byte aligned");
        // Scalar payload plus end padding for the WGSL uniform size rule.
        assert_eq!(n, 112, "expected 112-byte uniform, got {n}");
    }
}
