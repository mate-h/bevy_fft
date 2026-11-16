//! Water forces on voxelized floating bodies, evaluated on the GPU.
//!
//! Add [`FloatingBodyPlugin`] and spawn
//! [`FloatingBody`] on entities with a [`GlobalTransform`]. Each frame one compute pass samples the
//! shared water LOD data at every voxel of every body, so currents, wakes and surf push bodies
//! along with the waves, and sums the forces per body. Bevy's [`Readback`] returns them a few
//! frames later into [`FloatingBody::force`] and [`FloatingBody::torque`], for a physics engine to
//! apply. Only 80 bytes per body come back. [`FloatingBody::compensated`] corrects them for how
//! the body has moved since.
//!
//! `forces.wesl` documents the force model. [`FloatingBodySettings`] holds its coefficients.

use std::{collections::VecDeque, sync::Arc};

use bevy::{
    asset::RenderAssetUsages,
    prelude::*,
    render::{
        Render, RenderApp, RenderSystems,
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        gpu_readback::{Readback, ReadbackComplete},
        render_asset::RenderAssets,
        render_resource::{binding_types::*, *},
        renderer::{RenderContext, RenderDevice, RenderGraph, RenderQueue},
        storage::{GpuShaderBuffer, ShaderBuffer},
        texture::GpuImage,
    },
};

use super::VoxelizedMesh;
use crate::water::{
    WaterHeightfield, WaterLodData, WaterLodGpu, WaterLodGraphSystems, WaterLodSlices,
};

pub const FLOATING_BODY_SHADER: &str = "embedded://bevy_fft/bodies/forces.wesl";

/// Bytes of one voxel and of one `Patch` in `forces.wesl`.
const VOXEL_BYTES: usize = 16;
const PATCH_BYTES: usize = 32;
/// Bytes of one `Body` in `forces.wesl`.
const BODY_BYTES: usize = 112;
/// Bytes of the output header and of one `BodyForce` in `forces.wesl`.
const OUTPUT_HEADER_BYTES: usize = 16;
const OUTPUT_BODY_BYTES: usize = 80;
/// Most bodies one frame evaluates. Further bodies get no forces.
pub const FLOATING_BODY_MAX: usize = 256;
/// Submitted frames kept for pairing readbacks with the bodies they were computed for.
const PENDING_FRAMES: usize = 64;

/// Coefficients of the force model in `forces.wesl`.
#[derive(Resource, Clone, Copy, Debug, Reflect)]
#[reflect(Resource)]
pub struct FloatingBodySettings {
    /// m/s².
    pub gravity: f32,
    /// Water density in kg/m³: about 1025 for sea water and 1000 for fresh.
    pub density: f32,
    /// World Y of the water LOD data's zero height, the surface mesh plane.
    pub surface_y: f32,
    /// m²/s, about 1.19e-6 for sea water at 15 °C.
    pub kinematic_viscosity: f32,
    /// Pressure drag on surface moving into the water, in N/m² per m/s and per (m/s)².
    pub pressure_linear: f32,
    pub pressure_quadratic: f32,
    /// Exponent of the cosine of the angle of attack. Below 1 drag rises quickly off the
    /// tangent.
    pub pressure_falloff: f32,
    /// Suction on surface moving away from the water, as for pressure drag.
    pub suction_linear: f32,
    pub suction_quadratic: f32,
    pub suction_falloff: f32,
    /// Growth rate of wetted area times normal speed, per unit hull area, in m/s², at which
    /// slamming reaches full strength. Zero, the default, turns slamming off.
    ///
    /// Slamming stops the momentum of entering surface within one frame. Forces read back a few
    /// frames late keep acting for all of those frames, so with a physics engine applying them
    /// as-is slamming overshoots and throws bodies out of the water. Turn it on only where the
    /// forces are applied without that delay.
    pub slam_rate_max: f32,
    /// Exponent applied to the slamming strength below that rate.
    pub slam_power: f32,
    /// Share of the way the pressure head under bodies moves toward its new value each frame,
    /// from 0 to 1. The head is measured against the surface plus the previous head, so taking
    /// it whole makes it swing from frame to frame.
    pub pressure_relaxation: f32,
}

impl Default for FloatingBodySettings {
    fn default() -> Self {
        Self {
            gravity: 9.81,
            density: 1025.0,
            surface_y: 0.0,
            kinematic_viscosity: 1.19e-6,
            pressure_linear: 100.0,
            pressure_quadratic: 500.0,
            pressure_falloff: 0.5,
            suction_linear: 50.0,
            suction_quadratic: 250.0,
            suction_falloff: 0.5,
            slam_rate_max: 0.0,
            slam_power: 2.0,
            pressure_relaxation: 0.25,
        }
    }
}

/// A body that floats on the water LOD data.
///
/// The physics engine writes [`Self::linear_velocity`] and [`Self::angular_velocity`] each frame,
/// or leaves [`Self::estimate_velocity`] on to have them estimated from the [`GlobalTransform`].
/// The forces in [`Self::force`] and [`Self::torque`] are in world space, the torque about the
/// world-space centre of mass. They arrive a few frames late, so apply
/// [`Self::compensated`] instead when the body moves fast. [`Self::frame`] counts readbacks.
#[derive(Component, Clone, Debug, Reflect)]
#[require(Transform)]
pub struct FloatingBody {
    #[reflect(ignore)]
    pub hull: Arc<VoxelizedMesh>,
    /// Kilograms. Slamming uses it to scale the momentum it stops.
    pub mass: f32,
    /// Local centre of mass. [`Self::new`] uses the hull's centroid.
    pub center_of_mass: Vec3,
    /// Length along the flow in meters, for the Reynolds number of the friction line.
    pub length: f32,
    /// World velocity of the centre of mass in m/s.
    pub linear_velocity: Vec3,
    /// World angular velocity in rad/s.
    pub angular_velocity: Vec3,
    /// Estimate the velocities from how the [`GlobalTransform`] moves between frames.
    pub estimate_velocity: bool,
    /// Press the water of shallow water grids down under the body, so it pushes water aside,
    /// splashes when it drops in and leaves a wake. It takes a [`WaterHeightfield`] with a
    /// [`WaterHeightfield::pressure_head`].
    pub displaces_water: bool,
    /// Latest water force in newtons: [`Self::buoyancy`] plus [`Self::drag`].
    pub force: Vec3,
    /// Latest torque about the centre of mass in newton meters.
    pub torque: Vec3,
    /// The buoyancy part of [`Self::force`].
    pub buoyancy: Vec3,
    pub buoyancy_torque: Vec3,
    /// Friction, pressure drag and slamming: the part of [`Self::force`] that depends on velocity.
    pub drag: Vec3,
    pub drag_torque: Vec3,
    /// Water velocity averaged over the wetted surface, in m/s.
    pub water_velocity: Vec3,
    /// Hull volume below the surface in cubic meters.
    pub submerged_volume: f32,
    /// Hull surface area below the surface in square meters.
    pub wetted_area: f32,
    /// Waterplane area in square meters: how fast the submerged volume grows per meter the body
    /// sinks.
    pub waterplane_area: f32,
    /// World centre of mass when the forces were computed.
    pub sampled_center: Vec3,
    /// [`Self::linear_velocity`] and [`Self::angular_velocity`] when the forces were computed.
    pub sampled_velocity: Vec3,
    pub sampled_angular_velocity: Vec3,
    /// Readbacks so far. It changes when the forces do.
    pub frame: u32,
    #[reflect(ignore)]
    last: Option<(Vec3, Quat, f32)>,
}

impl FloatingBody {
    /// A body with this hull and mass, centred on the hull's centroid, with its length taken
    /// from the hull's longest horizontal extent.
    pub fn new(hull: impl Into<Arc<VoxelizedMesh>>, mass: f32) -> Self {
        let hull = hull.into();
        let (lo, hi) = hull
            .voxels
            .iter()
            .fold((Vec3::MAX, Vec3::MIN), |(lo, hi), v| {
                (lo.min(v.center), hi.max(v.center))
            });
        let extent = (hi - lo).max(Vec3::ZERO) + hull.voxel_size;
        Self {
            center_of_mass: hull.centroid(),
            length: extent.x.max(extent.z),
            hull,
            mass,
            linear_velocity: Vec3::ZERO,
            angular_velocity: Vec3::ZERO,
            estimate_velocity: true,
            displaces_water: true,
            force: Vec3::ZERO,
            torque: Vec3::ZERO,
            buoyancy: Vec3::ZERO,
            buoyancy_torque: Vec3::ZERO,
            drag: Vec3::ZERO,
            drag_torque: Vec3::ZERO,
            water_velocity: Vec3::ZERO,
            submerged_volume: 0.0,
            wetted_area: 0.0,
            waterplane_area: 0.0,
            sampled_center: Vec3::ZERO,
            sampled_velocity: Vec3::ZERO,
            sampled_angular_velocity: Vec3::ZERO,
            frame: 0,
            last: None,
        }
    }
}

impl FloatingBody {
    /// [`Self::force`] and [`Self::torque`] corrected for the frames they took to arrive, for a
    /// body whose centre of mass is now at `center`, moving at `linear` and `angular`, about to
    /// be stepped by `dt` seconds.
    ///
    /// Forces applied late act like a spring and a damper with a delay. Both pump energy into
    /// the motion, and at speed the drag reverses a body's motion before the next readback
    /// arrives. So:
    /// - buoyancy moves along its slope, ρ g times the waterplane area, by how far the centre of
    ///   mass has risen or sunk since [`Self::sampled_center`], limited to two voxels of travel;
    /// - drag scales with how much of the velocity relative to the water it was computed for
    ///   remains, drops to zero once that velocity has reversed, and never more than stops the
    ///   body within `dt`;
    /// - the drag torque scales the same way with the angular velocity.
    pub fn compensated(
        &self,
        center: Vec3,
        linear: Vec3,
        angular: Vec3,
        dt: f32,
        settings: &FloatingBodySettings,
    ) -> (Vec3, Vec3) {
        let limit = 2.0 * self.hull.voxel_size;
        let rise = (center.y - self.sampled_center.y).clamp(-limit, limit);
        let stiffness = settings.density * settings.gravity * self.waterplane_area;
        let buoyancy = self.buoyancy - Vec3::Y * stiffness * rise;

        let remaining = |now: Vec3, then: Vec3| {
            let then_squared = then.length_squared();
            if then_squared < 0.05 * 0.05 {
                1.0
            } else {
                (now.dot(then) / then_squared).clamp(0.0, 1.0)
            }
        };
        let relative = linear - self.water_velocity;
        let mut drag = self.drag * remaining(relative, self.sampled_velocity - self.water_velocity);
        if dt > 0.0 {
            drag = drag.clamp_length_max(self.mass * relative.length() / dt);
        }
        let drag_torque = self.drag_torque * remaining(angular, self.sampled_angular_velocity);
        (buoyancy + drag, self.buoyancy_torque + drag_torque)
    }
}

/// A grid that bodies press down and obstruct, other than a [`WaterHeightfield`], such as a
/// dispersive wake patch.
///
/// Each frame the body pass writes into the grid's two `R32Float` images, one texel per cell,
/// with row `j` along world +Z. `pressure_head` gets the depth bodies press the surface down by,
/// in units of `height_scale` meters, as [`WaterHeightfield::pressure_head`] does.
/// `obstruction` gets Gilligan's obstruction map (§3.3): 1 in open water and 0 where hulls fill
/// the cell at the waterline. Both images need storage binding.
#[derive(Component, Clone, Debug, Reflect)]
pub struct BodyFootprintTarget {
    /// World X and Z of the grid centre in meters.
    pub center: Vec2,
    pub cells: UVec2,
    /// Width of one cell in meters.
    pub cell_size: f32,
    /// Meters per unit of the grid's heights.
    pub height_scale: f32,
    pub pressure_head: Handle<Image>,
    pub obstruction: Handle<Image>,
}

impl bevy::render::sync_component::SyncComponent<RenderApp> for BodyFootprintTarget {
    type Target = Self;
}

impl bevy::render::extract_component::ExtractComponent<RenderApp> for BodyFootprintTarget {
    type QueryData = bevy::ecs::system::lifetimeless::Read<Self>;
    type QueryFilter = ();
    type Out = Self;

    fn extract_component(
        item: bevy::ecs::query::QueryItem<'_, '_, Self::QueryData>,
    ) -> Option<Self::Out> {
        Some(item.clone())
    }
}

/// Registers [`FloatingBody`] and its force pass, and adds
/// [`WaterPlugin`](crate::water::WaterPlugin) if it isn't there yet.
pub struct FloatingBodyPlugin;

impl Plugin for FloatingBodyPlugin {
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<crate::water::WaterPlugin>() {
            app.add_plugins(crate::water::WaterPlugin);
        }
        bevy::asset::embedded_asset!(app, "forces.wesl");
        app.register_type::<FloatingBody>()
            .register_type::<FloatingBodySettings>()
            .init_resource::<FloatingBodySettings>()
            .init_resource::<FloatingBodyFrame>()
            .add_plugins(ExtractResourcePlugin::<FloatingBodyFrame>::default())
            .add_plugins(bevy::render::extract_component::ExtractComponentPlugin::<
                BodyFootprintTarget,
            >::default())
            .register_type::<BodyFootprintTarget>()
            .add_systems(Startup, setup_floating_bodies)
            .add_systems(
                PostUpdate,
                submit_floating_bodies.after(bevy::transform::TransformSystems::Propagate),
            );
        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .init_resource::<FloatingBodyBindGroup>()
                .add_systems(
                    Render,
                    prepare_floating_bodies.in_set(RenderSystems::PrepareBindGroups),
                )
                .add_systems(
                    RenderGraph,
                    run_floating_bodies.in_set(WaterLodGraphSystems::Read),
                );
        }
    }

    fn finish(&self, app: &mut App) {
        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app.init_resource::<FloatingBodyPipeline>();
        }
    }
}

/// A body as the force pass sampled it: entity, centre of mass, velocity, angular velocity.
type SampledBody = (Entity, [Vec3; 3]);

/// Buffers and pairing state of the force pass.
#[derive(Resource)]
struct FloatingBodyBuffers {
    voxels: Handle<ShaderBuffer>,
    patches: Handle<ShaderBuffer>,
    bodies: Handle<ShaderBuffer>,
    state: Handle<ShaderBuffer>,
    output: Handle<ShaderBuffer>,
    /// Entities and hulls the voxel buffers were built for.
    layout: Vec<(Entity, usize)>,
    frame: u32,
    /// Per submitted frame, each body with its centre of mass, velocity and angular velocity.
    pending: VecDeque<(u32, Vec<SampledBody>)>,
}

/// Marks the entity that reads back the body forces.
#[derive(Component)]
struct FloatingBodyReadback;

fn setup_floating_bodies(mut commands: Commands, mut buffers: ResMut<Assets<ShaderBuffer>>) {
    let output = buffers.add(ShaderBuffer::with_size(
        (OUTPUT_HEADER_BYTES + FLOATING_BODY_MAX * OUTPUT_BODY_BYTES) as u64,
        RenderAssetUsages::RENDER_WORLD,
    ));
    let empty = |buffers: &mut Assets<ShaderBuffer>, bytes: usize| {
        buffers.add(ShaderBuffer::new(
            vec![0u8; bytes],
            RenderAssetUsages::default(),
        ))
    };
    let resource = FloatingBodyBuffers {
        voxels: empty(&mut buffers, VOXEL_BYTES),
        patches: empty(&mut buffers, PATCH_BYTES),
        bodies: empty(&mut buffers, FLOATING_BODY_MAX * BODY_BYTES),
        state: empty(&mut buffers, 4),
        output,
        layout: Vec::new(),
        frame: 0,
        pending: VecDeque::new(),
    };
    commands.insert_resource(resource);
    commands
        .spawn((FloatingBodyReadback, Name::new("Floating body readback")))
        .observe(accept_floating_body_forces);
}

fn accept_floating_body_forces(
    event: On<ReadbackComplete>,
    mut buffers: ResMut<FloatingBodyBuffers>,
    mut bodies: Query<&mut FloatingBody>,
) {
    let data = &event.data;
    if data.len() < OUTPUT_HEADER_BYTES {
        return;
    }
    let frame = read_u32(data, 0);
    let count = read_u32(data, 4) as usize;
    let Some(pos) = buffers.pending.iter().position(|(f, _)| *f == frame) else {
        return;
    };
    let (_, entities) = buffers.pending.remove(pos).unwrap();
    buffers.pending.retain(|(f, _)| *f > frame);
    for (i, (entity, [center, velocity, angular])) in entities.iter().enumerate().take(count) {
        let Ok(mut body) = bodies.get_mut(*entity) else {
            continue;
        };
        let o = OUTPUT_HEADER_BYTES + i * OUTPUT_BODY_BYTES;
        let Some(bytes) = data.get(o..o + OUTPUT_BODY_BYTES) else {
            break;
        };
        let f = |lane: usize| read_f32(bytes, lane * 4);
        let v = |lane: usize| Vec3::new(f(lane), f(lane + 1), f(lane + 2));
        body.buoyancy = v(0);
        body.submerged_volume = f(3);
        body.buoyancy_torque = v(4);
        body.waterplane_area = f(7);
        body.drag = v(8);
        body.wetted_area = f(11);
        body.drag_torque = v(12);
        body.water_velocity = v(16);
        body.force = body.buoyancy + body.drag;
        body.torque = body.buoyancy_torque + body.drag_torque;
        body.sampled_center = *center;
        body.sampled_velocity = *velocity;
        body.sampled_angular_velocity = *angular;
        body.frame = body.frame.wrapping_add(1);
    }
}

fn read_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
}

fn read_f32(bytes: &[u8], offset: usize) -> f32 {
    f32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
}

/// Render-world view of one frame of the force pass.
#[derive(Resource, Clone, Default, ExtractResource)]
#[extract_app(RenderApp)]
pub struct FloatingBodyFrame {
    body_count: u32,
    pressure_relaxation: f32,
    params: ForceParams,
    voxels: Handle<ShaderBuffer>,
    patches: Handle<ShaderBuffer>,
    bodies: Handle<ShaderBuffer>,
    state: Handle<ShaderBuffer>,
    output: Handle<ShaderBuffer>,
    lod_displacement: Handle<Image>,
    lod_foam: Handle<Image>,
    lod_flow: Handle<Image>,
}

/// Uniform `ForceParams` in `forces.wesl`.
#[derive(Clone, Copy, Default, ShaderType)]
struct ForceParams {
    gravity: f32,
    density: f32,
    surface_y: f32,
    dt: f32,
    kinematic_viscosity: f32,
    pressure_linear: f32,
    pressure_quadratic: f32,
    pressure_falloff: f32,
    suction_linear: f32,
    suction_quadratic: f32,
    suction_falloff: f32,
    slam_rate_max: f32,
    slam_power: f32,
    body_count: u32,
    frame: u32,
    _pad: u32,
}

fn push_vec4(bytes: &mut Vec<u8>, v: [f32; 4]) {
    for x in v {
        bytes.extend_from_slice(&x.to_le_bytes());
    }
}

#[allow(clippy::too_many_arguments)]
fn submit_floating_bodies(
    mut commands: Commands,
    mut state: ResMut<FloatingBodyBuffers>,
    mut frame_out: ResMut<FloatingBodyFrame>,
    mut shader_buffers: ResMut<Assets<ShaderBuffer>>,
    mut bodies: Query<(Entity, &mut FloatingBody, &GlobalTransform)>,
    readback: Query<(Entity, Has<Readback>), With<FloatingBodyReadback>>,
    settings: Res<FloatingBodySettings>,
    lod: Option<Res<WaterLodData>>,
    time: Res<Time>,
) {
    frame_out.body_count = 0;
    let Ok((readback_entity, has_readback)) = readback.single() else {
        return;
    };
    let now = time.elapsed_secs();
    let dt = time.delta_secs();
    let mut list = Vec::new();
    for (entity, mut body, transform) in &mut bodies {
        let (_, rotation, translation) = transform.to_scale_rotation_translation();
        let com = translation + rotation * body.center_of_mass;
        if body.estimate_velocity
            && let Some((last_com, last_rotation, last_time)) = body.last
            && now > last_time
        {
            let step = now - last_time;
            body.linear_velocity = (com - last_com) / step;
            let (axis, angle) = (rotation * last_rotation.inverse()).to_axis_angle();
            let angle = if angle > std::f32::consts::PI {
                angle - std::f32::consts::TAU
            } else {
                angle
            };
            body.angular_velocity = axis * angle / step;
        }
        body.last = Some((com, rotation, now));
        if list.len() < FLOATING_BODY_MAX && !body.hull.voxels.is_empty() {
            list.push((
                entity,
                body.into_inner().clone(),
                rotation,
                translation,
                com,
            ));
        }
    }
    let (Some(lod), false) = (lod, list.is_empty()) else {
        if has_readback {
            commands.entity(readback_entity).remove::<Readback>();
        }
        return;
    };

    // Rebuild the voxel and patch buffers when the bodies or their hulls change.
    let layout: Vec<(Entity, usize)> = list
        .iter()
        .map(|(e, b, ..)| (*e, Arc::as_ptr(&b.hull) as usize))
        .collect();
    let mut first = Vec::with_capacity(list.len());
    let (mut voxel_total, mut patch_total) = (0u32, 0u32);
    for (_, body, ..) in &list {
        first.push((voxel_total, patch_total));
        voxel_total += body.hull.voxels.len() as u32;
        patch_total += body.hull.surface.len() as u32;
    }
    if layout != state.layout {
        let mut voxels = Vec::with_capacity(voxel_total as usize * VOXEL_BYTES);
        let mut patches = Vec::with_capacity(patch_total as usize * PATCH_BYTES);
        for (_, body, ..) in &list {
            for v in &body.hull.voxels {
                push_vec4(&mut voxels, v.center.extend(v.volume).to_array());
            }
            for p in &body.hull.surface {
                push_vec4(&mut patches, p.position.extend(p.area).to_array());
                push_vec4(&mut patches, p.normal.extend(0.0).to_array());
            }
        }
        voxels.resize(voxels.len().max(VOXEL_BYTES), 0);
        patches.resize(patches.len().max(PATCH_BYTES), 0);
        state.voxels = shader_buffers.add(ShaderBuffer::new(voxels, RenderAssetUsages::default()));
        state.patches =
            shader_buffers.add(ShaderBuffer::new(patches, RenderAssetUsages::default()));
        state.state = shader_buffers.add(ShaderBuffer::new(
            vec![0u8; patch_total.max(1) as usize * 4],
            RenderAssetUsages::default(),
        ));
        state.layout = layout;
    }

    let mut bytes = Vec::with_capacity(list.len() * BODY_BYTES);
    for (i, (_, body, rotation, translation, com)) in list.iter().enumerate() {
        let area = body.hull.area();
        push_vec4(&mut bytes, rotation.to_array());
        push_vec4(
            &mut bytes,
            translation.extend(body.hull.voxel_size).to_array(),
        );
        push_vec4(&mut bytes, com.extend(body.mass).to_array());
        push_vec4(
            &mut bytes,
            body.linear_velocity.extend(body.length).to_array(),
        );
        push_vec4(&mut bytes, body.angular_velocity.extend(area).to_array());
        let ranges = [
            first[i].0,
            body.hull.voxels.len() as u32,
            first[i].1,
            body.hull.surface.len() as u32,
        ];
        let flags = [u32::from(body.displaces_water), 0, 0, 0];
        for v in ranges.into_iter().chain(flags) {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
    }
    bytes.resize(FLOATING_BODY_MAX * BODY_BYTES, 0);
    // A new asset each frame, as the ocean queries do for their points.
    state.bodies = shader_buffers.add(ShaderBuffer::new(bytes, RenderAssetUsages::default()));

    state.frame = state.frame.wrapping_add(1).max(1);
    let frame = state.frame;
    state.pending.push_back((
        frame,
        list.iter()
            .map(|(e, body, .., com)| (*e, [*com, body.linear_velocity, body.angular_velocity]))
            .collect(),
    ));
    while state.pending.len() > PENDING_FRAMES {
        state.pending.pop_front();
    }

    let s = *settings;
    *frame_out = FloatingBodyFrame {
        body_count: list.len() as u32,
        pressure_relaxation: s.pressure_relaxation,
        params: ForceParams {
            gravity: s.gravity,
            density: s.density,
            surface_y: s.surface_y,
            dt: dt.max(1.0e-4),
            kinematic_viscosity: s.kinematic_viscosity,
            pressure_linear: s.pressure_linear,
            pressure_quadratic: s.pressure_quadratic,
            pressure_falloff: s.pressure_falloff,
            suction_linear: s.suction_linear,
            suction_quadratic: s.suction_quadratic,
            suction_falloff: s.suction_falloff,
            slam_rate_max: s.slam_rate_max,
            slam_power: s.slam_power,
            body_count: list.len() as u32,
            frame,
            _pad: 0,
        },
        voxels: state.voxels.clone(),
        patches: state.patches.clone(),
        bodies: state.bodies.clone(),
        state: state.state.clone(),
        output: state.output.clone(),
        lod_displacement: lod.displacement.clone(),
        lod_foam: lod.foam.clone(),
        lod_flow: lod.flow.clone(),
    };
    if !has_readback {
        commands
            .entity(readback_entity)
            .insert(Readback::buffer(state.output.clone()));
    }
}

#[derive(Resource)]
struct FloatingBodyPipeline {
    layout: BindGroupLayoutDescriptor,
    pipeline: CachedComputePipelineId,
    /// Group 1 of the footprint kernels: one shallow water grid.
    grid_layout: BindGroupLayoutDescriptor,
    empty_layout: BindGroupLayoutDescriptor,
    footprints: CachedComputePipelineId,
    resolve: CachedComputePipelineId,
}

impl FromWorld for FloatingBodyPipeline {
    fn from_world(world: &mut World) -> Self {
        let pipeline_cache = world.resource::<PipelineCache>();
        let asset_server = world.resource::<AssetServer>();
        let unfiltered = TextureSampleType::Float { filterable: false };
        let entries = BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                uniform_buffer::<ForceParams>(false),
                storage_buffer_read_only_sized(false, None),
                storage_buffer_read_only_sized(false, None),
                storage_buffer_read_only_sized(false, None),
                storage_buffer_sized(false, None),
                storage_buffer_sized(false, None),
                uniform_buffer::<WaterLodSlices>(false),
                texture_2d_array(unfiltered),
                texture_2d_array(unfiltered),
                texture_2d_array(unfiltered),
            ),
        );
        let layout = BindGroupLayoutDescriptor::new("floating_body_layout", &entries);
        let grid_layout = BindGroupLayoutDescriptor::new(
            "floating_body_grid_layout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (
                    uniform_buffer::<GridParams>(false),
                    storage_buffer_sized(false, None),
                    texture_storage_2d(TextureFormat::R32Float, StorageTextureAccess::WriteOnly),
                    storage_buffer_sized(false, None),
                    texture_storage_2d(TextureFormat::R32Float, StorageTextureAccess::WriteOnly),
                ),
            ),
        );
        let empty_layout = BindGroupLayoutDescriptor::new("floating_body_empty_layout", &[]);
        let shader = asset_server.load(FLOATING_BODY_SHADER);
        let queue =
            |label: &'static str, entry: &'static str, layout: Vec<BindGroupLayoutDescriptor>| {
                pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                    label: Some(label.into()),
                    layout,
                    immediate_size: 0,
                    shader: shader.clone(),
                    shader_defs: vec![],
                    entry_point: Some(entry.into()),
                    zero_initialize_workgroup_memory: false,
                    constants: vec![],
                })
            };
        let pipeline = queue("floating_body_forces", "body_forces", vec![layout.clone()]);
        let footprints = queue(
            "floating_body_footprints",
            "body_footprints",
            vec![layout.clone(), grid_layout.clone()],
        );
        let resolve = queue(
            "floating_body_pressure",
            "resolve_pressure",
            vec![empty_layout.clone(), grid_layout.clone()],
        );
        Self {
            layout,
            pipeline,
            grid_layout,
            empty_layout,
            footprints,
            resolve,
        }
    }
}

/// Uniform `Grid` in `forces.wesl`.
#[derive(Clone, Copy, Default, ShaderType)]
struct GridParams {
    center: Vec2,
    cells: UVec2,
    cell_size: f32,
    height_scale: f32,
    relaxation: f32,
    has_obstruction: u32,
}

/// One shallow water grid that bodies press down, with its volume sums.
struct GridPass {
    group: BindGroup,
    displaced: Buffer,
    cells: UVec2,
}

/// Buffers of one grid: its size, the volume sums, the previous head, and its uniform.
struct GridBuffers {
    cells: UVec2,
    displaced: Buffer,
    previous: Buffer,
    params: UniformBuffer<GridParams>,
}

#[derive(Resource, Default)]
struct FloatingBodyBindGroup {
    group: Option<BindGroup>,
    params: UniformBuffer<ForceParams>,
    empty: Option<BindGroup>,
    grids: Vec<GridPass>,
    /// Volume sum buffers and grid uniforms, kept per grid entity while its size stays.
    grid_buffers: std::collections::HashMap<Entity, GridBuffers>,
    /// Bound as the obstruction map of grids without one.
    placeholder: Option<TextureView>,
}

#[allow(clippy::too_many_arguments)]
fn prepare_floating_bodies(
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    pipeline_cache: Res<PipelineCache>,
    pipeline: Option<Res<FloatingBodyPipeline>>,
    frame: Res<FloatingBodyFrame>,
    buffers: Res<RenderAssets<GpuShaderBuffer>>,
    images: Res<RenderAssets<GpuImage>>,
    lod_gpu: Res<WaterLodGpu>,
    heightfields: Query<(Entity, &WaterHeightfield)>,
    footprint_targets: Query<(Entity, &BodyFootprintTarget)>,
    mut bind_group: ResMut<FloatingBodyBindGroup>,
) {
    let state = &mut *bind_group;
    state.group = None;
    state.grids.clear();
    let Some(pipeline) = pipeline else {
        return;
    };

    // Grids with a pressure head get their footprints rebuilt every frame, even without bodies,
    // so a body that leaves takes its dent with it.
    let grid_layout = pipeline_cache.get_bind_group_layout(&pipeline.grid_layout);
    let mut targets = Vec::new();
    for (entity, heightfield) in &heightfields {
        if let Some(pressure) = &heightfield.pressure_head {
            targets.push((
                entity,
                heightfield.center,
                heightfield.cells,
                heightfield.cell_size,
                heightfield.height_scale,
                pressure.clone(),
                None,
            ));
        }
    }
    for (entity, target) in &footprint_targets {
        targets.push((
            entity,
            target.center,
            target.cells,
            target.cell_size,
            target.height_scale,
            target.pressure_head.clone(),
            Some(target.obstruction.clone()),
        ));
    }
    state
        .grid_buffers
        .retain(|e, _| targets.iter().any(|t| t.0 == *e));
    let placeholder = state.placeholder.get_or_insert_with(|| {
        device
            .create_texture(&TextureDescriptor {
                label: Some("floating_body_no_obstruction"),
                size: Extent3d::default(),
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::R32Float,
                usage: TextureUsages::STORAGE_BINDING,
                view_formats: &[],
            })
            .create_view(&TextureViewDescriptor::default())
    });
    for (entity, center, cells, cell_size, height_scale, pressure, obstruction) in targets {
        let Some(pressure) = images.get(&pressure) else {
            continue;
        };
        let obstruction = match &obstruction {
            Some(handle) => match images.get(handle) {
                Some(image) => Some(&image.texture_view),
                None => continue,
            },
            None => None,
        };
        let per_cell = |label: &'static str, lanes: u32| {
            device.create_buffer(&BufferDescriptor {
                label: Some(label),
                size: (cells.x * cells.y).max(1) as u64 * 4 * lanes as u64,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let stale = state
            .grid_buffers
            .get(&entity)
            .is_none_or(|buffers| buffers.cells != cells);
        if stale {
            state.grid_buffers.insert(
                entity,
                GridBuffers {
                    cells,
                    // Displaced volume, then waterline area, per cell.
                    displaced: per_cell("floating_body_displaced_volume", 2),
                    previous: per_cell("floating_body_previous_head", 1),
                    params: UniformBuffer::default(),
                },
            );
        }
        let buffers = state.grid_buffers.get_mut(&entity).unwrap();
        buffers.params.set(GridParams {
            center,
            cells,
            cell_size,
            height_scale,
            relaxation: frame.pressure_relaxation,
            has_obstruction: u32::from(obstruction.is_some()),
        });
        buffers.params.write_buffer(&device, &queue);
        let Some(params) = buffers.params.binding() else {
            continue;
        };
        let group = device.create_bind_group(
            "floating_body_grid_bind_group",
            &grid_layout,
            &BindGroupEntries::sequential((
                params,
                buffers.displaced.as_entire_binding(),
                &pressure.texture_view,
                buffers.previous.as_entire_binding(),
                obstruction.unwrap_or(placeholder),
            )),
        );
        state.grids.push(GridPass {
            group,
            displaced: buffers.displaced.clone(),
            cells,
        });
    }
    if state.empty.is_none() {
        let empty_layout = pipeline_cache.get_bind_group_layout(&pipeline.empty_layout);
        state.empty =
            Some(device.create_bind_group("floating_body_empty_bind_group", &empty_layout, &[]));
    }

    if frame.body_count == 0 {
        return;
    }
    let (Some(voxels), Some(patches), Some(bodies), Some(patch_state), Some(output)) = (
        buffers.get(&frame.voxels),
        buffers.get(&frame.patches),
        buffers.get(&frame.bodies),
        buffers.get(&frame.state),
        buffers.get(&frame.output),
    ) else {
        return;
    };
    let (Some(displacement), Some(foam), Some(flow), Some(slices)) = (
        images.get(&frame.lod_displacement),
        images.get(&frame.lod_foam),
        images.get(&frame.lod_flow),
        lod_gpu.slices.binding(),
    ) else {
        return;
    };
    state.params.set(frame.params);
    state.params.write_buffer(&device, &queue);
    let Some(params) = state.params.binding() else {
        return;
    };
    let layout = pipeline_cache.get_bind_group_layout(&pipeline.layout);
    state.group = Some(device.create_bind_group(
        "floating_body_bind_group",
        &layout,
        &BindGroupEntries::sequential((
            params,
            voxels.buffer.as_entire_binding(),
            patches.buffer.as_entire_binding(),
            bodies.buffer.as_entire_binding(),
            patch_state.buffer.as_entire_binding(),
            output.buffer.as_entire_binding(),
            slices,
            &displacement.texture_view,
            &foam.texture_view,
            &flow.texture_view,
        )),
    ));
}

fn run_floating_bodies(
    mut ctx: RenderContext,
    pipeline: Option<Res<FloatingBodyPipeline>>,
    cache: Res<PipelineCache>,
    frame: Res<FloatingBodyFrame>,
    bind_group: Res<FloatingBodyBindGroup>,
) {
    let Some(pipeline) = pipeline else {
        return;
    };
    let encoder = ctx.command_encoder();
    if let (Some(group), Some(compute)) = (
        &bind_group.group,
        cache.get_compute_pipeline(pipeline.pipeline),
    ) {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("floating_body_forces_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(compute);
        pass.set_bind_group(0, group, &[]);
        pass.dispatch_workgroups(frame.body_count, 1, 1);
    }

    let (Some(empty), Some(footprints), Some(resolve)) = (
        &bind_group.empty,
        cache.get_compute_pipeline(pipeline.footprints),
        cache.get_compute_pipeline(pipeline.resolve),
    ) else {
        return;
    };
    for grid in &bind_group.grids {
        encoder.clear_buffer(&grid.displaced, 0, None);
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("floating_body_footprints_pass"),
            timestamp_writes: None,
        });
        if let Some(group) = &bind_group.group {
            pass.set_pipeline(footprints);
            pass.set_bind_group(0, group, &[]);
            pass.set_bind_group(1, &grid.group, &[]);
            pass.dispatch_workgroups(frame.body_count, 1, 1);
        }
        pass.set_pipeline(resolve);
        pass.set_bind_group(0, empty, &[]);
        pass.set_bind_group(1, &grid.group, &[]);
        pass.dispatch_workgroups(grid.cells.x.div_ceil(8), grid.cells.y.div_ceil(8), 1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bodies::voxelize;

    fn body() -> FloatingBody {
        let hull = voxelize(&Mesh::from(Cuboid::new(1.0, 1.0, 1.0)), 0.25).unwrap();
        let mut body = FloatingBody::new(hull, 500.0);
        body.buoyancy = Vec3::Y * 5000.0;
        body.waterplane_area = 1.0;
        body.drag = Vec3::NEG_X * 1000.0;
        body.sampled_velocity = Vec3::X * 2.0;
        body
    }

    #[test]
    fn buoyancy_follows_the_body_between_readbacks() {
        let settings = FloatingBodySettings::default();
        let (force, _) =
            body().compensated(Vec3::Y * 0.1, Vec3::X * 2.0, Vec3::ZERO, 0.0, &settings);
        let slope = settings.density * settings.gravity;
        assert!((force.y - (5000.0 - 0.1 * slope)).abs() < 1e-2, "{force:?}");
        // Travel past two voxels is not extrapolated further.
        let (far, _) = body().compensated(Vec3::Y * 5.0, Vec3::X * 2.0, Vec3::ZERO, 0.0, &settings);
        assert!((far.y - (5000.0 - 0.5 * slope)).abs() < 1e-2, "{far:?}");
    }

    #[test]
    fn drag_fades_with_the_motion_it_was_computed_for() {
        let settings = FloatingBodySettings::default();
        let drag = |velocity: Vec3, dt: f32| {
            body()
                .compensated(Vec3::ZERO, velocity, Vec3::ZERO, dt, &settings)
                .0
                .x
        };
        assert_eq!(drag(Vec3::X * 2.0, 0.0), -1000.0);
        assert_eq!(drag(Vec3::X * 1.0, 0.0), -500.0);
        // Once the body has turned around, the old drag would push it the wrong way.
        assert_eq!(drag(Vec3::NEG_X, 0.0), 0.0);
        // It never more than stops the body: 500 kg at 0.1 m/s within 0.1 s is 500 N.
        assert!((drag(Vec3::X * 0.1, 0.1) + 50.0).abs() < 1e-3);
        let mut slow = body();
        slow.sampled_velocity = Vec3::X * 0.2;
        let capped = slow
            .compensated(Vec3::ZERO, Vec3::X * 0.2, Vec3::ZERO, 0.1, &settings)
            .0
            .x;
        assert!((capped + 1000.0).abs() < 1e-3, "{capped}");
        let capped = slow
            .compensated(Vec3::ZERO, Vec3::X * 0.1, Vec3::ZERO, 0.5, &settings)
            .0
            .x;
        assert!((capped + 100.0).abs() < 1e-3, "{capped}");
    }
}
