//! Interactive shallow water on a heightfield using the **virtual-pipe** model (WGSL in
//! `assets/shallow_water/simulator.wgsl`).
//!
//! Overview and integration notes: **`docs/shallow_water.md`**. Based on [webgpu-shallow-water](https://github.com/mate-h/webgpu-shallow-water).
//!
//! Add [`ShallowWaterPlugin`], insert [`ShallowWaterController`] as a resource with GPU images, and use
//! [`ShallowWaterSurfaceMaterial`] on an XZ plane. The compute pass runs on the root [`RenderGraph`]
//! before [`bevy::render::graph::CameraDriverLabel`].

mod render;

use bevy::{
    asset::{Asset, Assets, Handle, RenderAssetUsages, load_internal_asset},
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
    shader::{Shader, ShaderRef},
};

pub use render::{
    ShallowWaterGpuResources, ShallowWaterPipelines, ShallowWaterSimLabel, ShallowWaterSimNode,
    ShallowWaterTimestamp, plug_shallow_water_render_app, prepare_shallow_water_gpu,
    round_particle_count, splice_shallow_water_before_camera,
};

/// Border condition encoded like the reference: wall, source, drain, waves.
#[derive(Clone, Copy, Default, PartialEq, Eq, Reflect, Debug)]
pub enum ShallowWaterBorder {
    #[default]
    Wall,
    Source,
    Drain,
    Waves,
}

impl ShallowWaterBorder {
    fn bits(self) -> u32 {
        self as u32
    }
}

/// Main-world simulation parameters (extracted to the render world each frame).
#[derive(Resource, Clone, ExtractResource, Reflect)]
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
    pub pointer_sim: Vec2,
    pub pointer_prev_sim: Vec2,
    /// Terrain preset index for `loadPreset` (0–3).
    pub preset_index: u32,
    pub bed_water: Handle<Image>,
    pub velocity: Handle<Image>,
    pub flow_x: Handle<Image>,
    pub flow_y: Handle<Image>,
    /// Increment to re-run `clearBuffers` + `loadPreset` on the GPU.
    pub sim_apply_serial: u32,
}

impl ShallowWaterController {
    pub fn packed_border_mask(&self) -> u32 {
        self.left_border.bits()
            | (self.right_border.bits() << 2)
            | (self.bottom_border.bits() << 4)
            | (self.top_border.bits() << 6)
    }

    /// Builds textures and returns a controller. `cells_*` should be multiples of 32.
    pub fn new(images: &mut Assets<Image>, cells_x: u32, cells_y: u32) -> Self {
        Self {
            cells_x,
            cells_y,
            bed_water: images.add(shallow_rg32_image(cells_x, cells_y)),
            velocity: images.add(shallow_rg32_image(cells_x, cells_y)),
            flow_x: images.add(shallow_r32_image(cells_x + 1, cells_y + 1)),
            flow_y: images.add(shallow_r32_image(cells_x + 1, cells_y + 1)),
            dt: 0.2,
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
            pointer_sim: Vec2::ZERO,
            pointer_prev_sim: Vec2::ZERO,
            preset_index: 0,
            sim_apply_serial: 1,
        }
    }

    /// Recreate grid textures after a resolution change and schedule GPU re-init.
    pub fn rebuild_grid(&mut self, images: &mut Assets<Image>, cells_x: u32, cells_y: u32) {
        self.cells_x = cells_x;
        self.cells_y = cells_y;
        self.bed_water = images.add(shallow_rg32_image(cells_x, cells_y));
        self.velocity = images.add(shallow_rg32_image(cells_x, cells_y));
        self.flow_x = images.add(shallow_r32_image(cells_x + 1, cells_y + 1));
        self.flow_y = images.add(shallow_r32_image(cells_x + 1, cells_y + 1));
        self.sim_apply_serial = self.sim_apply_serial.wrapping_add(1);
    }
}

/// Rgba32Float grid for GPU storage read-write on Metal (only R and G carry sim data).
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
    image.texture_descriptor.usage =
        TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST;
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
    image.texture_descriptor.usage =
        TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST;
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
        ShaderRef::Handle(shaders::SHALLOW_WATER_SURFACE.clone())
    }

    fn fragment_shader() -> ShaderRef {
        ShaderRef::Handle(shaders::SHALLOW_WATER_SURFACE.clone())
    }

    fn prepass_vertex_shader() -> ShaderRef {
        ShaderRef::Handle(shaders::SHALLOW_WATER_SURFACE.clone())
    }

    fn deferred_vertex_shader() -> ShaderRef {
        ShaderRef::Handle(shaders::SHALLOW_WATER_SURFACE.clone())
    }

    fn deferred_fragment_shader() -> ShaderRef {
        ShaderRef::Handle(shaders::SHALLOW_WATER_SURFACE.clone())
    }
}

pub type ShallowWaterSurfaceMaterial =
    ExtendedMaterial<StandardMaterial, ShallowWaterSurfaceExtension>;

#[derive(Component)]
pub struct ShallowWaterSurfaceTag;

pub mod shaders {
    use bevy::asset::{Handle, uuid_handle};
    use bevy::shader::Shader;

    pub const SHALLOW_WATER_SURFACE: Handle<Shader> =
        uuid_handle!("a1b2c3d4-e5f6-7890-abcd-ef1234567890");
}

fn sync_shallow_water_mesh_material(
    sim: Res<ShallowWaterController>,
    mut materials: ResMut<Assets<ShallowWaterSurfaceMaterial>>,
    q: Query<&MeshMaterial3d<ShallowWaterSurfaceMaterial>, With<ShallowWaterSurfaceTag>>,
) {
    let Ok(h) = q.single() else {
        return;
    };
    let Some(mat) = materials.get_mut(&h.0) else {
        return;
    };
    let g = sim.cells_x.max(sim.cells_y) as f32;
    mat.extension.settings.grid_size = g;
    mat.extension.bed_water = sim.bed_water.clone();
}

pub struct ShallowWaterPlugin;

impl Plugin for ShallowWaterPlugin {
    fn build(&self, app: &mut App) {
        load_internal_asset!(
            app,
            shaders::SHALLOW_WATER_SURFACE,
            "surface.wgsl",
            Shader::from_wgsl
        );

        app.register_type::<ShallowWaterController>()
            .register_type::<ShallowWaterBorder>()
            .register_type::<ShallowWaterMaterialUniform>()
            .add_plugins(ExtractResourcePlugin::<ShallowWaterController>::default())
            .add_plugins(MaterialPlugin::<ShallowWaterSurfaceMaterial>::default())
            .add_systems(PostUpdate, sync_shallow_water_mesh_material);
    }

    fn finish(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        plug_shallow_water_render_app(render_app);
    }
}
