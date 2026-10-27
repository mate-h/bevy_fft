//! Surf on the Lumahaʻi shoreline of Kauai, with the FFT ocean and a shallow water grid over real
//! topobathy.
//!
//! The ground is a 2 km square of 1 m LiDAR elevation and bathymetry, drawn by bevy_terrain. A
//! shallow water grid with 1 m cells covers the same square and starts from the same heights, so
//! the sea fills the measured bed and the waterline follows the island's coast, bay, and river
//! mouth. The grid writes into the shared water LOD data, so the ocean clipmap shows the surf and
//! the open sea past the map with one material, and the grid fades into the ocean waves along its
//! edges. Dry cells are left out of the water surface so the ground shows through. The buoys read
//! the same data through `OceanQueries`, one offshore past the map, one in the forcing band, and
//! one in the surf.
//!
//! Every map edge is an ocean border. The ocean's long waves are written over the grid each frame,
//! and a band of cells along each edge blends the water height toward them wherever the edge is
//! under water, so swell enters from the open sea, shoals over the reef and bar, and runs up the
//! beach. On land the band has nothing to do. A PML behind the band absorbs the backwash. The
//! panel switches the edges to walls or drains for comparison. Drag with the left mouse button to
//! add water.
//!
//! The terrain is baked from bevy_terrain's
//! `source_data/kauai_lumahai_hanalei_shoreline_lidar_1m_2048x2048.tif` into a planar tile set.
//! The tiles are not in git. From this directory, with bevy_terrain checked out next to it:
//!
//! ```sh
//! cargo run --release --manifest-path ../bevy_terrain/Cargo.toml -p bevy_terrain_preprocess -- \
//!     ../bevy_terrain/source_data/kauai_lumahai_hanalei_shoreline_lidar_1m_2048x2048.tif \
//!     --terrain-path assets/terrains/kauai_shoreline --overwrite \
//!     --data-type float32 --format r32f --planar --side-length 2048
//! ```
//!
//! The planar bake puts the GeoTIFF's columns along +X and its north along +Z, so the map is
//! mirrored east to west compared with a top-down view with north up.

use bevy::{
    anti_alias::fxaa::Fxaa,
    camera::Exposure,
    camera_controller::free_camera::{FreeCamera, FreeCameraPlugin, FreeCameraState},
    core_pipeline::{
        prepass::{DeferredPrepass, DepthPrepass},
        tonemapping::Tonemapping,
    },
    light::{
        Atmosphere, AtmosphereEnvironmentMapLight, CascadeShadowConfigBuilder, GlobalAmbientLight,
        atmosphere::ScatteringMedium, light_consts::lux,
    },
    material::OpaqueRendererMethod,
    pbr::{AtmosphereSettings, DefaultOpaqueRendererMethod, StandardMaterial},
    post_process::bloom::Bloom,
    prelude::*,
    render::render_resource::AsBindGroup,
    shader::ShaderRef,
    shape::InfinitePlane3d,
    transform::TransformSystems,
};
use bevy_egui::{
    EguiContexts, EguiPlugin, EguiPrimaryContextPass, egui,
    input::{EguiWantsInput, write_egui_wants_input_system},
};
use bevy_fft::prelude::*;
use bevy_terrain::prelude::{
    SpawnTerrainCommandsExt, TerrainMaterialPlugin, TerrainPlugin, TerrainSettings,
    TerrainViewConfig,
};

#[path = "../common/screenshot.rs"]
mod screenshot;

/// World-space tile edge in meters for each FFT cascade, largest first.
const CASCADE_TILES: [f32; 3] = [256.0, 48.0, 9.0];
/// A cascade hands waves to the next smaller one at this many wavelengths per smaller tile.
const CASCADE_HANDOFF_WAVES: f32 = 6.0;

/// bevy_terrain tile set of the Kauai shoreline, relative to the asset folder.
const TERRAIN_CONFIG: &str = "terrains/kauai_shoreline/config.tc.ron";
/// Tile folder the shallow water bed is read from. LOD 2 holds 4×4 tiles, about 1 m per texel.
const TERRAIN_BED_TILES: &str = "assets/terrains/kauai_shoreline/height/2";
/// Tiles per edge at that LOD.
const TERRAIN_BED_TILE_COUNT: usize = 4;
/// Edge length of the terrain square in meters.
const TERRAIN_SIDE: f32 = 2048.0;
/// Finest LOD in the tile set.
const TERRAIN_VIEW_LOD: u32 = 3;

/// Shallow water cell width in meters.
const CELL_SIZE: f32 = 1.0;
/// Shallow water cells per edge. The grid covers the terrain square, centred on the origin.
const GRID: u32 = (TERRAIN_SIDE / CELL_SIZE) as u32;
/// Depth in meters of the solver's zero level below sea level. It sits under the deepest bed in
/// the data, so solver heights stay positive, and the still sea is at this solver height.
const SEA_LEVEL: f32 = 30.0;
/// Mean water depth in meters off the beach, for the ocean's finite-depth dispersion.
const OFFSHORE_DEPTH: f32 = 18.0;

/// World `xz` of the demo buoys: offshore past the map, in the forcing band, and in the surf.
const BUOY_POSITIONS: [Vec2; 3] = [
    Vec2::new(-300.0, 1150.0),
    Vec2::new(-300.0, 1010.0),
    Vec2::new(-420.0, 240.0),
];

#[derive(Asset, AsBindGroup, TypePath, Clone, Default)]
struct KauaiGroundMaterial {}

impl Material for KauaiGroundMaterial {
    fn fragment_shader() -> ShaderRef {
        "examples/kauai_ground.wesl".into()
    }
}

fn main() {
    App::new()
        .insert_resource(DefaultOpaqueRendererMethod::deferred())
        .insert_resource(ClearColor(Color::BLACK))
        .insert_resource(GlobalAmbientLight::NONE)
        .insert_resource(TerrainSettings::default())
        // Twelve slices reach 8 km from the camera, past the far corner of the map from the
        // starting view.
        .insert_resource(WaterLodSettings {
            slice_count: 12,
            ..default()
        })
        .add_plugins((
            DefaultPlugins,
            FreeCameraPlugin,
            FftPlugin,
            OceanPlugin,
            ShallowWaterPlugin,
            TerrainPlugin,
            TerrainMaterialPlugin::<KauaiGroundMaterial>::default(),
            EguiPlugin::default(),
            screenshot::ScreenshotOnEnvPlugin,
        ))
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (
                spawn_surface_when_ready.after(FftSystemSet::PrepareTextures),
                float_buoys,
            ),
        )
        .add_systems(EguiPrimaryContextPass, beach_panel)
        .add_systems(
            PostUpdate,
            (
                pointer_to_sim_space
                    .after(TransformSystems::Propagate)
                    .after(write_egui_wants_input_system),
                sync_free_camera_with_egui_focus.after(write_egui_wants_input_system),
            ),
        )
        .run();
}

#[derive(Component)]
struct Buoy {
    xz: Vec2,
}

/// Wavenumber band in rad/m kept by cascade `index`. Bands touch but never overlap.
fn cascade_band(index: usize) -> (f32, f32) {
    let handoff = |i: usize| CASCADE_HANDOFF_WAVES * std::f32::consts::TAU / CASCADE_TILES[i];
    let k_min = if index == 0 { 0.0 } else { handoff(index) };
    let k_max = if index + 1 < CASCADE_TILES.len() {
        handoff(index + 1)
    } else {
        f32::INFINITY
    };
    (k_min, k_max)
}

fn setup(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut standard_materials: ResMut<Assets<StandardMaterial>>,
    mut scattering_mediums: ResMut<Assets<ScatteringMedium>>,
    asset_server: Res<AssetServer>,
) {
    let n = 256u32;
    for (index, &tile) in CASCADE_TILES.iter().enumerate() {
        let (band_k_min, band_k_max) = cascade_band(index);
        commands.spawn((
            FftSource::square_inverse_only(n),
            OceanSimSettings {
                texture_size: n,
                tile_size: tile,
                band_k_min,
                band_k_max,
                wind_speed: 9.0,
                // Waves travel toward -Z, from the open sea in the north onto the beach.
                wind_direction: -std::f32::consts::FRAC_PI_2,
                swell: 0.6,
                depth: OFFSHORE_DEPTH,
                ..default()
            },
            OceanCascade {
                index: index as u32,
            },
            OceanH0Uniform::default(),
            OceanDynamicUniform::default(),
            OceanFoamUniform::default(),
            OceanFoamPhase(0),
        ));
    }

    let mut controller = ShallowWaterController::new(&mut images, GRID, GRID);
    let placement = ShallowWaterPlacement {
        center: Vec2::ZERO,
        cell_size: CELL_SIZE,
        datum: -SEA_LEVEL,
        blend_cells: 16.0,
    };
    let bed = TerrainBed::load().expect(
        "Kauai terrain tiles are missing. Bake them with the command in this example's docs.",
    );
    let heights: Vec<f32> = (0..GRID * GRID)
        .map(|i| {
            let cell = Vec2::new((i % GRID) as f32, (i / GRID) as f32) + 0.5;
            let xz = placement.center + (cell - 0.5 * GRID as f32) * CELL_SIZE;
            bed.height(xz) + SEA_LEVEL
        })
        .collect();
    controller.bed_source = images.add(bed_source_image(GRID, GRID, &heights));
    controller.placement = Some(placement);
    controller.preset_index = SHALLOW_WATER_PRESET_BED_SOURCE;
    controller.fill_level = SEA_LEVEL;
    controller.pml_eta_rest = SEA_LEVEL;
    controller.pml_width = 8;
    controller.left_border = ShallowWaterBorder::Ocean;
    controller.right_border = ShallowWaterBorder::Ocean;
    controller.bottom_border = ShallowWaterBorder::Ocean;
    controller.top_border = ShallowWaterBorder::Ocean;
    controller.ocean_forcing = true;
    controller.gravity = 9.81;
    controller.dt = 1.0 / 60.0;
    controller.brush_radius = 6.0;
    commands.insert_resource(controller);

    let buoy_mesh = meshes.add(Sphere::new(1.2).mesh().uv(24, 12));
    let buoy_material = standard_materials.add(StandardMaterial {
        base_color: Color::srgb(0.95, 0.35, 0.1),
        perceptual_roughness: 0.5,
        ..default()
    });
    for xz in BUOY_POSITIONS {
        commands.spawn((
            Mesh3d(buoy_mesh.clone()),
            MeshMaterial3d(buoy_material.clone()),
            Transform::from_xyz(xz.x, 0.0, xz.y),
            Buoy { xz },
        ));
    }

    let earth_medium = scattering_mediums.add(ScatteringMedium::earth(256, 256));
    commands.spawn(Atmosphere::earth(earth_medium));
    let camera = commands
        .spawn((
            Camera3d::default(),
            Transform::from_xyz(0.0, 1500.0, 2300.0)
                .looking_at(Vec3::new(0.0, 0.0, -150.0), Vec3::Y),
            FreeCamera {
                walk_speed: 60.0,
                run_speed: 400.0,
                ..default()
            },
            AtmosphereSettings::default(),
            Exposure { ev100: 12.0 },
            Tonemapping::AcesFitted,
            Bloom::NATURAL,
            AtmosphereEnvironmentMapLight::default(),
            Msaa::Off,
            Fxaa::default(),
            DepthPrepass,
            DeferredPrepass,
        ))
        .id();
    commands.spawn_terrain(
        asset_server.load(TERRAIN_CONFIG),
        TerrainViewConfig {
            view_lod: TERRAIN_VIEW_LOD,
            ..default()
        },
        KauaiGroundMaterial::default(),
        camera,
    );

    commands.spawn((
        DirectionalLight {
            shadow_maps_enabled: true,
            illuminance: lux::RAW_SUNLIGHT,
            ..default()
        },
        Transform::from_xyz(-0.4, 0.35, 1.0).looking_at(Vec3::ZERO, Vec3::Y),
        CascadeShadowConfigBuilder {
            first_cascade_far_bound: 300.0,
            maximum_distance: 6000.0,
            ..default()
        }
        .build(),
    ));
}

#[allow(clippy::type_complexity)]
fn spawn_surface_when_ready(
    mut commands: Commands,
    mut materials: ResMut<Assets<WaterSurfaceMaterial>>,
    cascades: Query<&OceanSimSettings, (With<FftTextures>, With<OceanFoamMask>)>,
    existing: Query<(), With<WaterSurfaceTag>>,
) {
    if !existing.is_empty() || cascades.iter().count() < CASCADE_TILES.len() {
        return;
    }
    commands.spawn((
        // Thirteen levels reach 65 km, so the water meets the horizon from above the map.
        OceanClipmap {
            levels: 13,
            ..default()
        },
        MeshMaterial3d(materials.add(WaterSurfaceMaterial {
            base: StandardMaterial {
                base_color: Color::srgb(0.04, 0.12, 0.22),
                perceptual_roughness: 0.08,
                metallic: 0.0,
                opaque_render_method: OpaqueRendererMethod::Auto,
                ..default()
            },
            extension: WaterSurfaceExtension {
                settings: WaterSurfaceUniform {
                    dry_depth: 0.03,
                    ..WaterSurfaceExtension::default().settings
                },
                ..default()
            },
        })),
        WaterSurfaceTag,
    ));
}

/// Places each buoy on the queried surface, extrapolated over the readback latency.
fn float_buoys(
    mut queries: ResMut<OceanQueries>,
    mut buoys: Query<(&Buoy, &mut Transform)>,
    time: Res<Time>,
) {
    if queries.points().is_empty() {
        queries.set_points(BUOY_POSITIONS);
    }
    let Some(results) = queries.results() else {
        return;
    };
    let age = time.elapsed_secs() - results.time;
    for (buoy, mut transform) in &mut buoys {
        let Some(i) = results.points.iter().position(|p| *p == buoy.xz) else {
            continue;
        };
        let sample = results.samples[i];
        transform.translation = Vec3::new(buoy.xz.x, sample.height_after(age), buoy.xz.y);
        transform.rotation = Quat::from_rotation_arc(Vec3::Y, sample.normal());
    }
}

fn sync_free_camera_with_egui_focus(
    mut state: Query<&mut FreeCameraState, With<FreeCamera>>,
    egui_wants: Res<EguiWantsInput>,
) {
    if let Ok(mut camera_state) = state.single_mut() {
        camera_state.enabled = !egui_wants.wants_any_input();
    }
}

/// Maps the cursor onto the sea plane and into grid cells for the water brush.
fn pointer_to_sim_space(
    mouse: Res<ButtonInput<MouseButton>>,
    egui_wants: Res<EguiWantsInput>,
    window: Query<&Window, With<bevy::window::PrimaryWindow>>,
    camera: Query<(&Camera, &GlobalTransform), With<FreeCamera>>,
    mut controller: ResMut<ShallowWaterController>,
) {
    controller.brush_input_active = false;
    let (Ok(window), Ok((camera, cam_transform))) = (window.single(), camera.single()) else {
        return;
    };
    let Some(placement) = controller.placement else {
        return;
    };
    let Some(cursor) = window.cursor_position() else {
        return;
    };
    let Ok(ray) = camera.viewport_to_world(cam_transform, cursor) else {
        return;
    };
    let Some(hit) = ray.plane_intersection_point(Vec3::ZERO, InfinitePlane3d::new(Vec3::Y)) else {
        return;
    };
    let cells = UVec2::new(controller.cells_x, controller.cells_y);
    let sim = placement.world_to_cells(hit.xz(), cells);
    let paint = mouse.pressed(MouseButton::Left) && !egui_wants.is_pointer_over_area();
    controller.pointer_prev_sim = if paint && !mouse.just_pressed(MouseButton::Left) {
        controller.pointer_sim
    } else {
        sim
    };
    controller.pointer_sim = sim;
    controller.brush_input_active = paint;
}

fn beach_panel(mut contexts: EguiContexts, mut controller: ResMut<ShallowWaterController>) {
    let Ok(ctx) = contexts.ctx_mut() else {
        return;
    };
    egui::Window::new("Beach").show(ctx, |ui| {
        ui.checkbox(&mut controller.paused, "Paused");
        if ui.button("Reset").clicked() {
            controller.sim_apply_serial = controller.sim_apply_serial.wrapping_add(1);
        }
        if let Some(mut placement) = controller.placement {
            ui.add(egui::Slider::new(&mut placement.blend_cells, 0.0..=64.0).text("Edge fade"));
            controller.placement = Some(placement);
        }
        ui.add(egui::Slider::new(&mut controller.pml_width, 0u32..=32u32).text("PML width"));
        ui.add(egui::Slider::new(&mut controller.ocean_band_cells, 0.0..=16.0).text("Ocean band"));
        ui.add(egui::Slider::new(&mut controller.friction, 0.0..=0.5).text("Friction"));
        ui.horizontal(|ui| {
            ui.label("Map edges");
            let mut border = controller.top_border;
            egui::ComboBox::from_id_salt("map_edges")
                .selected_text(format!("{border:?}"))
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut border, ShallowWaterBorder::Ocean, "Ocean");
                    ui.selectable_value(&mut border, ShallowWaterBorder::Wall, "Wall");
                    ui.selectable_value(&mut border, ShallowWaterBorder::Drain, "Drain");
                });
            if border != controller.top_border {
                controller.left_border = border;
                controller.right_border = border;
                controller.bottom_border = border;
                controller.top_border = border;
            }
        });
        ui.add(egui::Slider::new(&mut controller.brush_radius, 2.0..=24.0).text("Brush radius"));
    });
}

/// Terrain heights in meters from the baked bevy_terrain tiles, for the shallow water bed.
///
/// Each tile holds a 1 texel border around its centre texels. Tile `(x, y)` covers the terrain's
/// UV square `[x, x + 1] / n × [y, y + 1] / n`, with texel rows running along +Z.
struct TerrainBed {
    tiles: Vec<Vec<f32>>,
    size: usize,
}

impl TerrainBed {
    const BORDER: usize = 1;

    fn load() -> Option<Self> {
        let n = TERRAIN_BED_TILE_COUNT;
        let mut tiles = Vec::with_capacity(n * n);
        let mut size = 0;
        for y in 0..n {
            for x in 0..n {
                let path = format!("{TERRAIN_BED_TILES}/0_0/0_2_{x}_{y}.tif");
                let file = std::fs::File::open(path).ok()?;
                let mut decoder =
                    tiff::decoder::Decoder::new(std::io::BufReader::new(file)).ok()?;
                size = decoder.dimensions().ok()?.0 as usize;
                let tiff::decoder::DecodingResult::F32(data) = decoder.read_image().ok()? else {
                    return None;
                };
                tiles.push(data);
            }
        }
        Some(Self { tiles, size })
    }

    /// Bilinear height at world `xz`, clamped to the terrain square.
    fn height(&self, xz: Vec2) -> f32 {
        let n = TERRAIN_BED_TILE_COUNT as f32;
        let center = (self.size - 2 * Self::BORDER) as f32;
        let uv = (xz / TERRAIN_SIDE + 0.5).clamp(Vec2::ZERO, Vec2::splat(1.0 - 1e-6)) * n;
        let tile = uv.floor();
        let texel = (uv - tile) * center - 0.5 + Self::BORDER as f32;
        let data = &self.tiles[tile.y as usize * TERRAIN_BED_TILE_COUNT + tile.x as usize];
        let i0 = texel.floor();
        let f = texel - i0;
        let at = |dx: usize, dy: usize| {
            let x = (i0.x as usize + dx).min(self.size - 1);
            let y = (i0.y as usize + dy).min(self.size - 1);
            data[y * self.size + x]
        };
        let top = at(0, 0) + (at(1, 0) - at(0, 0)) * f.x;
        let bottom = at(0, 1) + (at(1, 1) - at(0, 1)) * f.x;
        top + (bottom - top) * f.y
    }
}
