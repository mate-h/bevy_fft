use bevy::{
    asset::RenderAssetUsages,
    core_pipeline::core_2d::graph::Core2d,
    image::Image,
    input::keyboard::KeyCode,
    prelude::*,
    render::{
        RenderApp,
        extract_component::{ExtractComponent, ExtractComponentPlugin},
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        render_asset::RenderAssets,
        render_graph::{Node, NodeRunError, RenderGraphContext, RenderGraphExt, RenderLabel},
        render_resource::{
            CachedComputePipelineId, ComputePassDescriptor, ComputePipelineDescriptor,
            Extent3d, PipelineCache, TextureDimension, TextureFormat, TextureUsages,
        },
        renderer::RenderContext,
        texture::GpuImage,
    },
    shader::ShaderDefVal,
};
use bevy_fft::{
    complex::c32,
    fft::{
        FftInputTexture, FftNode, FftPlugin, FftSettings, FftSource, FftTextures,
        resources::{prepare_fft_textures, FftBindGroupLayouts, FftBindGroups},
    },
};

fn main() {
    App::new()
        .add_plugins((DefaultPlugins, FftPlugin, PatternPlugin))
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (
                setup_pattern_snapshot_textures.after(prepare_fft_textures),
                update_output_sprites.after(setup_pattern_snapshot_textures),
                switch_input_pattern,
            ),
        )
        .run();
}

#[derive(Component)]
struct OutputImage;

#[derive(Component)]
struct GridPosition {
    index: usize,
}

/// Copy of buffer A (real) right after pattern generation, for input visualization.
#[derive(Component, Clone, ExtractComponent)]
struct InputPatternTextures {
    re: Handle<Image>,
}

/// **1** = radial (`generate_concentric_circles`), **2** = horizontal stripes (`generate_horizontal_rgb_sine`).
#[derive(Resource, Clone, Copy, Default, PartialEq, Eq, ExtractResource)]
struct InputPattern(InputPatternKind);

#[derive(Clone, Copy, Default, PartialEq, Eq)]
enum InputPatternKind {
    #[default]
    Radial,
    Horizontal,
}

struct PatternPlugin;

fn setup(mut commands: Commands) {
    commands.spawn(Camera2d);

    let size = Vec2::splat(256.0);
    let mut roots = [c32::new(0.0, 0.0); 8192];
    bevy_fft::fft::fill_forward_fft_twiddles(&mut roots);

    commands.spawn(FftSource {
        size: size.as_uvec2(),
        orders: 8,
        padding: UVec2::ZERO,
        roots,
        inverse: false,
        roundtrip: true,
    });

    let columns = 2;
    let spacing = 256.0 + 16.0;
    let start_x = -((columns - 1) as f32) * spacing / 2.0;

    for col in 0..columns {
        let x = start_x + col as f32 * spacing;
        commands.spawn((
            Sprite {
                custom_size: Some(size),
                ..default()
            },
            Transform::from_xyz(x, 0.0, 0.0),
            OutputImage,
            GridPosition { index: col },
        ));
    }
}

fn setup_pattern_snapshot_textures(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    query: Query<(Entity, &FftSource, &FftTextures), Without<InputPatternTextures>>,
) {
    for (entity, source, _) in &query {
        let extent = Extent3d {
            width: source.size.x,
            height: source.size.y,
            depth_or_array_layers: 1,
        };
        let mut image = Image::new_fill(
            extent,
            TextureDimension::D2,
            &[0; 16],
            TextureFormat::Rgba32Float,
            RenderAssetUsages::default(),
        );
        image.texture_descriptor.usage = TextureUsages::COPY_DST | TextureUsages::TEXTURE_BINDING;

        let re = images.add(image);
        commands.entity(entity).insert(InputPatternTextures { re });
    }
}

fn switch_input_pattern(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut pattern: ResMut<InputPattern>,
) {
    if keyboard.just_pressed(KeyCode::Digit1) {
        pattern.0 = InputPatternKind::Radial;
    }
    if keyboard.just_pressed(KeyCode::Digit2) {
        pattern.0 = InputPatternKind::Horizontal;
    }
}

#[derive(Resource)]
struct ExamplePatternPipeline {
    radial: CachedComputePipelineId,
    horizontal: CachedComputePipelineId,
}

impl FromWorld for ExamplePatternPipeline {
    fn from_world(world: &mut World) -> Self {
        let pipeline_cache = world.resource::<PipelineCache>();
        let layouts = world.resource::<FftBindGroupLayouts>();
        let asset_server = world.resource::<AssetServer>();
        let pattern = asset_server.load("examples/pattern.wgsl");
        let defs = vec![ShaderDefVal::UInt("CHANNELS".into(), 4)];

        let radial = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("example_pattern_radial".into()),
            layout: vec![layouts.common.clone()],
            push_constant_ranges: vec![],
            shader: pattern.clone(),
            shader_defs: defs.clone(),
            entry_point: Some("generate_concentric_circles".into()),
            zero_initialize_workgroup_memory: false,
        });

        let horizontal = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("example_pattern_horizontal".into()),
            layout: vec![layouts.common.clone()],
            push_constant_ranges: vec![],
            shader: pattern.clone(),
            shader_defs: defs,
            entry_point: Some("generate_horizontal_rgb_sine".into()),
            zero_initialize_workgroup_memory: false,
        });

        Self { radial, horizontal }
    }
}

struct ExamplePatternNode {
    query: QueryState<(
        &'static FftBindGroups,
        &'static FftSettings,
        Option<&'static FftInputTexture>,
    )>,
}

impl FromWorld for ExamplePatternNode {
    fn from_world(world: &mut World) -> Self {
        Self {
            query: world.query(),
        }
    }
}

impl Node for ExamplePatternNode {
    fn update(&mut self, world: &mut World) {
        self.query.update_archetypes(world);
    }

    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), NodeRunError> {
        let pipelines = world.resource::<ExamplePatternPipeline>();
        let pattern = world.resource::<InputPattern>();
        let pipeline_cache = world.resource::<PipelineCache>();

        let pipeline_id = match pattern.0 {
            InputPatternKind::Radial => pipelines.radial,
            InputPatternKind::Horizontal => pipelines.horizontal,
        };

        let Some(pattern_pipeline) = pipeline_cache.get_compute_pipeline(pipeline_id) else {
            return Ok(());
        };

        for (bind_groups, settings, input_texture) in self.query.iter_manual(world) {
            if input_texture.is_some() {
                continue;
            }

            let command_encoder = render_context.command_encoder();

            let workgroup_size = 16;
            let num_workgroups_x = settings.size.x.div_ceil(workgroup_size);
            let num_workgroups_y = settings.size.y.div_ceil(workgroup_size);

            let mut compute_pass = command_encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("example_pattern_generation_pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(pattern_pipeline);
            compute_pass.set_bind_group(0, &bind_groups.common, &[]);
            compute_pass.dispatch_workgroups(num_workgroups_x, num_workgroups_y, 1);
        }

        Ok(())
    }
}

#[derive(PartialEq, Eq, Debug, Copy, Clone, Hash, RenderLabel)]
enum PatternExampleNode {
    CopyInputSnapshot,
}

struct CopyPatternInputSnapshotNode {
    query: QueryState<(&'static FftTextures, &'static InputPatternTextures)>,
}

impl FromWorld for CopyPatternInputSnapshotNode {
    fn from_world(world: &mut World) -> Self {
        Self {
            query: world.query(),
        }
    }
}

impl Node for CopyPatternInputSnapshotNode {
    fn update(&mut self, world: &mut World) {
        self.query.update_archetypes(world);
    }

    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), NodeRunError> {
        let gpu_images = world.resource::<RenderAssets<GpuImage>>();
        let encoder = render_context.command_encoder();

        for (fft, snap) in self.query.iter_manual(world) {
            let Some(src_re) = gpu_images.get(&fft.buffer_a_re) else {
                continue;
            };
            let Some(dst_re) = gpu_images.get(&snap.re) else {
                continue;
            };

            let extent = src_re.size;
            encoder.copy_texture_to_texture(
                src_re.texture.as_image_copy(),
                dst_re.texture.as_image_copy(),
                extent,
            );
        }

        Ok(())
    }
}

impl Plugin for PatternPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<InputPattern>()
            .add_plugins((
                ExtractResourcePlugin::<InputPattern>::default(),
                ExtractComponentPlugin::<InputPatternTextures>::default(),
            ));
    }

    fn finish(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .init_resource::<ExamplePatternPipeline>()
            .add_render_graph_node::<ExamplePatternNode>(Core2d, FftNode::GeneratePattern)
            .add_render_graph_node::<CopyPatternInputSnapshotNode>(
                Core2d,
                PatternExampleNode::CopyInputSnapshot,
            )
            .add_render_graph_edge(Core2d, FftNode::GeneratePattern, PatternExampleNode::CopyInputSnapshot)
            .add_render_graph_edge(
                Core2d,
                PatternExampleNode::CopyInputSnapshot,
                FftNode::ComputeFFT,
            );
    }
}

fn update_output_sprites(
    fft_query: Query<(&FftTextures, &InputPatternTextures)>,
    mut outputs: Query<(&mut Sprite, &GridPosition), With<OutputImage>>,
) {
    if let Ok((fft_textures, snap)) = fft_query.single() {
        for (mut sprite, grid_pos) in outputs.iter_mut() {
            match grid_pos.index {
                0 => sprite.image = snap.re.clone(),
                1 => sprite.image = fft_textures.spatial_output.clone(),
                _ => {}
            }
        }
    }
}
