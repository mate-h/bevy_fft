use bevy::{
    core_pipeline::core_2d::graph::Core2d,
    prelude::*,
    render::{
        RenderApp,
        render_graph::{Node, NodeRunError, RenderGraphContext, RenderGraphExt},
        render_resource::{
            CachedComputePipelineId, ComputePassDescriptor, ComputePipelineDescriptor,
            PipelineCache,
        },
        renderer::RenderContext,
    },
    shader::ShaderDefVal,
};
use bevy_fft::{
    complex::c32,
    fft::{
        FftInputTexture, FftNode, FftPlugin, FftSettings, FftSource, FftTextures,
        resources::{FftBindGroupLayouts, FftBindGroups},
    },
};

fn main() {
    App::new()
        .add_plugins((DefaultPlugins, FftPlugin, PatternPlugin))
        .add_systems(Startup, setup)
        .add_systems(Update, update_output_sprites)
        .run();
}

#[derive(Component)]
struct OutputImage;

// 1. Create an identifier component
#[derive(Component)]
struct GridPosition {
    index: usize,
}

struct PatternPlugin;

fn setup(mut commands: Commands) {
    // Camera
    commands.spawn(Camera2d);

    // Clone the images to release the immutable borrow on 'images'
    let size = Vec2::splat(256.0);
    // Calculate FFT roots
    let mut roots = [c32::new(0.0, 0.0); 8192];
    for order in 0..13 {
        let base = 1 << order;
        let count = base >> 1;
        for k in 0..count {
            let theta = -2.0 * std::f32::consts::PI * (k as f32) / (base as f32);
            let root = c32::new(theta.cos(), theta.sin());
            roots[base + k] = root;
        }
    }

    // Spawn FFT entity. With `inverse: true`, the example pattern shader writes
    // directly into the IFFT input buffers (see `assets/examples/pattern.wgsl`).
    //
    // To drive the FFT from a CPU-side image instead, insert `FftInputTexture`
    // whose real (and optional imag) textures match `size` exactly — e.g.
    // `showcase.png` in the README is not 256×256, so copying would be skipped.
    commands.spawn(FftSource {
        size: size.as_uvec2(),
        orders: 8,
        padding: UVec2::ZERO,
        roots,
        inverse: true,
    });

    let columns = 4;
    let rows = 2;
    let spacing = 256.0 + 8.0;
    let start_x = -((columns - 1) as f32) * spacing / 2.0;
    let start_y = ((rows - 1) as f32) * spacing / 2.0;

    for row in 0..rows {
        for col in 0..columns {
            let index = row * columns + col;
            let x = start_x + col as f32 * spacing;
            let y = start_y - row as f32 * spacing;

            commands.spawn((
                Sprite {
                    custom_size: Some(size),
                    ..default()
                },
                Transform::from_xyz(x, y, 0.0),
                OutputImage,
                GridPosition { index },
            ));
        }
    }
}

// Render-world pattern generation node, moved to the example. This runs only
// when no external input texture has been provided.
#[derive(Resource)]
struct ExamplePatternPipeline {
    pattern_generation: CachedComputePipelineId,
}

impl FromWorld for ExamplePatternPipeline {
    fn from_world(world: &mut World) -> Self {
        let pipeline_cache = world.resource::<PipelineCache>();
        let layouts = world.resource::<FftBindGroupLayouts>();
        let asset_server = world.resource::<AssetServer>();
        let pattern = asset_server.load("examples/pattern.wgsl");

        let pattern_generation = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("example_pattern_generation_pipeline".into()),
            layout: vec![layouts.common.clone()],
            push_constant_ranges: vec![],
            shader: pattern.clone(),
            shader_defs: vec![ShaderDefVal::UInt("CHANNELS".into(), 4)],
            entry_point: Some("generate_concentric_circles".into()),
            zero_initialize_workgroup_memory: false,
        });

        Self { pattern_generation }
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
        let pattern_pipeline = world.resource::<ExamplePatternPipeline>();
        let pipeline_cache = world.resource::<PipelineCache>();

        let Some(pattern_pipeline) =
            pipeline_cache.get_compute_pipeline(pattern_pipeline.pattern_generation)
        else {
            return Ok(());
        };

        for (bind_groups, settings, input_texture) in self.query.iter_manual(world) {
            // If an external texture was provided, skip generating the pattern.
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

impl Plugin for PatternPlugin {
    fn build(&self, _app: &mut App) {}

    fn finish(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .init_resource::<ExamplePatternPipeline>()
            .add_render_graph_node::<ExamplePatternNode>(Core2d, FftNode::GeneratePattern)
            .add_render_graph_edge(Core2d, FftNode::GeneratePattern, FftNode::ComputeFFT);
    }
}

// System to update the output sprite with the FFT texture.
// IFFT writes the viridis visualization to `buffer_d_*`; intermediate A/B/C are raw complex
// (often nearly black when displayed as linear RGB), so we show D first in the grid.
fn update_output_sprites(
    fft_query: Query<&FftTextures>,
    mut outputs: Query<(&mut Sprite, &GridPosition), With<OutputImage>>,
) {
    if let Ok(fft_textures) = fft_query.single() {
        for (mut sprite, grid_pos) in outputs.iter_mut() {
            match grid_pos.index {
                0 => sprite.image = fft_textures.buffer_d_re.clone(),
                1 => sprite.image = fft_textures.buffer_d_im.clone(),
                2 => sprite.image = fft_textures.buffer_a_re.clone(),
                3 => sprite.image = fft_textures.buffer_a_im.clone(),
                4 => sprite.image = fft_textures.buffer_b_re.clone(),
                5 => sprite.image = fft_textures.buffer_b_im.clone(),
                6 => sprite.image = fft_textures.buffer_c_re.clone(),
                7 => sprite.image = fft_textures.buffer_c_im.clone(),
                _ => sprite.image = fft_textures.buffer_d_re.clone(),
            }
        }
    }
}
