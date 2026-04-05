//! Procedural patterns written into FFT buffer A (demo only).

use bevy::{
    core_pipeline::core_2d::graph::Core2d,
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
            PipelineCache,
        },
        renderer::RenderContext,
        texture::GpuImage,
    },
    shader::ShaderDefVal,
};
use bevy_fft::fft::{
    FftInputTexture, FftNode, FftSettings, FftTextures,
    resources::{FftBindGroupLayouts, FftBindGroups},
};

#[derive(Resource, Clone, Copy, Default, PartialEq, Eq, ExtractResource)]
pub(crate) struct InputPattern(InputPatternKind);

#[derive(Clone, Copy, Default, PartialEq, Eq)]
enum InputPatternKind {
    #[default]
    Radial,
    Horizontal,
}

/// GPU snapshot of buffer A (real) after the pattern pass, for input visualization.
#[derive(Component, Clone, ExtractComponent)]
pub(crate) struct InputPatternTextures {
    pub(crate) re: Handle<Image>,
}

pub(crate) struct PatternPlugin;

impl Plugin for PatternPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<InputPattern>().add_plugins((
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
            .add_render_graph_node::<CopyInputTextureNode>(Core2d, PatternGraph::CopyInputTexture)
            .add_render_graph_edge(
                Core2d,
                FftNode::GeneratePattern,
                PatternGraph::CopyInputTexture,
            )
            .add_render_graph_edge(Core2d, PatternGraph::CopyInputTexture, FftNode::ComputeFFT);
    }
}

pub(crate) fn switch_pattern(
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

#[derive(PartialEq, Eq, Debug, Copy, Clone, Hash, RenderLabel)]
enum PatternGraph {
    CopyInputTexture,
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
            shader: pattern,
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

struct CopyInputTextureNode {
    query: QueryState<(&'static FftTextures, &'static InputPatternTextures)>,
}

impl FromWorld for CopyInputTextureNode {
    fn from_world(world: &mut World) -> Self {
        Self {
            query: world.query(),
        }
    }
}

impl Node for CopyInputTextureNode {
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

            encoder.copy_texture_to_texture(
                src_re.texture.as_image_copy(),
                dst_re.texture.as_image_copy(),
                src_re.size,
            );
        }

        Ok(())
    }
}
