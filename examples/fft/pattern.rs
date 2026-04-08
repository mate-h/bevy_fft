//! Procedural GPU patterns or PNG inputs through [`bevy_fft::fft::FftInputTexture`] using assets such as `sunflower.png` and `clouds.png`.

use crate::band_pass::BandPassParams;
use bevy::{
    asset::RenderAssetUsages,
    ecs::{change_detection::Mut, world::FromWorld},
    input::keyboard::KeyCode,
    prelude::*,
    render::{
        RenderApp,
        extract_component::{ExtractComponent, ExtractComponentPlugin},
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        render_asset::RenderAssets,
        render_graph::{Node, NodeRunError, RenderGraph, RenderGraphContext, RenderLabel},
        render_resource::{
            CachedComputePipelineId, ComputePassDescriptor, ComputePipelineDescriptor, Extent3d,
            PipelineCache, TextureDimension, TextureFormat,
        },
        renderer::RenderContext,
        texture::GpuImage,
    },
    shader::ShaderDefVal,
};
use bevy_fft::fft::{
    FftInputTexture, FftNode, FftSettings, FftSource, FftTextures,
    resources::{FftBindGroupLayouts, FftBindGroups},
};
use bytemuck::cast_slice;
use image::imageops::FilterType;

const PATTERN_SHADER: &str = "examples/pattern.wgsl";
const WG: u32 = 16;

#[derive(Resource, Clone, Copy, Default, PartialEq, Eq, ExtractResource)]
pub(crate) struct InputPattern(InputPatternKind);

#[derive(Clone, Copy, Default, PartialEq, Eq)]
enum InputPatternKind {
    #[default]
    Radial,
    Horizontal,
    Sunflower,
    Clouds,
}

impl InputPatternKind {
    #[inline]
    fn uses_external_png(self) -> bool {
        matches!(self, Self::Sunflower | Self::Clouds)
    }
}

#[derive(Component, Clone, ExtractComponent)]
pub(crate) struct InputPatternTextures {
    pub(crate) re: Handle<Image>,
}

#[derive(Resource)]
struct ExternalImageFftCache {
    source: Handle<Image>,
    fft_rgba32f: Option<Handle<Image>>,
    cached_for_size: UVec2,
}

#[derive(Resource)]
struct ExternalPatternImages {
    sunflower: ExternalImageFftCache,
    clouds: ExternalImageFftCache,
}

impl FromWorld for ExternalPatternImages {
    fn from_world(world: &mut World) -> Self {
        let assets = world.resource::<AssetServer>();
        Self {
            sunflower: ExternalImageFftCache {
                source: assets.load("sunflower.png"),
                fft_rgba32f: None,
                cached_for_size: UVec2::ZERO,
            },
            clouds: ExternalImageFftCache {
                source: assets.load("clouds.png"),
                fft_rgba32f: None,
                cached_for_size: UVec2::ZERO,
            },
        }
    }
}

impl ExternalPatternImages {
    fn slot_mut(&mut self, kind: InputPatternKind) -> Option<&mut ExternalImageFftCache> {
        match kind {
            InputPatternKind::Sunflower => Some(&mut self.sunflower),
            InputPatternKind::Clouds => Some(&mut self.clouds),
            _ => None,
        }
    }
}

pub(crate) struct PatternPlugin;

impl Plugin for PatternPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<InputPattern>()
            .init_resource::<ExternalPatternImages>()
            .add_systems(PreUpdate, sync_fft_external_inputs)
            .add_plugins((
                ExtractResourcePlugin::<InputPattern>::default(),
                ExtractComponentPlugin::<InputPatternTextures>::default(),
            ));
    }

    fn finish(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app.init_resource::<ExamplePatternPipeline>();
        render_app
            .world_mut()
            .resource_scope(|world, mut graph: Mut<RenderGraph>| {
                graph.add_node(
                    FftNode::GeneratePattern,
                    ExamplePatternNode::from_world(world),
                );
                graph.add_node(
                    PatternGraph::CopyInputTexture,
                    CopyInputTextureNode::from_world(world),
                );
                graph.add_node_edge(FftNode::GeneratePattern, PatternGraph::CopyInputTexture);
                graph.add_node_edge(PatternGraph::CopyInputTexture, FftNode::ComputeFFT);
            });
    }
}

pub(crate) fn switch_pattern(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut pattern: ResMut<InputPattern>,
) {
    for (key, kind) in [
        (KeyCode::Digit1, InputPatternKind::Radial),
        (KeyCode::Digit2, InputPatternKind::Horizontal),
        (KeyCode::Digit3, InputPatternKind::Sunflower),
        (KeyCode::Digit4, InputPatternKind::Clouds),
    ] {
        if keyboard.just_pressed(key) {
            pattern.0 = kind;
        }
    }
}

fn sync_fft_external_inputs(
    pattern: Res<InputPattern>,
    mut external: ResMut<ExternalPatternImages>,
    asset_server: Res<AssetServer>,
    mut images: ResMut<Assets<Image>>,
    fft: Query<(Entity, &FftSource), With<BandPassParams>>,
    mut commands: Commands,
) {
    let Ok((entity, fft_src)) = fft.single() else {
        return;
    };

    let Some(slot) = external.slot_mut(pattern.0) else {
        commands.entity(entity).remove::<FftInputTexture>();
        return;
    };

    if !asset_server.is_loaded(&slot.source) {
        commands.entity(entity).remove::<FftInputTexture>();
        return;
    }

    let Some(src_img) = images.get(&slot.source) else {
        commands.entity(entity).remove::<FftInputTexture>();
        return;
    };

    let sz = fft_src.size;
    let need_rebuild = slot.fft_rgba32f.is_none() || slot.cached_for_size != sz;

    if need_rebuild {
        slot.fft_rgba32f = None;
        if let Some(out) = image_resized_to_rgba32f(src_img, sz.x, sz.y) {
            let h = images.add(out);
            slot.fft_rgba32f = Some(h.clone());
            slot.cached_for_size = sz;
            commands.entity(entity).insert(FftInputTexture {
                real: h,
                imag: None,
            });
        } else {
            commands.entity(entity).remove::<FftInputTexture>();
        }
    } else if let Some(ref h) = slot.fft_rgba32f {
        commands.entity(entity).insert(FftInputTexture {
            real: h.clone(),
            imag: None,
        });
    }
}

fn image_resized_to_rgba32f(src: &Image, width: u32, height: u32) -> Option<Image> {
    let rgba = src.clone().try_into_dynamic().ok()?.into_rgba8();
    let resized = image::imageops::resize(&rgba, width, height, FilterType::Lanczos3);
    let mut data = Vec::with_capacity(width as usize * height as usize * 16);
    for (_, _, p) in resized.enumerate_pixels() {
        let px = [
            p.0[0] as f32 / 255.0,
            p.0[1] as f32 / 255.0,
            p.0[2] as f32 / 255.0,
            p.0[3] as f32 / 255.0,
        ];
        data.extend_from_slice(cast_slice(&px));
    }
    Some(Image::new(
        Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        data,
        TextureFormat::Rgba32Float,
        RenderAssetUsages::default(),
    ))
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
        let shader = world.resource::<AssetServer>().load(PATTERN_SHADER);
        let defs = vec![ShaderDefVal::UInt("CHANNELS".into(), 4)];

        let queue = |label: &'static str, entry: &'static str| {
            pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some(label.into()),
                layout: vec![layouts.common.clone()],
                push_constant_ranges: vec![],
                shader: shader.clone(),
                shader_defs: defs.clone(),
                entry_point: Some(entry.into()),
                zero_initialize_workgroup_memory: false,
            })
        };

        Self {
            radial: queue("example_pattern_radial", "generate_concentric_circles"),
            horizontal: queue("example_pattern_horizontal", "generate_horizontal_rgb_sine"),
        }
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

        if pattern.0.uses_external_png() {
            return Ok(());
        }

        let pipeline_id = match pattern.0 {
            InputPatternKind::Radial => pipelines.radial,
            InputPatternKind::Horizontal => pipelines.horizontal,
            _ => unreachable!(),
        };

        let Some(pipeline) = pipeline_cache.get_compute_pipeline(pipeline_id) else {
            return Ok(());
        };

        let command_encoder = render_context.command_encoder();
        for (bind_groups, settings, input_texture) in self.query.iter_manual(world) {
            if input_texture.is_some() {
                continue;
            }
            let mut pass = command_encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("example_pattern_generation_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_groups.common, &[]);
            let nx = settings.size.x.div_ceil(WG);
            let ny = settings.size.y.div_ceil(WG);
            pass.dispatch_workgroups(nx, ny, 1);
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
        let gpu = world.resource::<RenderAssets<GpuImage>>();
        let enc = render_context.command_encoder();

        for (fft, snap) in self.query.iter_manual(world) {
            let Some(src) = gpu.get(&fft.buffer_a_re) else {
                continue;
            };
            let Some(dst) = gpu.get(&snap.re) else {
                continue;
            };
            enc.copy_texture_to_texture(
                src.texture.as_image_copy(),
                dst.texture.as_image_copy(),
                src.size,
            );
        }
        Ok(())
    }
}
