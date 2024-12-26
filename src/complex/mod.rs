use bevy_app::{App, Plugin};
use bevy_asset::{embedded_asset, AssetServer, Handle};
use bevy_ecs::{
    system::Resource,
    world::{FromWorld, World},
};
use bevy_render::render_resource::Shader;

pub struct ComplexNumsPlugin;

impl Plugin for ComplexNumsPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "c32.wgsl");
        embedded_asset!(app, "c32_2.wgsl");
        embedded_asset!(app, "c32_3.wgsl");
        embedded_asset!(app, "c32_4.wgsl");

        app.init_resource::<ComplexNumsShaders>();
    }
}

#[derive(Resource)]
#[expect(dead_code)]
struct ComplexNumsShaders {
    c32: Handle<Shader>,
    c32_2: Handle<Shader>,
    c32_3: Handle<Shader>,
    c32_4: Handle<Shader>,
}

impl FromWorld for ComplexNumsShaders {
    fn from_world(world: &mut World) -> Self {
        let asset_server = world.resource::<AssetServer>();
        let c32 = asset_server.load::<Shader>("embedded://bevy_fft/complex/c32.wgsl");
        let c32_2 = asset_server.load::<Shader>("embedded://bevy_fft/complex/c32_2.wgsl");
        let c32_3 = asset_server.load::<Shader>("embedded://bevy_fft/complex/c32_3.wgsl");
        let c32_4 = asset_server.load::<Shader>("embedded://bevy_fft/complex/c32_4.wgsl");
        Self {
            c32,
            c32_2,
            c32_3,
            c32_4,
        }
    }
}
