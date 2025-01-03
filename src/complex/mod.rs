use bevy_render::render_resource::ShaderType;

pub mod image;

#[allow(non_camel_case_types)]
#[derive(ShaderType)]
pub struct c32 {
    pub real: f32,
    pub imag: f32,
}
