//! Ocean-style mesh displacement on the GPU. Samples a height texture, often produced by the FFT pipeline.
//!
//! Add [`OceanPlugin`] and reference [`shaders::OCEAN_MESH`] while building a custom material that
//! embeds this vertex shader. The WGSL `OceanMaterial` uniform ends with `_pad` so the struct stays
//! **16 bytes** wide. Keep the same layout in Rust or shader `Material` code that mirrors it.

use bevy::{
    app::{App, Plugin},
    asset::load_internal_asset,
    shader::Shader,
};

/// Stable handles for ocean WGSL chunks registered by [`OceanPlugin`].
pub mod shaders {
    use bevy::asset::{Handle, uuid_handle};
    use bevy::shader::Shader;

    pub const OCEAN_MESH: Handle<Shader> = uuid_handle!("b7c8e9f0-1a2b-4c3d-9e8f-7a6b5c4d3e2f");
}

/// Registers the ocean vertex shader as an internal asset at [`shaders::OCEAN_MESH`].
pub struct OceanPlugin;

impl Plugin for OceanPlugin {
    fn build(&self, app: &mut App) {
        load_internal_asset!(
            app,
            shaders::OCEAN_MESH,
            "ocean.wgsl",
            Shader::from_wgsl
        );
    }
}
