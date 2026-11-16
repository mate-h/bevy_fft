// Mip chain of each cascade's Jacobian statistics for far whitecaps.

/// One level of the Jacobian statistics, and the next level.
@group(0) @binding(0) var mip_in: texture_2d<f32>;
@group(0) @binding(1) var mip_out: texture_storage_2d<rgba16float, write>;

/// Averages 2x2 texels of one level of the Jacobian statistics into the next. Means of the term
/// and of its square both filter linearly.
@compute @workgroup_size(8, 8, 1)
fn ocean_jacobian_downsample(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = textureDimensions(mip_out);
    if gid.x >= size.x || gid.y >= size.y {
        return;
    }
    let src = vec2<i32>(gid.xy) * 2;
    let sum = textureLoad(mip_in, src, 0) + textureLoad(mip_in, src + vec2(1, 0), 0)
        + textureLoad(mip_in, src + vec2(0, 1), 0) + textureLoad(mip_in, src + vec2(1, 1), 0);
    textureStore(mip_out, vec2<i32>(gid.xy), sum * 0.25);
}
