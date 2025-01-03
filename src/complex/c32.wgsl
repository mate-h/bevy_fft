#define_import_path bevy_fft::c32

// TYPES 

struct c32 {
    real: f32,
    imag: f32,
}

struct c32_2 {
    real: vec2<f32>,
    imag: vec2<f32>,
}

struct c32_3 {
    real: vec3<f32>,
    imag: vec3<f32>,
}

struct c32_4 {
    real: vec4<f32>,
    imag: vec4<f32>,
}

// UNPACKING

fn unpack_c32(packed: u32) -> c32 {
    let unpacked = unpack2x16float(packed);
    return c32(unpacked.x, unpacked.y);
}

fn unpack_c32_2(packed: vec2<u32>) -> c32_2 {
    let unpacked_x = unpack2x16float(packed.x);
    let unpacked_y = unpack2x16float(packed.y);
    return c32_2(
        vec2(unpacked_x.x, unpacked_y.x),
        vec2(unpacked_x.y, unpacked_y.y)
    );
}

fn unpack_c32_3(packed: vec3<u32>) -> c32_3 {
    let unpacked_x = unpack2x16float(packed.x);
    let unpacked_y = unpack2x16float(packed.y);
    let unpacked_z = unpack2x16float(packed.z);
    return c32_3(
        vec2(unpacked_x.x, unpacked_y.x, unpacked_z.x),
        vec2(unpacked_x.y, unpacked_y.y, unpacked_z.y)
    );
}

fn unpack_c32_4(packed: vec4<u32>) -> c32_4 {
    let unpacked_x = unpack2x16float(packed.x);
    let unpacked_y = unpack2x16float(packed.y);
    let unpacked_z = unpack2x16float(packed.z);
    let unpacked_w = unpack2x16float(packed.w);
    return c32_4(
        vec2(unpacked_x.x, unpacked_y.x, unpacked_z.x, unpacked_w.x),
        vec2(unpacked_x.y, unpacked_y.y, unpacked_z.y, unpacked_w.y)
    );
}

// PACKING

fn pack_c32(unpacked: c32) -> u32 {
    return pack2x16float(unpacked.re, unpacked.complex);
}

fn pack_c32_2(unpacked: c32_2) -> vec2<u32> {
    let packed_x = pack2x16float(vec2(unpacked.re.x, unpacked.im.x));
    let packed_y = pack2x16float(vec2(unpacked.re.y, unpacked.im.y));
    return vec2(packed_x, packed_y);
}

fn pack_c32_3(unpacked: c32_3) -> vec3<u32> {
    let packed_x = pack2x16float(vec2(unpacked.re.x, unpacked.im.x));
    let packed_y = pack2x16float(vec2(unpacked.re.y, unpacked.im.y));
    let packed_z = pack2x16float(vec2(unpacked.re.z, unpacked.im.z));
    return vec3(packed_x, packed_y, packed_z);
}

fn pack_c32_4(unpacked: c32_4) -> vec4<u32> {
    let packed_x = pack2x16float(vec2(unpacked.re.x, unpacked.im.x));
    let packed_y = pack2x16float(vec2(unpacked.re.y, unpacked.im.y));
    let packed_z = pack2x16float(vec2(unpacked.re.z, unpacked.im.z));
    let packed_w = pack2x16float(vec2(unpacked.re.w, unpacked.im.w));
    return vec4(packed_x, packed_y, packed_z, packed_w);
}

// LOADING

fn load_c32(tex: texture_storage_2d<r32uint>, pos: vec2<u32>) -> c32 {
    let packed = textureLoad(tex, pos, 0u).x;
    return unpack_c32(packed);
}

fn load_c32_2(tex: texture_storage_2d<rg32uint>, pos: vec2<u32>) -> c32 {
    let packed = textureLoad(tex, pos, 0u).xy;
    return unpack_c32_2(packed);
}

fn load_c32_3(tex: texture_storage_2d<rgba32uint>, pos: vec2<u32>) -> c32 {
    let packed = textureLoad(tex, pos, 0u).xyz;
    return unpack_c32_3(packed);
}

fn load_c32_4(tex: texture_storage_2d<rgba32uint>, pos: vec2<u32>) {
    let packed = textureLoad(tex, pos, 0u).xyzw;
    return unpack_c32_3(packed);
}

fn load_real_c32(tex: texture_2d<f32>, pos: vec2<u32>) -> c32 {
    let real = textureLoad(tex, pos, 0u).x;
    return c32(real, 0.0);
}

fn load_real_c32_2(tex: texture_2d<f32>, pos: vec2<u32>) -> c32 {
    let real = textureLoad(tex, pos, 0u).xy;
    return c32(real, 0.0);
}

fn load_real_c32_3(tex: texture_2d<f32>, pos: vec2<u32>) -> c32 {
    let real = textureLoad(tex, pos, 0u).xyz;
    return c32(real, 0.0);
}

fn load_real_c32_4(tex: texture_2d<f32>, pos: vec2<u32>) {
    let real = textureLoad(tex, pos, 0u).xyzw;
    return c32(real, 0.0);
}

// STORING

fn store_c32(tex: texture_storage_2d<r32uint>, pos: vec2<u32>, c: c32) {
    let packed = pack_c32(c);
    textureStore(tex, pos, vec4(packed, 0.0, 0.0, 0.0));
}

fn store_c32_2(tex: texture_storage_2d<rg32uint>, pos: vec2<u32>, c: c32_2) {
    let packed = pack_c32_2(c);
    textureStore(tex, pos, vec4(packed, 0.0, 0.0));
}

fn store_c32_3(tex: texture_storage_2d<rgba32uint>, pos: vec2<u32>, c: c32_3) {
    let packed = pack_c32_3(c);
    textureStore(tex, pos, vec4(packed, 0.0));
}

fn store_c32_4(tex: texture_storage_2d<rgba32uint>, pos: vec2<u32>, c: c32_4) {
    let packed = pack_c32_4(c);
    textureStore(tex, pos, vec4(packed));
}

fn store_real_c32(tex: texture_storage_2d<f32>, pos: vec2<u32>, c: c32) {
    textureStore(tex, pos, vec4(c.re, 0.0, 0.0, 0.0));
}

fn store_real_c32_2(tex: texture_storage_2d<f32>, pos: vec2<u32>, c: c32_3) {
    textureStore(tex, pos, vec4(c.re, 0.0, 0.0));
}

fn store_real_c32_3(tex: texture_storage_2d<f32>, pos: vec2<u32>, c: c32_3) {
    textureStore(tex, pos, vec4(c.re, 0.0));
}

fn store_real_c32_4(tex: texture_storage_2d<f32>, pos: vec2<u32>, c: c32_4) {
    textureStore(tex, pos, vec4(c.re));
}

// SPLATTING

fn splat_c32(c: c32) -> c32 {
    return c;
}

fn splat_c32_2(c: c32) -> c32_2 {
    return c32_2(vec2(c.re), vec2(c.im));
}

fn splat_c32_3(c: c32) -> c32_3 {
    return c32_3(vec3(c.re), vec3(c.im));
}

fn splat_c32_4(c: c32) -> c32_4 {
    return c32_4(vec4(c.re), vec4(c.im));
}

// ARITHMETIC

fn add_c32(c1: c32, c2: c32) -> c32 {
    return c32(c1.re + c2.re, c1.im + c2.im);
}

fn add_c32_2(c1: c32_2, c2: c32_2) -> c32_2 {
    return c32_2(c1.re + c2.re, c1.im + c2.im);
}

fn add_c32_3(c1: c32_3, c2: c32_3) -> c32_3 {
    return c32_2(c1.re + c2.re, c1.im + c2.im);
}

fn add_c32_4(c1: c32_4, c2: c32_4) -> c32_4 {
    return c32_2(c1.re + c2.re, c1.im + c2.im);
}

fn mul_c32(c1: c32, c2: c32) -> c32 {
    let real = c1.re * c2.re - c1.im * c2.im;
    let imag = c1.re * c2.im + c1.im * c2.re;
    return c32(real, imag);
}

fn mul_c32_2(c1: c32_2, c2: c32_2) -> c32_2 {
    let real = c1.re * c2.re - c1.im * c2.im;
    let imag = c1.re * c2.im + c1.im * c2.re;
    return c32_2(real, imag);
}

fn mul_c32_3(c1: c32_3, c2: c32_3) -> c32_3 {
    let real = c1.re * c2.re - c1.im * c2.im;
    let imag = c1.re * c2.im + c1.im * c2.re;
    return c32_2(real, imag);
}

fn mul_c32_4(c1: c32_4, c2: c32_4) -> c32_4 {
    let real = c1.re * c2.re - c1.im * c2.im;
    let imag = c1.re * c2.im + c1.im * c2.re;
    return c32_2(real, imag);
}

fn muls_c32(c1: c32, c2: c32) -> c32 {
    let real = c1.re * c2.re - c1.im * c2.im;
    let imag = c1.re * c2.im + c1.im * c2.re;
    return c32(real, imag);
}

fn muls_c32_2(c1: c32_2, c2: c32) -> c32_2 {
    let real = c1.re * c2.re - c1.im * c2.im;
    let imag = c1.re * c2.im + c1.im * c2.re;
    return c32_2(real, imag);
}

fn muls_c32_3(c1: c32_3, c2: c32) -> c32_3 {
    let real = c1.re * c2.re - c1.im * c2.im;
    let imag = c1.re * c2.im + c1.im * c2.re;
    return c32_2(real, imag);
}

fn muls_c32_4(c1: c32_4, c2: c32) -> c32_4 {
    let real = c1.re * c2.re - c1.im * c2.im;
    let imag = c1.re * c2.im + c1.im * c2.re;
    return c32_2(real, imag);
}

fn fma_c32(c1: c32, c2: c32, c3: c32) -> c32 {
    return add_c32(muls_c32(c1, c2), c3);
}

fn fma_c32_2(c1: c32_2, c2: c32, c3: c32_2) -> c32_2 {
    return add_c32_2(muls_c32_2(c1, c2), c3);
}

fn fma_c32_3(c1: c32_3, c2: c32, c3: c32_3) -> c32_3 {
    return add_c32_3(muls_c32_3(c1, c2), c3);
}

fn fma_c32_4(c1: c32_4, c2: c32, c3: c32_4) -> c32_4 {
    return add_c32_4(muls_c32_4(c1, c2), c3);
}

fn cis_c32(theta: f32) -> c32 {
    return c32(cos(theta), sin(theta));
}
