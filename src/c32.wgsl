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

// PACKING

fn unpack_c32(packed: u32) -> c32 {
    let unpacked = unpack2x16float(packed);
    return c32(unpacked.x, unpacked.y);
}

fn unpack_c32_2(packed: vec2<u32>) -> c32_2 {
    let unpacked_x = unpack2x16float(packed);
    let unpacked_y = unpack2x16float(packed);
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

// UNPACKING

fn pack_c32(unpacked: c32) -> u32 {
    return pack2x16float(unpacked.real, unpacked.complex);
}

fn pack_c32_2(unpacked: c32_2) -> vec2<u32> {
    let packed_x = pack2x16float(vec2(unpacked.real.x, unpacked.imag.x));
    let packed_y = pack2x16float(vec2(unpacked.real.y, unpacked.imag.y));
    return vec2(packed_x, packed_y);
}

fn pack_c32_3(unpacked: c32_3) -> vec3<u32> {
    let packed_x = pack2x16float(vec2(unpacked.real.x, unpacked.imag.x));
    let packed_y = pack2x16float(vec2(unpacked.real.y, unpacked.imag.y));
    let packed_z = pack2x16float(vec2(unpacked.real.z, unpacked.imag.z));
    return vec3(packed_x, packed_y, packed_z);
}

fn pack_c32_4(unpacked: c32_4) -> vec3<u32> {
    let packed_x = pack2x16float(vec2(unpacked.real.x, unpacked.imag.x));
    let packed_y = pack2x16float(vec2(unpacked.real.y, unpacked.imag.y));
    let packed_z = pack2x16float(vec2(unpacked.real.z, unpacked.imag.z));
    let packed_w = pack2x16float(vec2(unpacked.real.w, unpacked.imag.w));
    return vec3(packed_x, packed_y, packed_z, packed_w);
}

// ARITHMETIC

fn add_c32(c1: c32, c2: c32) -> c32 {
    return c32(c1.real + c2.real, c1.imag + c2.imag);
}

fn add_c32_2(c1: c32_2, c2: c32_2) -> c32_2 {
    return c32_2(c1.real + c2.real, c1.imag + c2.imag);
}

fn add_c32_3(c1: c32_3, c2: c32_3) -> c32_3 {
    return c32_2(c1.real + c2.real, c1.imag + c2.imag);
}

fn add_c32_4(c1: c32_4, c2: c32_4) -> c32_4 {
    return c32_2(c1.real + c2.real, c1.imag + c2.imag);
}

fn mul_c32(c1: c32, c2: c32) -> c32 {
    let real = c1.real * c2.real - c1.imag * c2.imag;
    let imag = c1.real * c2.imag + c1.imag * c2.real;
    return c32(real, imag);
}

fn mul_c32_2(c1: c32_2, c2: c32_2) -> c32_2 {
    let real = c1.real * c2.real - c1.imag * c2.imag;
    let imag = c1.real * c2.imag + c1.imag * c2.real;
    return c32_2(real, imag);
}

fn mul_c32_3(c1: c32_3, c2: c32_3) -> c32_3 {
    let real = c1.real * c2.real - c1.imag * c2.imag;
    let imag = c1.real * c2.imag + c1.imag * c2.real;
    return c32_2(real, imag);
}

fn mul_c32_4(c1: c32_4, c2: c32_4) -> c32_4 {
    let real = c1.real * c2.real - c1.imag * c2.imag;
    let imag = c1.real * c2.imag + c1.imag * c2.real;
    return c32_2(real, imag);
}

// fma(c1, c2, c3) = c1 * c2 + c3
// these are manually implemented for scalar*vector multiplication

fn fma_c32(c1: c32, c2: c32, c3: c32) -> c32 {
    let real = c1.real * c2.real - c1.imag * c2.imag;
    let imag = c1.real * c2.imag + c1.imag * c2.real;
    return c32_2(real + c3.real, imag + c3.imag);
}

fn fma_c32_2(c1: c32_2, c2: c32, c3: c32_2) -> c32_2 {
    let real = c1.real * c2.real - c1.imag * c2.imag;
    let imag = c1.real * c2.imag + c1.imag * c2.real;
    return c32_2(real + c3.real, imag + c3.imag);
}

fn fma_c32_3(c1: c32_3, c2: c32, c3: c32_3) -> c32_3 {
    let real = c1.real * c2.real - c1.imag * c2.imag;
    let imag = c1.real * c2.imag + c1.imag * c2.real;
    return c32_2(real + c3.real, imag + c3.imag);
}

fn fma_c32_4(c1: c32_4, c2: c32, c3: c32_4) -> c32_4 {
    let real = c1.real * c2.real - c1.imag * c2.imag;
    let imag = c1.real * c2.imag + c1.imag * c2.real;
    return c32_2(real + c3.real, imag + c3.imag);
}

fn exp_c32(theta: f32) -> c32 {
    return c32(cos(theta), sin(theta));
}

fn exp_c32_2(theta: vec2<f32>) -> c32_2 {
    return c32_2(cos(theta), sin(theta));
}

fn exp_c32_3(theta: vec3<f32>) -> c32_3 {
    return c32_2(cos(theta), sin(theta));
}


fn exp_c32_4(theta: vec4<f32>) -> c32_4 {
    return c32_2(cos(theta), sin(theta));
}
