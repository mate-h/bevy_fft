#define_import_path bevy_fft::c32_3

struct c32_3 {
    real: vec3<f32>,
    imag: vec3<f32>,
}

fn from_u32(packed: vec3<u32>) -> c32_3 {
    let unpacked_x = unpack2x16float(packed.x);
    let unpacked_y = unpack2x16float(packed.y);
    let unpacked_z = unpack2x16float(packed.z);
    return c32_3(
        vec2(unpacked_x.x, unpacked_y.x, unpacked_z.x),
        vec2(unpacked_x.y, unpacked_y.y, unpacked_z.y)
    );
}

fn into_u32(unpacked: c32_3) -> vec3<u32> {
    let packed_x = pack2x16float(vec2(unpacked.real.x, unpacked.imag.x));
    let packed_y = pack2x16float(vec2(unpacked.real.y, unpacked.imag.y));
    let packed_z = pack2x16float(vec2(unpacked.real.z, unpacked.imag.z));
    return vec3(packed_x, packed_y, packed_z);
}

fn mul(c1: c32_3, c2: c32_3) -> c32_3 {
    let real = c1.real * c2.real - c1.imag * c2.imag;
    let imag = c1.real * c2.imag + c1.imag * c2.real;
    return c32_3(real, imag);
}

fn mul_f(c: c32_3, f: f32) -> c32_3 {
    return c32_3(c.real * f, c.imag * f);
}

fn mul_v(c: c32_3, f: vec3<f32>) -> c32_3 {
    return c32_3(c.real * f, c.imag * f);
}

fn add(c1: c32_3, c2: c32_3) -> c32_3 {
    return c32_3(c1.real + c2.real, c1.imag + c2.imag);
}

fn cfma(c1: c32_3, f2: vec3<f32>, c3: c32_3) -> c32_3 {
    return c32_3(fma(c1.real, f2, c3.real), fma(c1.imag, f2, c3.imag));
}

fn exp(theta: vec3<f32>) -> c32_3 {
    return c32_3(cos(theta), sin(theta));
}
