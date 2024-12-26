#define_import_path bevy_fft::c32_2

struct c32_2 {
    real: vec2<f32>,
    imag: vec2<f32>,
}

fn from_u32(packed: vec2<u32>) -> c32_2 {
    let unpacked_x = unpack2x16float(packed);
    let unpacked_y = unpack2x16float(packed);
    return c32_2(
        vec2(unpacked_x.x, unpacked_y.x),
        vec2(unpacked_x.y, unpacked_y.y)
    );
}

fn into_u32(unpacked: c32_2) -> vec2<u32> {
    let packed_x = pack2x16float(vec2(unpacked.real.x, unpacked.imag.x));
    let packed_y = pack2x16float(vec2(unpacked.real.y, unpacked.imag.y));
    return vec2(packed_x, packed_y);
}

fn mul(c1: c32_2, c2: c32_2) -> c32_2 {
    let real = c1.real * c2.real - c1.imag * c2.imag;
    let imag = c1.real * c2.imag + c1.imag * c2.real;
    return c32_2(real, imag);
}

fn mul_f(c: c32_2, f: f32) -> c32_2 {
    return c32_2(c.real * f, c.imag * f);
}

fn mul_v(c: c32_2, f: vec2<f32>) -> c32_2 {
    return c32_2(c.real * f, c.imag * f);
}

fn add(c1: c32_2, c2: c32_2) -> c32_2 {
    return c32_2(c1.real + c2.real, c1.imag + c2.imag);
}

fn cfma(c1: c32_2, f2: vec2<f32>, c3: c32_2) -> c32_2 {
    return c32_2(fma(c1.real, f2, c3.real), fma(c1.imag, f2, c3.imag));
}

fn exp(theta: vec2<f32>) -> c32_2_2 {
    return c32_2(cos(theta), sin(theta));
}
