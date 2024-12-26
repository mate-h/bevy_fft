#define_import_path bevy_fft::c32

struct c32 {
    real: f32,
    imag: f32,
}

fn from_u32(packed: u32) -> c32 {
    let unpacked = unpack2x16float(packed);
    return c32(unpacked.x, unpacked.y);
}

fn into_u32(unpacked: c32) -> u32 {
    return pack2x16float(unpacked.real, unpacked.complex);
}

fn mul(c1: c32, c2: c32) -> c32 {
    let real = c1.real * c2.real - c1.imag * c2.imag;
    let imag = c1.real * c2.imag + c1.imag * c2.real;
    return c32(real, imag);
}

fn mul_f(c: c32, f: f32) -> c32 {
    return c32(c.real * f, c.imag * f);
}

fn add(c1: c32, c2: c32) -> c32 {
    return c32(c1.real + c2.real, c1.imag + c2.imag);
}

fn cfma(c1: c32, f2: f32, c3: c32) -> c32 {
    return c32(fma(c1.real, f2, c3.real), fma(c1.imag, f2, c3.imag));
}

fn exp(theta: f32) -> c32 {
    return c32(cos(theta), sin(theta));
}
