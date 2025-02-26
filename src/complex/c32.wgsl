#define_import_path bevy_fft::complex

// TYPES

struct c32 {
    re: f32,
    im: f32,
}

struct packed_c32 {
    value: u32,
}

struct c32_2 {
    re: vec2<f32>,
    im: vec2<f32>,
}

struct packed_c32_2 {
    value: vec2<u32>,
}

struct c32_3 {
    re: vec3<f32>,
    im: vec3<f32>,
}

struct packed_c32_3 {
    value: vec3<u32>,
}

struct c32_4 {
    re: vec4<f32>,
    im: vec4<f32>,
}

struct packed_c32_4 {
    value: vec4<u32>,
}

#ifdef CHANNELS
#if CHANNELS == 1 
    alias c32_n = c32;
    alias packed_c32_n = packed_c32;
#else if CHANNELS == 2;
    alias c32_n = c32_2;
    alias packed_c32_n = packed_c32_2;
#else if CHANNELS == 3;
    alias c32_n = c32_3;
    alias packed_c32_n = packed_c32_3;
#else if CHANNELS == 4;
    alias c32_n = c32_4;
    alias packed_c32_n = packed_c32_4;
#endif
#endif

// UNPACKING

fn unpack_c32(p: packed_c32) -> c32 {
    let packed = p.value;
    let unpacked = unpack2x16float(packed);
    return c32(unpacked.x, unpacked.y);
}

fn unpack_c32_2(p: packed_c32_2) -> c32_2 {
    let packed = p.value;
    let unpacked_x = unpack2x16float(packed.x);
    let unpacked_y = unpack2x16float(packed.y);
    return c32_2(
        vec2(unpacked_x.x, unpacked_y.x),
        vec2(unpacked_x.y, unpacked_y.y)
    );
}

fn unpack_c32_3(p: packed_c32_3) -> c32_3 {
    let packed = p.value;
    let unpacked_x = unpack2x16float(packed.x);
    let unpacked_y = unpack2x16float(packed.y);
    let unpacked_z = unpack2x16float(packed.z);
    return c32_3(
        vec3(unpacked_x.x, unpacked_y.x, unpacked_z.x),
        vec3(unpacked_x.y, unpacked_y.y, unpacked_z.y)
    );
}

fn unpack_c32_4(p: packed_c32_4) -> c32_4 {
    let packed = p.value;
    let unpacked_x = unpack2x16float(packed.x);
    let unpacked_y = unpack2x16float(packed.y);
    let unpacked_z = unpack2x16float(packed.z);
    let unpacked_w = unpack2x16float(packed.w);
    return c32_4(
        vec4(unpacked_x.x, unpacked_y.x, unpacked_z.x, unpacked_w.x),
        vec4(unpacked_x.y, unpacked_y.y, unpacked_z.y, unpacked_w.y)
    );
}

#ifdef CHANNELS
fn unpack_c32_n(packed: packed_c32_n) -> c32_n {
#if CHANNELS == 1
    return unpack_c32(packed);
#else if CHANNELS == 2
    return unpack_c32_2(packed);
#else if CHANNELS == 3
    return unpack_c32_3(packed);
#else if CHANNELS == 4
    return unpack_c32_4(packed);
#endif
}
#endif

// PACKING

fn pack_c32(unpacked: c32) -> packed_c32 {
    let packed = pack2x16float(unpacked.re, unpacked.im);
    return packed_c32(packed);
}

fn pack_c32_2(unpacked: c32_2) -> packed_c32_2 {
    let packed_x = pack2x16float(vec2(unpacked.re.x, unpacked.im.x));
    let packed_y = pack2x16float(vec2(unpacked.re.y, unpacked.im.y));
    return packed_c32_2(vec2(packed_x, packed_y));
}

fn pack_c32_3(unpacked: c32_3) -> packed_c32_3 {
    let packed_x = pack2x16float(vec2(unpacked.re.x, unpacked.im.x));
    let packed_y = pack2x16float(vec2(unpacked.re.y, unpacked.im.y));
    let packed_z = pack2x16float(vec2(unpacked.re.z, unpacked.im.z));
    return packed_c32_3(vec3(packed_x, packed_y, packed_z));
}

fn pack_c32_4(unpacked: c32_4) -> packed_c32_4 {
    let packed_x = pack2x16float(vec2(unpacked.re.x, unpacked.im.x));
    let packed_y = pack2x16float(vec2(unpacked.re.y, unpacked.im.y));
    let packed_z = pack2x16float(vec2(unpacked.re.z, unpacked.im.z));
    let packed_w = pack2x16float(vec2(unpacked.re.w, unpacked.im.w));
    return packed_c32_4(vec4(packed_x, packed_y, packed_z, packed_w));
}

#ifdef CHANNELS
fn pack_c32_n(unpacked: c32_n) -> packed_c32_n {
#if CHANNELS == 1
    return pack_c32(unpacked);
#else if CHANNELS == 2
    return pack_c32_2(unpacked);
#else if CHANNELS == 3
    return pack_c32_3(unpacked);
#else if CHANNELS == 4
    return pack_c32_4(unpacked);
#endif
}
#endif

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

#ifdef CHANNELS
fn splat_c32_n(c: c32) -> c32_n {
#if CHANNELS == 1
    return splat_c32(c);
#else if CHANNELS == 2
    return splat_c32_2(c);
#else if CHANNELS == 3
    return splat_c32_3(c);
#else if CHANNELS == 4
    return splat_c32_4(c);
#endif
}
#endif

// ARITHMETIC

fn add_c32(c1: c32, c2: c32) -> c32 {
    return c32(c1.re + c2.re, c1.im + c2.im);
}

fn add_c32_2(c1: c32_2, c2: c32_2) -> c32_2 {
    return c32_2(c1.re + c2.re, c1.im + c2.im);
}

fn add_c32_3(c1: c32_3, c2: c32_3) -> c32_3 {
    return c32_3(c1.re + c2.re, c1.im + c2.im);
}

fn add_c32_4(c1: c32_4, c2: c32_4) -> c32_4 {
    return c32_4(c1.re + c2.re, c1.im + c2.im);
}

#ifdef CHANNELS
fn add_c32_n(c1: c32_n, c2: c32_n) -> c32_n {
#if CHANNELS == 1
    return add_c32(c1, c2);
#else if CHANNELS == 2
    return add_c32_2(c1, c2);
#else if CHANNELS == 3
    return add_c32_3(c1, c2);
#else if CHANNELS == 4
    return add_c32_4(c1, c2);
#endif
}
#endif

fn neg_c32(c: c32) -> c32 {
    return c32(-c.re, -c.im);
}

fn neg_c32_2(c: c32_2) -> c32_2 {
    return c32_2(-c.re, -c.im);
}

fn neg_c32_3(c: c32_3) -> c32_3 {
    return c32_3(-c.re, -c.im);
}

fn neg_c32_4(c: c32_4) -> c32_4 {
    return c32_4(-c.re, -c.im);
}

#ifdef CHANNELS
fn neg_c32_n(c: c32_n) -> c32_n {
#if CHANNELS == 1
    return neg_c32(c);
#else if CHANNELS == 2
    return neg_c32_2(c);
#else if CHANNELS == 3
    return neg_c32_3(c);
#else if CHANNELS == 4
    return neg_c32_4(c);
#endif
}
#endif

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
    return c32_3(real, imag);
}

fn mul_c32_4(c1: c32_4, c2: c32_4) -> c32_4 {
    let real = c1.re * c2.re - c1.im * c2.im;
    let imag = c1.re * c2.im + c1.im * c2.re;
    return c32_4(real, imag);
}

#ifdef CHANNELS
fn mul_c32_n(c1: c32_n, c2: c32_n) -> c32_n {
#if CHANNELS == 1
    return mul_c32(c1, c2);
#else if CHANNELS == 2
    return mul_c32_2(c1, c2);
#else if CHANNELS == 3
    return mul_c32_3(c1, c2);
#else if CHANNELS == 4
    return mul_c32_4(c1, c2);
#endif
}
#endif

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
    return c32_3(real, imag);
}

fn muls_c32_4(c1: c32_4, c2: c32) -> c32_4 {
    let real = c1.re * c2.re - c1.im * c2.im;
    let imag = c1.re * c2.im + c1.im * c2.re;
    return c32_4(real, imag);
}

#ifdef CHANNELS
fn muls_c32_n(c1: c32_n, c2: c32) -> c32_n {
#if CHANNELS == 1
    return muls_c32(c1, c2);
#else if CHANNELS == 2
    return muls_c32_2(c1, c2);
#else if CHANNELS == 3
    return muls_c32_3(c1, c2);
#else if CHANNELS == 4
    return muls_c32_4(c1, c2);
#endif
}
#endif


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

#ifdef CHANNELS
fn fma_c32_n(c1: c32_n, c2: c32, c3: c32_n) -> c32_n {
#if CHANNELS == 1
    return fma_c32(c1, c2, c3);
#else if CHANNELS == 2
    return fma_c32_2(c1, c2, c3);
#else if CHANNELS == 3
    return fma_c32_3(c1, c2, c3);
#else if CHANNELS == 4
    return fma_c32_4(c1, c2, c3);
#endif
}
#endif

fn cis_c32(theta: f32) -> c32 {
    return c32(cos(theta), sin(theta));
}

fn conj_c32(c: c32) -> c32 {
    return c32(c.re, -c.im);
}

fn conj_c32_2(c: c32_2) -> c32_2 {
    return c32_2(c.re, -c.im);
}

fn conj_c32_3(c: c32_3) -> c32_3 {
    return c32_3(c.re, -c.im);
}

fn conj_c32_4(c: c32_4) -> c32_4 {
    return c32_4(c.re, -c.im);
}

#ifdef CHANNELS
fn conj_c32_n(c: c32_n) -> c32_n {
#if CHANNELS == 1
    return conj_c32(c);
#else if CHANNELS == 2
    return conj_c32_2(c);
#else if CHANNELS == 3
    return conj_c32_3(c);
#else if CHANNELS == 4
    return conj_c32_4(c);
#endif
}
#endif
