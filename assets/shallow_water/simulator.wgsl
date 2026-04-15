const PI = 3.1415926535;

struct InteractionSettings {
    mode : u32,
    radius : f32,
    force : f32,
    dt : f32,
    oldPosition : vec2f,
    position : vec2f,
    preset : u32,
}

struct SimulationSettings {
    size : vec2u,
    dt : f32,
    dx : f32,
    gravity : f32,
    friction_factor : f32,
    timestamp : u32,
    border_mask : u32,
    pml_width : u32,
    flags : u32,
    pml_h_rest : f32,
    vel_clamp_alpha : f32,
    h_avgmax_beta : f32,
    eps_wet : f32,
    pml_lambda_decay : f32,
    pml_lambda_update : f32,
    pml_sigma_max : f32,
    overshoot_alpha : f32,
    overshoot_lambda_edge : f32,
}

struct Particle {
    position : vec2f,
    lifetime : u32,
    alive : u32,
}

struct RNGState {
    state : u32,
}

fn rngHash(state : u32) -> u32 {
    var x = state;
    x = x * 747796405u + 2891336453u;
    let y = ((x >> ((x >> 28u) + 4u)) ^ x) * 277803737u;
    x = (y >> 22u) ^ y;
    return x;
}

fn randomUint(state : ptr<function, RNGState>) -> u32 {
    (*state).state = rngHash((*state).state);
    return (*state).state;
}

fn rngInit(state : ptr<function, RNGState>, seed : u32) {
    (*state).state += seed;
    randomUint(state);
}

fn randomFloat(state : ptr<function, RNGState>) -> f32 {
    return f32(randomUint(state)) / 4294967295.0;
}

fn perlinNoiseGridVector(gridPoint : vec2u, seed : u32) -> vec2f {
    var state = RNGState(0u);
    rngInit(&state, seed);
    rngInit(&state, gridPoint.x);
    rngInit(&state, gridPoint.y);
    let angle = randomFloat(&state) * (2.0 * PI);
    return vec2f(cos(angle), sin(angle));
}

fn perlinNoise(point : vec2f, gridSize : f32, seed : u32) -> f32 {
    let gridPosition = point / gridSize;
    let ix = u32(floor(gridPosition.x));
    let iy = u32(floor(gridPosition.y));
    let tx = gridPosition.x - f32(ix);
    let ty = gridPosition.y - f32(iy);
    let sx = smoothstep(0.0, 1.0, tx);
    let sy = smoothstep(0.0, 1.0, ty);
    let v00 = perlinNoiseGridVector(vec2u(ix + 0u, iy + 0u), seed);
    let v01 = perlinNoiseGridVector(vec2u(ix + 1u, iy + 0u), seed);
    let v10 = perlinNoiseGridVector(vec2u(ix + 0u, iy + 1u), seed);
    let v11 = perlinNoiseGridVector(vec2u(ix + 1u, iy + 1u), seed);
    let d00 = dot(v00, vec2f(tx, ty));
    let d01 = dot(v01, vec2f(tx - 1.0, ty));
    let d10 = dot(v10, vec2f(tx, ty - 1.0));
    let d11 = dot(v11, vec2f(tx - 1.0, ty - 1.0));
    let d = mix(mix(d00, d01, sx), mix(d10, d11, sx), sy);
    return 0.5 + d / sqrt(2.0);
}

@group(0) @binding(0) var bedWaterTexture : texture_storage_2d<rgba32float, read_write>;
@group(0) @binding(1) var faceUTexture : texture_storage_2d<r32float, read_write>;
@group(0) @binding(2) var faceWTexture : texture_storage_2d<r32float, read_write>;
@group(0) @binding(3) var macUTemps : texture_storage_2d<rgba32float, read_write>;
@group(0) @binding(4) var macWTemps : texture_storage_2d<rgba32float, read_write>;
@group(0) @binding(5) var velocityTexture : texture_storage_2d<rgba32float, read_write>;
@group(0) @binding(6) var pmlStateTexture : texture_storage_2d<rgba32float, read_write>;
@group(0) @binding(7) var<storage, read_write> particles : array<Particle>;

@group(1) @binding(0) var<uniform> interactionSettings : InteractionSettings;
@group(1) @binding(1) var<uniform> simulationSettings : SimulationSettings;

fn nx() -> i32 { return i32(simulationSettings.size.x); }
fn ny() -> i32 { return i32(simulationSettings.size.y); }

fn cell_ij_clamped(ic : i32, jc : i32) -> vec2u {
    let sx = simulationSettings.size.x;
    let sy = simulationSettings.size.y;
    return vec2u(u32(clamp(ic, 0, i32(sx) - 1)), u32(clamp(jc, 0, i32(sy) - 1)));
}

fn bed_hw(ic : i32, jc : i32) -> vec2f {
    return textureLoad(bedWaterTexture, cell_ij_clamped(ic, jc)).xy;
}

fn h_at(ic : i32, jc : i32) -> f32 { return bed_hw(ic, jc).y; }
fn H_at(ic : i32, jc : i32) -> f32 { return bed_hw(ic, jc).x; }

fn eta_at(ic : i32, jc : i32) -> f32 {
    let hw = bed_hw(ic, jc);
    return hw.x + hw.y;
}

fn u_face_ij(fi : i32, jc : i32) -> vec2u {
    return vec2u(u32(clamp(fi, 0, nx())), u32(clamp(jc, 0, ny() - 1)));
}

fn w_face_ij(ic : i32, fj : i32) -> vec2u {
    return vec2u(u32(clamp(ic, 0, nx() - 1)), u32(clamp(fj, 0, ny())));
}

fn u_face_at(fi : i32, jc : i32) -> f32 {
    return textureLoad(faceUTexture, u_face_ij(fi, jc)).r;
}

fn u_face_mac0_at(fi : i32, jc : i32) -> f32 {
    return textureLoad(macUTemps, u_face_ij(fi, jc)).x;
}

fn w_face_mac0_at(ic : i32, fj : i32) -> f32 {
    return textureLoad(macWTemps, w_face_ij(ic, fj)).x;
}

fn w_face_at(ic : i32, fj : i32) -> f32 {
    return textureLoad(faceWTexture, w_face_ij(ic, fj)).r;
}

fn sample_u_bilinear(x : f32, y : f32) -> f32 {
    let nyi = ny();
    let nxf = nx() + 1;
    let jf = y - 0.5;
    var j0 = i32(floor(jf));
    let ty = jf - f32(j0);
    j0 = clamp(j0, 0, nyi - 1);
    let j1 = min(j0 + 1, nyi - 1);
    var i0 = i32(floor(x));
    let tx = x - f32(i0);
    i0 = clamp(i0, 0, nxf - 2);
    let i1 = min(i0 + 1, nxf - 1);
    let u00 = u_face_at(i0, j0);
    let u10 = u_face_at(i1, j0);
    let u01 = u_face_at(i0, j1);
    let u11 = u_face_at(i1, j1);
    return mix(mix(u00, u10, tx), mix(u01, u11, tx), ty);
}

fn sample_u_bilinear_mac0(x : f32, y : f32) -> f32 {
    let nyi = ny();
    let nxf = nx() + 1;
    let jf = y - 0.5;
    var j0 = i32(floor(jf));
    let ty = jf - f32(j0);
    j0 = clamp(j0, 0, nyi - 1);
    let j1 = min(j0 + 1, nyi - 1);
    var i0 = i32(floor(x));
    let tx = x - f32(i0);
    i0 = clamp(i0, 0, nxf - 2);
    let i1 = min(i0 + 1, nxf - 1);
    let u00 = u_face_mac0_at(i0, j0);
    let u10 = u_face_mac0_at(i1, j0);
    let u01 = u_face_mac0_at(i0, j1);
    let u11 = u_face_mac0_at(i1, j1);
    return mix(mix(u00, u10, tx), mix(u01, u11, tx), ty);
}

fn sample_w_bilinear(x : f32, y : f32) -> f32 {
    let nxi = nx();
    let nyf = ny() + 1;
    let ifloat = x - 0.5;
    var i0 = i32(floor(ifloat));
    let tx = ifloat - f32(i0);
    i0 = clamp(i0, 0, nxi - 1);
    let i1 = min(i0 + 1, nxi - 1);
    var fj = floor(y);
    let ty = y - fj;
    var j0 = i32(fj);
    j0 = clamp(j0, 0, nyf - 2);
    let j1 = min(j0 + 1, nyf - 1);
    let w00 = w_face_at(i0, j0);
    let w10 = w_face_at(i1, j0);
    let w01 = w_face_at(i0, j1);
    let w11 = w_face_at(i1, j1);
    return mix(mix(w00, w10, tx), mix(w01, w11, tx), ty);
}

fn sample_w_bilinear_mac0(x : f32, y : f32) -> f32 {
    let nxi = nx();
    let nyf = ny() + 1;
    let ifloat = x - 0.5;
    var i0 = i32(floor(ifloat));
    let tx = ifloat - f32(i0);
    i0 = clamp(i0, 0, nxi - 1);
    let i1 = min(i0 + 1, nxi - 1);
    var fj = floor(y);
    let ty = y - fj;
    var j0 = i32(fj);
    j0 = clamp(j0, 0, nyf - 2);
    let j1 = min(j0 + 1, nyf - 1);
    let w00 = w_face_mac0_at(i0, j0);
    let w10 = w_face_mac0_at(i1, j0);
    let w01 = w_face_mac0_at(i0, j1);
    let w11 = w_face_mac0_at(i1, j1);
    return mix(mix(w00, w10, tx), mix(w01, w11, tx), ty);
}

fn sample_mac_velocity_u_advect(x : f32, y : f32) -> vec2f {
    return vec2f(sample_u_bilinear_mac0(x, y), sample_w_bilinear(x, y));
}

fn sample_mac_velocity_w_advect(x : f32, y : f32) -> vec2f {
    return vec2f(sample_u_bilinear(x, y), sample_w_bilinear_mac0(x, y));
}

fn trace_semi_lag(pos : vec2f, v : vec2f, dt : f32) -> vec2f {
    let dx = simulationSettings.dx;
    let p = pos - v * (dt / dx);
    return vec2f(clamp(p.x, 0.0, f32(nx())), clamp(p.y, 0.0, f32(ny())));
}

fn semi_lag_u_from(pos : vec2f, dt_sign : f32) -> f32 {
    let v = sample_mac_velocity_u_advect(pos.x, pos.y);
    let pback = trace_semi_lag(pos, v, simulationSettings.dt * dt_sign);
    return sample_u_bilinear_mac0(pback.x, pback.y);
}

fn semi_lag_w_from(pos : vec2f, dt_sign : f32) -> f32 {
    let v = sample_mac_velocity_w_advect(pos.x, pos.y);
    let pback = trace_semi_lag(pos, v, simulationSettings.dt * dt_sign);
    return sample_w_bilinear_mac0(pback.x, pback.y);
}

fn semi_lag_u_reverse_hat(pos : vec2f) -> f32 {
    let v = sample_mac_velocity_u_advect(pos.x, pos.y);
    let pback = trace_semi_lag(pos, v, -simulationSettings.dt);
    return sample_u_bilinear(pback.x, pback.y);
}

fn semi_lag_w_reverse_hat(pos : vec2f) -> f32 {
    let v = sample_mac_velocity_w_advect(pos.x, pos.y);
    let pback = trace_semi_lag(pos, v, -simulationSettings.dt);
    return sample_w_bilinear(pback.x, pback.y);
}

fn vel_max_clamp() -> f32 {
    return simulationSettings.vel_clamp_alpha * simulationSettings.dx / simulationSettings.dt;
}

fn clamp_face_vel(v : f32) -> f32 {
    let vm = vel_max_clamp();
    return clamp(v, -vm, vm);
}

@compute @workgroup_size(16, 16)
fn clearBuffers(@builtin(global_invocation_id) id: vec3u) {
    let sx = simulationSettings.size.x;
    let sy = simulationSettings.size.y;
    if (id.x < sx && id.y < sy) {
        textureStore(bedWaterTexture, id.xy, vec4f(0.0));
        textureStore(velocityTexture, id.xy, vec4f(0.0));
        textureStore(pmlStateTexture, id.xy, vec4f(0.0));
    }
    let nx1 = sx + 1u;
    if (id.x < nx1 && id.y < sy) {
        textureStore(faceUTexture, id.xy, vec4f(0.0));
        textureStore(macUTemps, id.xy, vec4f(0.0));
    }
    let ny1 = sy + 1u;
    if (id.x < sx && id.y < ny1) {
        textureStore(faceWTexture, id.xy, vec4f(0.0));
        textureStore(macWTemps, id.xy, vec4f(0.0));
    }
}

@compute @workgroup_size(16, 16)
fn loadPreset(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= simulationSettings.size.x || id.y >= simulationSettings.size.y) { return; }
    let position = vec2f(id.xy) + vec2f(0.5);
    let simulationMinSize = f32(min(simulationSettings.size.x, simulationSettings.size.y));
    let baseNoiseGridSize = simulationMinSize / 16.0;
    var bed = 0.0;
    if (interactionSettings.preset == 0u) {
        let noise = 0.75 * perlinNoise(position, baseNoiseGridSize, simulationSettings.timestamp)
            + 0.25 * perlinNoise(position, baseNoiseGridSize / 2.0, simulationSettings.timestamp);
        let center = vec2f(simulationSettings.size) / 2.0;
        let t = noise - 0.5 * length(position - center) / simulationMinSize;
        bed = 10.0 * smoothstep(0.45, 0.55, t);
    } else if (interactionSettings.preset == 1u) {
        let noise = perlinNoise(vec2f(position.x, 0.0), baseNoiseGridSize * 2.0, simulationSettings.timestamp);
        let riverY = mix(0.4, 0.6, noise) * f32(simulationSettings.size.y);
        bed = 10.0 * clamp(8.0 * abs(position.y - riverY) / simulationMinSize - 0.25, 0.0, 1.0);
    } else if (interactionSettings.preset == 2u) {
        let noise = perlinNoise(position, baseNoiseGridSize * 2.0, simulationSettings.timestamp);
        bed = clamp(20.0 * abs(2.0 * noise - 1.0) - 4.0, 0.0, 10.0);
    } else if (interactionSettings.preset == 3u) {
        let noise = perlinNoise(vec2f(position.x, 0.0), baseNoiseGridSize * 2.0, simulationSettings.timestamp);
        bed = clamp(40.0 * (position.y / f32(simulationSettings.size.y) - mix(0.4, 0.6, noise)), 0.0, 10.0);
    }
    textureStore(bedWaterTexture, id.xy, vec4f(bed, 0.0, 0.0, 0.0));
}

fn pointToSegmentDistance(p : vec2f, s0 : vec2f, s1 : vec2f) -> f32 {
    if (all(s0 == s1)) { return length(p - s0); }
    let s = s1 - s0;
    let d = p - s0;
    let t = dot(d, s) / dot(s, s);
    if (t < 0.0) { return length(d); }
    else if (t > 1.0) { return length(p - s1); }
    else { return length(d - s * t); }
}

@compute @workgroup_size(16, 16)
fn interact(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= simulationSettings.size.x || id.y >= simulationSettings.size.y) { return; }
    let position = vec2f(id.xy) + vec2f(0.5);
    if (interactionSettings.mode >= 1u && interactionSettings.mode <= 4u) {
        let distance = pointToSegmentDistance(position, interactionSettings.oldPosition, interactionSettings.position);
        let delta = pow(128.0, interactionSettings.force) * interactionSettings.force * interactionSettings.dt
            * smoothstep(interactionSettings.radius, interactionSettings.radius * interactionSettings.force - 1.0, distance);
        var value = textureLoad(bedWaterTexture, id.xy);
        if (interactionSettings.mode == 1u) { value.x += 10.0 * delta; }
        else if (interactionSettings.mode == 2u) { value.x -= 10.0 * delta; }
        else if (interactionSettings.mode == 3u) { value.y += delta; }
        else if (interactionSettings.mode == 4u) { value.y -= delta; }
        value.x = max(0.0, min(10.0, value.x));
        value.y = max(0.0, value.y);
        textureStore(bedWaterTexture, id.xy, value);
    } else if (interactionSettings.mode == 5u) {
        let distance = length(interactionSettings.position - position);
        let distanceFactor = smoothstep(interactionSettings.radius * 1.1, interactionSettings.radius * 0.9, distance);
        let impulse = distanceFactor * (interactionSettings.position - interactionSettings.oldPosition)
            / interactionSettings.dt * interactionSettings.force;
        if (id.x >= 1u) {
            var u = textureLoad(faceUTexture, id.xy).r;
            u += impulse.x;
            textureStore(faceUTexture, id.xy, vec4f(clamp_face_vel(u), 0.0, 0.0, 0.0));
        }
        if (id.y >= 1u) {
            var wv = textureLoad(faceWTexture, id.xy).r;
            wv += impulse.y;
            textureStore(faceWTexture, id.xy, vec4f(clamp_face_vel(wv), 0.0, 0.0, 0.0));
        }
    }
}

@compute @workgroup_size(16, 16)
fn macU_copy(@builtin(global_invocation_id) id: vec3u) {
    let sx = simulationSettings.size.x + 1u;
    let sy = simulationSettings.size.y;
    if (id.x >= sx || id.y >= sy) { return; }
    let u = textureLoad(faceUTexture, id.xy).r;
    textureStore(macUTemps, id.xy, vec4f(u, 0.0, 0.0, 0.0));
}

@compute @workgroup_size(16, 16)
fn macU_sl_forward(@builtin(global_invocation_id) id: vec3u) {
    let sx = simulationSettings.size.x + 1u;
    let sy = simulationSettings.size.y;
    if (id.x >= sx || id.y >= sy) { return; }
    let fi = i32(id.x);
    let jc = i32(id.y);
    let pos = vec2f(f32(fi), f32(jc) + 0.5);
    let u_hat = semi_lag_u_from(pos, 1.0);
    textureStore(faceUTexture, id.xy, vec4f(clamp_face_vel(u_hat), 0.0, 0.0, 0.0));
}

@compute @workgroup_size(16, 16)
fn macU_sl_reverse(@builtin(global_invocation_id) id: vec3u) {
    let sx = simulationSettings.size.x + 1u;
    let sy = simulationSettings.size.y;
    if (id.x >= sx || id.y >= sy) { return; }
    let fi = i32(id.x);
    let jc = i32(id.y);
    let pos = vec2f(f32(fi), f32(jc) + 0.5);
    let u_tilde = semi_lag_u_reverse_hat(pos);
    var mu = textureLoad(macUTemps, id.xy);
    mu.y = u_tilde;
    textureStore(macUTemps, id.xy, mu);
}

@compute @workgroup_size(16, 16)
fn macU_combine(@builtin(global_invocation_id) id: vec3u) {
    let sx = simulationSettings.size.x + 1u;
    let sy = simulationSettings.size.y;
    if (id.x >= sx || id.y >= sy) { return; }
    let mu = textureLoad(macUTemps, id.xy);
    let u_n = mu.x;
    let u_hat = textureLoad(faceUTexture, id.xy).r;
    let u_tilde = mu.y;
    var u_new = u_hat + 0.5 * (u_n - u_tilde);
    let min_u = min(u_n, u_hat);
    let max_u = max(u_n, u_hat);
    if (u_new < min_u - 1e-5 || u_new > max_u + 1e-5) { u_new = u_hat; }
    textureStore(faceUTexture, id.xy, vec4f(clamp_face_vel(u_new), 0.0, 0.0, 0.0));
}

@compute @workgroup_size(16, 16)
fn macW_copy(@builtin(global_invocation_id) id: vec3u) {
    let sx = simulationSettings.size.x;
    let sy = simulationSettings.size.y + 1u;
    if (id.x >= sx || id.y >= sy) { return; }
    let w = textureLoad(faceWTexture, id.xy).r;
    textureStore(macWTemps, id.xy, vec4f(w, 0.0, 0.0, 0.0));
}

@compute @workgroup_size(16, 16)
fn macW_sl_forward(@builtin(global_invocation_id) id: vec3u) {
    let sx = simulationSettings.size.x;
    let sy = simulationSettings.size.y + 1u;
    if (id.x >= sx || id.y >= sy) { return; }
    let ic = i32(id.x);
    let fj = i32(id.y);
    let pos = vec2f(f32(ic) + 0.5, f32(fj));
    let w_hat = semi_lag_w_from(pos, 1.0);
    textureStore(faceWTexture, id.xy, vec4f(clamp_face_vel(w_hat), 0.0, 0.0, 0.0));
}

@compute @workgroup_size(16, 16)
fn macW_sl_reverse(@builtin(global_invocation_id) id: vec3u) {
    let sx = simulationSettings.size.x;
    let sy = simulationSettings.size.y + 1u;
    if (id.x >= sx || id.y >= sy) { return; }
    let ic = i32(id.x);
    let fj = i32(id.y);
    let pos = vec2f(f32(ic) + 0.5, f32(fj));
    let w_tilde = semi_lag_w_reverse_hat(pos);
    var mw = textureLoad(macWTemps, id.xy);
    mw.y = w_tilde;
    textureStore(macWTemps, id.xy, mw);
}

@compute @workgroup_size(16, 16)
fn macW_combine(@builtin(global_invocation_id) id: vec3u) {
    let sx = simulationSettings.size.x;
    let sy = simulationSettings.size.y + 1u;
    if (id.x >= sx || id.y >= sy) { return; }
    let mw = textureLoad(macWTemps, id.xy);
    let w_n = mw.x;
    let w_hat = textureLoad(faceWTexture, id.xy).r;
    let w_tilde = mw.y;
    var w_new = w_hat + 0.5 * (w_n - w_tilde);
    let min_w = min(w_n, w_hat);
    let max_w = max(w_n, w_hat);
    if (w_new < min_w - 1e-5 || w_new > max_w + 1e-5) { w_new = w_hat; }
    textureStore(faceWTexture, id.xy, vec4f(clamp_face_vel(w_new), 0.0, 0.0, 0.0));
}

fn h_adj_cell(ic : i32, jc : i32) -> f32 {
    let g = simulationSettings.gravity;
    let dt = simulationSettings.dt;
    let dx = simulationSettings.dx;
    let beta = simulationSettings.h_avgmax_beta;
    let h_avgmax = beta * dx / (g * dt);
    let h_e = h_at(ic + 1, jc);
    let h_w = h_at(ic - 1, jc);
    let h_n = h_at(ic, jc + 1);
    let h_s = h_at(ic, jc - 1);
    let avg = 0.25 * (h_e + h_w + h_n + h_s);
    return max(0.0, avg - h_avgmax);
}

fn upwind_h_flux(h_up : f32, h_dn : f32, u_face : f32, h_adj_up : f32, h_adj_dn : f32) -> f32 {
    let hu = select(h_dn - h_adj_dn, h_up - h_adj_up, u_face > 0.0);
    return u_face * max(0.0, hu);
}

@compute @workgroup_size(16, 16)
fn integrateHeight(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= simulationSettings.size.x || id.y >= simulationSettings.size.y) { return; }
    let i = i32(id.x);
    let j = i32(id.y);
    let dx = simulationSettings.dx;
    let dt = simulationSettings.dt;
    let uL = u_face_at(i, j);
    let uR = u_face_at(i + 1, j);
    let wB = w_face_at(i, j);
    let wT = w_face_at(i, j + 1);
    let aj = h_adj_cell(i, j);
    let aj_l = h_adj_cell(i - 1, j);
    let aj_r = h_adj_cell(i + 1, j);
    let aj_b = h_adj_cell(i, j - 1);
    let aj_t = h_adj_cell(i, j + 1);
    let h_c = h_at(i, j);
    let h_l = h_at(i - 1, j);
    let h_r = h_at(i + 1, j);
    let h_d = h_at(i, j - 1);
    let h_u = h_at(i, j + 1);
    let f_e = upwind_h_flux(h_c, h_r, uR, aj, aj_r);
    let f_w = upwind_h_flux(h_l, h_c, uL, aj_l, aj);
    let f_n = upwind_h_flux(h_c, h_u, wT, aj, aj_t);
    let f_s = upwind_h_flux(h_d, h_c, wB, aj_b, aj);
    let div = (f_e - f_w + f_n - f_s);
    var bw = textureLoad(bedWaterTexture, id.xy);
    bw.y += -dt / dx * div;
    bw.y = max(0.0, bw.y);
    textureStore(bedWaterTexture, id.xy, bw);
}

@compute @workgroup_size(16, 16)
fn integrateVelocityU(@builtin(global_invocation_id) id: vec3u) {
    let sx = simulationSettings.size.x + 1u;
    let sy = simulationSettings.size.y;
    if (id.x >= sx || id.y >= sy) { return; }
    let fi = i32(id.x);
    let jc = i32(id.y);
    if (fi < 1 || fi > nx()) { return; }
    let g = simulationSettings.gravity;
    let dx = simulationSettings.dx;
    let dt = simulationSettings.dt;
    let eta_l = eta_at(fi - 1, jc);
    let eta_r = eta_at(fi, jc);
    var u = textureLoad(faceUTexture, id.xy).r;
    u += (-g / dx * (eta_r - eta_l)) * dt;
    u = clamp_face_vel(u);
    textureStore(faceUTexture, id.xy, vec4f(u, 0.0, 0.0, 0.0));
}

@compute @workgroup_size(16, 16)
fn integrateVelocityW(@builtin(global_invocation_id) id: vec3u) {
    let sx = simulationSettings.size.x;
    let sy = simulationSettings.size.y + 1u;
    if (id.x >= sx || id.y >= sy) { return; }
    let ic = i32(id.x);
    let fj = i32(id.y);
    if (fj < 1 || fj > ny()) { return; }
    let g = simulationSettings.gravity;
    let dx = simulationSettings.dx;
    let dt = simulationSettings.dt;
    let eta_d = eta_at(ic, fj - 1);
    let eta_u = eta_at(ic, fj);
    var wv = textureLoad(faceWTexture, id.xy).r;
    wv += (-g / dx * (eta_u - eta_d)) * dt;
    wv = clamp_face_vel(wv);
    textureStore(faceWTexture, id.xy, vec4f(wv, 0.0, 0.0, 0.0));
}

fn border_u_value(borderType : u32, bed : f32) -> f32 {
    if (bed > 1.0) { return 0.0; }
    if (borderType == 0u) { return 0.0; }
    else if (borderType == 1u) { return 3.0; }
    else if (borderType == 2u) { return -3.0; }
    else { return 3.0 * sin(f32(simulationSettings.timestamp) / 30.0); }
}

@compute @workgroup_size(16, 16)
fn applyDomainBoundaries(@builtin(global_invocation_id) id: vec3u) {
    let leftBorder = simulationSettings.border_mask & 3u;
    let rightBorder = (simulationSettings.border_mask >> 2u) & 3u;
    let bottomBorder = (simulationSettings.border_mask >> 4u) & 3u;
    let topBorder = (simulationSettings.border_mask >> 6u) & 3u;
    let sy = simulationSettings.size.y;
    let sx = simulationSettings.size.x;
    if (id.x == 0u && id.y < sy) {
        let bed = textureLoad(bedWaterTexture, vec2u(0u, id.y)).x;
        let v = border_u_value(leftBorder, bed);
        textureStore(faceUTexture, vec2u(0u, id.y), vec4f(clamp_face_vel(v), 0.0, 0.0, 0.0));
    }
    if (id.x == sx && id.y < sy) {
        let bed = textureLoad(bedWaterTexture, vec2u(sx - 1u, id.y)).x;
        let v = border_u_value(rightBorder, bed);
        textureStore(faceUTexture, vec2u(sx, id.y), vec4f(clamp_face_vel(v), 0.0, 0.0, 0.0));
    }
    if (id.y == 0u && id.x < sx) {
        let bed = textureLoad(bedWaterTexture, vec2u(id.x, 0u)).x;
        let v = border_u_value(bottomBorder, bed);
        textureStore(faceWTexture, vec2u(id.x, 0u), vec4f(clamp_face_vel(v), 0.0, 0.0, 0.0));
    }
    if (id.y == sy && id.x < sx) {
        let bed = textureLoad(bedWaterTexture, vec2u(id.x, sy - 1u)).x;
        let v = border_u_value(topBorder, bed);
        textureStore(faceWTexture, vec2u(id.x, sy), vec4f(clamp_face_vel(v), 0.0, 0.0, 0.0));
    }
}

@compute @workgroup_size(16, 16)
fn applyWetDryReflectU(@builtin(global_invocation_id) id: vec3u) {
    let sx = simulationSettings.size.x + 1u;
    let sy = simulationSettings.size.y;
    if (id.x >= sx || id.y >= sy) { return; }
    let fi = i32(id.x);
    let jc = i32(id.y);
    if (fi < 1 || fi > nx()) { return; }
    let eps = simulationSettings.eps_wet;
    let h_l = h_at(fi - 1, jc);
    let h_r = h_at(fi, jc);
    let H_l = H_at(fi - 1, jc);
    let H_r = H_at(fi, jc);
    let eta_l = H_l + h_l;
    let eta_r = H_r + h_r;
    var block = false;
    if (h_l <= eps && H_l > eta_r) { block = true; }
    if (h_r <= eps && H_r > eta_l) { block = true; }
    if (block) {
        textureStore(faceUTexture, id.xy, vec4f(0.0, 0.0, 0.0, 0.0));
    }
}

@compute @workgroup_size(16, 16)
fn applyWetDryReflectW(@builtin(global_invocation_id) id: vec3u) {
    let sx = simulationSettings.size.x;
    let sy = simulationSettings.size.y + 1u;
    if (id.x >= sx || id.y >= sy) { return; }
    let ic = i32(id.x);
    let fj = i32(id.y);
    if (fj < 1 || fj > ny()) { return; }
    let eps = simulationSettings.eps_wet;
    let h_d = h_at(ic, fj - 1);
    let h_u = h_at(ic, fj);
    let H_d = H_at(ic, fj - 1);
    let H_u = H_at(ic, fj);
    let eta_d = H_d + h_d;
    let eta_u = H_u + h_u;
    var block = false;
    if (h_d <= eps && H_d > eta_u) { block = true; }
    if (h_u <= eps && H_u > eta_d) { block = true; }
    if (block) {
        textureStore(faceWTexture, id.xy, vec4f(0.0, 0.0, 0.0, 0.0));
    }
}

@compute @workgroup_size(16, 16)
fn applyFrictionU(@builtin(global_invocation_id) id: vec3u) {
    let f = simulationSettings.friction_factor;
    let sx = simulationSettings.size.x + 1u;
    let sy = simulationSettings.size.y;
    if (id.x >= sx || id.y >= sy) { return; }
    var u = textureLoad(faceUTexture, id.xy).r;
    u *= f;
    textureStore(faceUTexture, id.xy, vec4f(u, 0.0, 0.0, 0.0));
}

@compute @workgroup_size(16, 16)
fn applyFrictionW(@builtin(global_invocation_id) id: vec3u) {
    let f = simulationSettings.friction_factor;
    let sx = simulationSettings.size.x;
    let sy = simulationSettings.size.y + 1u;
    if (id.x >= sx || id.y >= sy) { return; }
    var wv = textureLoad(faceWTexture, id.xy).r;
    wv *= f;
    textureStore(faceWTexture, id.xy, vec4f(wv, 0.0, 0.0, 0.0));
}

fn sigma_pml_x(i : i32) -> f32 {
    let w = i32(simulationSettings.pml_width);
    if (w <= 0) { return 0.0; }
    let nxv = nx();
    let ii = clamp(i, 0, nxv - 1);
    let smax = simulationSettings.pml_sigma_max;
    var s = 0.0;
    if (ii < w) {
        let t = (f32(w - ii) / f32(w));
        s = smax * t * t;
    } else if (ii >= nxv - w) {
        let t = (f32(ii - (nxv - w)) / f32(w));
        s = smax * t * t;
    }
    return s;
}

fn sigma_pml_y(j : i32) -> f32 {
    let w = i32(simulationSettings.pml_width);
    if (w <= 0) { return 0.0; }
    let nyv = ny();
    let jj = clamp(j, 0, nyv - 1);
    let smax = simulationSettings.pml_sigma_max;
    var s = 0.0;
    if (jj < w) {
        let t = (f32(w - jj) / f32(w));
        s = smax * t * t;
    } else if (jj >= nyv - w) {
        let t = (f32(jj - (nyv - w)) / f32(w));
        s = smax * t * t;
    }
    return s;
}

fn sigma_u_face(fi : i32) -> f32 {
    let nxv = nx();
    if (fi == 0) {
        return sigma_pml_x(0);
    }
    if (fi == nxv) {
        return sigma_pml_x(nxv - 1);
    }
    return 0.5 * (sigma_pml_x(fi - 1) + sigma_pml_x(fi));
}

fn sigma_w_face(fj : i32) -> f32 {
    let nyv = ny();
    if (fj == 0) {
        return sigma_pml_y(0);
    }
    if (fj == nyv) {
        return sigma_pml_y(nyv - 1);
    }
    return 0.5 * (sigma_pml_y(fj - 1) + sigma_pml_y(fj));
}

@compute @workgroup_size(16, 16)
fn pmlStep(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= simulationSettings.size.x || id.y >= simulationSettings.size.y) { return; }
    if (simulationSettings.pml_width == 0u) {
        textureStore(pmlStateTexture, id.xy, vec4f(0.0));
        return;
    }
    let i = i32(id.x);
    let j = i32(id.y);
    let sx = sigma_pml_x(i);
    let sy = sigma_pml_y(j);
    if (sx <= 0.0 && sy <= 0.0) {
        textureStore(pmlStateTexture, id.xy, vec4f(0.0));
        return;
    }
    let dt = simulationSettings.dt;
    let h_rest = simulationSettings.pml_h_rest;
    let d = simulationSettings.pml_lambda_decay;
    let g = simulationSettings.pml_lambda_update;
    var bw = textureLoad(bedWaterTexture, id.xy);
    let h = bw.y;
    let dh = h - h_rest;
    let uc = 0.5 * (u_face_at(i, j) + u_face_at(i + 1, j));
    let wc = 0.5 * (w_face_at(i, j) + w_face_at(i, j + 1));
    var p = textureLoad(pmlStateTexture, id.xy);
    p.x = p.x * d + g * sx * dh * dt;
    p.y = p.y * d + g * sy * dh * dt;
    p.z = p.z * d + g * sx * uc * dt;
    p.w = p.w * d + g * sy * wc * dt;
    let sponge = -(sx + sy) * dh * dt;
    let mem = -0.25 * g * (p.x + p.y) * dt;
    let vel_coupling = -0.05 * g * (sx * p.z + sy * p.w) * dt;
    bw.y = max(0.0, h + sponge + mem + vel_coupling);
    textureStore(bedWaterTexture, id.xy, bw);
    textureStore(pmlStateTexture, id.xy, p);
}

@compute @workgroup_size(16, 16)
fn pmlDampFaceU(@builtin(global_invocation_id) id: vec3u) {
    if (simulationSettings.pml_width == 0u) { return; }
    let sx = simulationSettings.size.x + 1u;
    let sy = simulationSettings.size.y;
    if (id.x >= sx || id.y >= sy) { return; }
    let fi = i32(id.x);
    let jc = i32(id.y);
    let sig = sigma_u_face(fi);
    if (sig <= 0.0) { return; }
    let dt = simulationSettings.dt;
    var u = textureLoad(faceUTexture, id.xy).r;
    u += -0.5 * sig * u * dt;
    textureStore(faceUTexture, id.xy, vec4f(u, 0.0, 0.0, 0.0));
}

@compute @workgroup_size(16, 16)
fn pmlDampFaceW(@builtin(global_invocation_id) id: vec3u) {
    if (simulationSettings.pml_width == 0u) { return; }
    let sx = simulationSettings.size.x;
    let sy = simulationSettings.size.y + 1u;
    if (id.x >= sx || id.y >= sy) { return; }
    let ic = i32(id.x);
    let fj = i32(id.y);
    let sig = sigma_w_face(fj);
    if (sig <= 0.0) { return; }
    let dt = simulationSettings.dt;
    var wv = textureLoad(faceWTexture, id.xy).r;
    wv += -0.5 * sig * wv * dt;
    textureStore(faceWTexture, id.xy, vec4f(wv, 0.0, 0.0, 0.0));
}

@compute @workgroup_size(16, 16)
fn overshootReduce(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= simulationSettings.size.x || id.y >= simulationSettings.size.y) { return; }
    if ((simulationSettings.flags & 1u) == 0u) { return; }
    let i = i32(id.x);
    let j = i32(id.y);
    let lam = simulationSettings.overshoot_lambda_edge;
    let a_edge = simulationSettings.overshoot_alpha;
    var bw = textureLoad(bedWaterTexture, id.xy);
    let eta_c = bw.x + bw.y;
    let eta_w = eta_at(i - 1, j);
    let eta_e = eta_at(i + 1, j);
    let eta_s = eta_at(i, j - 1);
    let eta_n = eta_at(i, j + 1);
    if (eta_c - eta_w > lam && eta_c > eta_e) {
        let half = max(0.0, 0.5 * (bw.y + h_at(i + 1, j)));
        bw.y += a_edge * (half - bw.y);
    }
    if (eta_c - eta_e > lam && eta_c > eta_w) {
        let half = max(0.0, 0.5 * (bw.y + h_at(i - 1, j)));
        bw.y += a_edge * (half - bw.y);
    }
    if (eta_c - eta_s > lam && eta_c > eta_n) {
        let half = max(0.0, 0.5 * (bw.y + h_at(i, j + 1)));
        bw.y += a_edge * (half - bw.y);
    }
    if (eta_c - eta_n > lam && eta_c > eta_s) {
        let half = max(0.0, 0.5 * (bw.y + h_at(i, j - 1)));
        bw.y += a_edge * (half - bw.y);
    }
    bw.y = max(0.0, bw.y);
    textureStore(bedWaterTexture, id.xy, bw);
}

@compute @workgroup_size(16, 16)
fn reconstructCellVelocity(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= simulationSettings.size.x || id.y >= simulationSettings.size.y) { return; }
    let i = i32(id.x);
    let j = i32(id.y);
    let uc = 0.5 * (u_face_at(i, j) + u_face_at(i + 1, j));
    let wc = 0.5 * (w_face_at(i, j) + w_face_at(i, j + 1));
    textureStore(velocityTexture, id.xy, vec4f(uc, wc, 0.0, 0.0));
}

@compute @workgroup_size(64)
fn updateParticles(@builtin(global_invocation_id) id: vec3u) {
    var particle = particles[id.x];
    if (particle.lifetime > 0u) { particle.lifetime -= 1u; }
    if (particle.alive == 0u
        || particle.lifetime == 0u
        || particle.position[0] < 0.0
        || particle.position[1] < 0.0
        || particle.position[0] >= f32(simulationSettings.size[0])
        || particle.position[1] >= f32(simulationSettings.size[1])
    ) {
        var rngState = RNGState(0u);
        rngInit(&rngState, id.x);
        rngInit(&rngState, simulationSettings.timestamp);
        particle.position.x = randomFloat(&rngState) * f32(simulationSettings.size[0]);
        particle.position.y = randomFloat(&rngState) * f32(simulationSettings.size[1]);
        particle.alive = 1u;
        particle.lifetime = (randomUint(&rngState) % 300u);
    }
    let px = max(0.0, min(f32(simulationSettings.size[0]) - 1.0, particle.position[0] - 0.5));
    let py = max(0.0, min(f32(simulationSettings.size[1]) - 1.0, particle.position[1] - 0.5));
    let ix = u32(floor(px));
    let iy = u32(floor(py));
    let tx = px - f32(ix);
    let ty = py - f32(iy);
    if (textureLoad(bedWaterTexture, vec2u(ix, iy)).y < 1e-3) {
        particle.alive = 0u;
    }
    let v00 = textureLoad(velocityTexture, vec2u(ix, iy)).xy;
    let v01 = textureLoad(velocityTexture, vec2u(ix + 1u, iy)).xy;
    let v10 = textureLoad(velocityTexture, vec2u(ix, iy + 1u)).xy;
    let v11 = textureLoad(velocityTexture, vec2u(ix + 1u, iy + 1u)).xy;
    let velocity = mix(mix(v00, v01, tx), mix(v10, v11, tx), ty);
    particle.position += velocity * simulationSettings.dt * 0.5;
    particles[id.x] = particle;
}
