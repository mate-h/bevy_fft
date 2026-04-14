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
    frictionFactor : f32,
    timestamp : u32,
    borderMask : u32,
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

    let d = mix(
        mix(d00, d01, sx),
        mix(d10, d11, sx),
        sy
    );

    return 0.5 + d / sqrt(2.0);
}

// Rgba32Float so storage read-write works on Metal; only .xy are used (bed, water) or (vx, vy).
@group(0) @binding(0) var bedWaterTexture : texture_storage_2d<rgba32float, read_write>;
@group(0) @binding(1) var flowXTexture : texture_storage_2d<r32float, read_write>;
@group(0) @binding(2) var flowYTexture : texture_storage_2d<r32float, read_write>;
@group(0) @binding(3) var velocityTexture : texture_storage_2d<rgba32float, read_write>;
@group(0) @binding(4) var<storage, read_write> particles : array<Particle>;

@group(1) @binding(0) var<uniform> interactionSettings : InteractionSettings;
@group(1) @binding(1) var<uniform> simulationSettings : SimulationSettings;

@compute @workgroup_size(16, 16)
fn clearBuffers(@builtin(global_invocation_id) id: vec3u) {
    textureStore(bedWaterTexture, id.xy, vec4f(0.0));
    textureStore(flowXTexture, id.xy, vec4f(0.0));
    textureStore(flowYTexture, id.xy, vec4f(0.0));

    if (id.x + 1u == simulationSettings.size.x) {
        textureStore(flowXTexture, id.xy + vec2u(1u, 0u), vec4f(0.0));
    }

    if (id.y + 1u == simulationSettings.size.y) {
        textureStore(flowXTexture, id.xy + vec2u(0u, 1u), vec4f(0.0));
    }
}

@compute @workgroup_size(16, 16)
fn loadPreset(@builtin(global_invocation_id) id: vec3u) {
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
    if (all(s0 == s1)) {
        return length(p - s0);
    }

    let s = s1 - s0;
    let d = p - s0;

    let t = dot(d, s) / dot(s, s);

    if (t < 0.0) {
        return length(d);
    } else if (t > 1.0) {
        return length(p - s1);
    } else {
        return length(d - s * t);
    }
}

@compute @workgroup_size(16, 16)
fn interact(@builtin(global_invocation_id) id: vec3u) {
    let position = vec2f(id.xy) + vec2f(0.5);

    if (interactionSettings.mode >= 1u && interactionSettings.mode <= 4u) {

        let distance = pointToSegmentDistance(position, interactionSettings.oldPosition, interactionSettings.position);

        let delta = pow(128.0, interactionSettings.force) * interactionSettings.force * interactionSettings.dt * smoothstep(interactionSettings.radius, interactionSettings.radius * interactionSettings.force - 1.0, distance);

        var value = textureLoad(bedWaterTexture, id.xy);

        if (interactionSettings.mode == 1u) {
            value.x += 10.0 * delta;
        } else if (interactionSettings.mode == 2u) {
            value.x -= 10.0 * delta;
        } else if (interactionSettings.mode == 3u) {
            value.y += delta;
        } else if (interactionSettings.mode == 4u) {
            value.y -= delta;
        }

        value.x = max(0.0, min(10.0, value.x));
        value.y = max(0.0, value.y);

        textureStore(bedWaterTexture, id.xy, value);
    } else if (interactionSettings.mode == 5u) {
        if (id.x >= 1u && id.y >= 1u) {
            let distance = length(interactionSettings.position - position);
            let distanceFactor = smoothstep(interactionSettings.radius * 1.1, interactionSettings.radius * 0.9, distance);

            let impulse = distanceFactor * (interactionSettings.position - interactionSettings.oldPosition) / interactionSettings.dt * interactionSettings.force;

            var flowX = textureLoad(flowXTexture, id.xy).r;
            var flowY = textureLoad(flowYTexture, id.xy).r;

            flowX += impulse.x;
            flowY += impulse.y;

            textureStore(flowXTexture, id.xy, vec4f(flowX, 0.0, 0.0, 0.0));
            textureStore(flowYTexture, id.xy, vec4f(flowY, 0.0, 0.0, 0.0));
        }
    }
}

fn waterSurfaceAt(p : vec2u) -> f32 {
    let bedWaterSample = textureLoad(bedWaterTexture, p);
    return bedWaterSample.x + bedWaterSample.y;
}

fn borderFlow(borderType : u32, bed : f32) -> f32 {
    if (bed > 1.0) {
        return 0.0;
    }

    if (borderType == 0u) {
        return 0.0;
    } else if (borderType == 1u) {
        return 10.0;
    } else if (borderType == 2u) {
        return -10.0;
    } else {
        return 10.0 * sin(f32(simulationSettings.timestamp) / 30.0);
    }
}

@compute @workgroup_size(16, 16)
fn stepAccelerate(@builtin(global_invocation_id) id: vec3u) {
    let bedWaterSample = textureLoad(bedWaterTexture, id.xy);
    let waterSurfaceBase = bedWaterSample.x + bedWaterSample.y;

    let leftBorder = simulationSettings.borderMask & 3u;
    let rightBorder = (simulationSettings.borderMask >> 2) & 3u;
    let bottomBorder = (simulationSettings.borderMask >> 4) & 3u;
    let topBorder = (simulationSettings.borderMask >> 6) & 3u;

    if (id.x == 0u) {
        textureStore(flowXTexture, id.xy, vec4f(borderFlow(leftBorder, bedWaterSample.x), 0.0, 0.0, 0.0));
    }

    if (id.x + 1u == simulationSettings.size[0]) {
        textureStore(flowXTexture, vec2u(id.x + 1u, id.y), vec4f(-borderFlow(rightBorder, bedWaterSample.x), 0.0, 0.0, 0.0));
    }

    if (id.y == 0u) {
        textureStore(flowYTexture, id.xy, vec4f(borderFlow(bottomBorder, bedWaterSample.x), 0.0, 0.0, 0.0));
    }

    if (id.y + 1u == simulationSettings.size[1]) {
        textureStore(flowYTexture, vec2u(id.x, id.y + 1u), vec4f(-borderFlow(topBorder, bedWaterSample.x), 0.0, 0.0, 0.0));
    }

    if (id.x >= 1u) {
        var flowX = textureLoad(flowXTexture, id.xy).x;
        flowX *= simulationSettings.frictionFactor;
        flowX += simulationSettings.gravity * simulationSettings.dt * (waterSurfaceAt(id.xy - vec2u(1, 0)) - waterSurfaceBase);
        textureStore(flowXTexture, id.xy, vec4f(flowX, 0.0, 0.0, 0.0));
    }

    if (id.y >= 1u) {
        var flowY = textureLoad(flowYTexture, id.xy).x;
        flowY *= simulationSettings.frictionFactor;
        flowY += simulationSettings.gravity * simulationSettings.dt * (waterSurfaceAt(id.xy - vec2u(0, 1)) - waterSurfaceBase);
        textureStore(flowYTexture, id.xy, vec4f(flowY, 0.0, 0.0, 0.0));
    }
}

@compute @workgroup_size(16, 16)
fn stepScale(@builtin(global_invocation_id) id: vec3u) {
    let inFlowX = textureLoad(flowXTexture, id.xy).x;
    let inFlowY = textureLoad(flowYTexture, id.xy).x;
    let outFlowX = textureLoad(flowXTexture, id.xy + vec2u(1, 0)).x;
    let outFlowY = textureLoad(flowYTexture, id.xy + vec2u(0, 1)).x;

    let totalOutflow = 0.0
        + max(0.0, -inFlowX)
        + max(0.0, outFlowX)
        + max(0.0, -inFlowY)
        + max(0.0, outFlowY)
        ;

    let water = textureLoad(bedWaterTexture, id.xy).y;

    let maxOutflow = water * simulationSettings.dx * simulationSettings.dx / simulationSettings.dt;

    if (totalOutflow > maxOutflow && totalOutflow > 0.0) {
        let scale = maxOutflow / totalOutflow;

        if (inFlowX < 0.0) { textureStore(flowXTexture, id.xy, vec4f(inFlowX * scale, 0.0, 0.0, 0.0)); }
        if (inFlowY < 0.0) { textureStore(flowYTexture, id.xy, vec4f(inFlowY * scale, 0.0, 0.0, 0.0)); }
        if (outFlowX > 0.0) { textureStore(flowXTexture, id.xy + vec2u(1, 0), vec4f(outFlowX * scale, 0.0, 0.0, 0.0)); }
        if (outFlowY > 0.0) { textureStore(flowYTexture, id.xy + vec2u(0, 1), vec4f(outFlowY * scale, 0.0, 0.0, 0.0)); }
    }
}

@compute @workgroup_size(16, 16)
fn stepMove(@builtin(global_invocation_id) id: vec3u) {
    let inFlowX = textureLoad(flowXTexture, id.xy).x;
    let inFlowY = textureLoad(flowYTexture, id.xy).x;
    let outFlowX = textureLoad(flowXTexture, id.xy + vec2u(1, 0)).x;
    let outFlowY = textureLoad(flowYTexture, id.xy + vec2u(0, 1)).x;

    let totalFlow = inFlowX + inFlowY - outFlowX - outFlowY;

    var bedWaterSample = textureLoad(bedWaterTexture, id.xy);
    let waterOld = bedWaterSample.y;
    bedWaterSample.y += totalFlow * simulationSettings.dt / simulationSettings.dx / simulationSettings.dx;
    textureStore(bedWaterTexture, id.xy, bedWaterSample);

    let waterAverage = (bedWaterSample.y + waterOld) / 2.0;

    var velocity = vec2f(0.0);

    if (waterAverage > 0.0) {
        velocity = vec2f(inFlowX + outFlowX, inFlowY + outFlowY) / (2.0 * simulationSettings.dx * waterAverage);
    }

    textureStore(velocityTexture, id.xy, vec4f(velocity, 0.0, 0.0));
}

@compute @workgroup_size(64)
fn updateParticles(@builtin(global_invocation_id) id: vec3u) {
    var particle = particles[id.x];

    if (particle.lifetime > 0u) {
        particle.lifetime -= 1u;
    }

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

    let velocity = mix(
        mix(v00, v01, tx),
        mix(v10, v11, tx),
        ty
    );

    particle.position += velocity * simulationSettings.dt * 0.5;

    particles[id.x] = particle;
}
