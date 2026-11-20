//! River channels on a heightfield, on the CPU: where a river runs, the level its banks fall
//! along, how wide and deep its channel is for its discharge, and the bed to carve for it.
//!
//! 1. [`route`]: a priority flood from a source cell. It always grows into the lowest cell next
//!    to what it has reached, so it runs down the valley, fills any hollow on the way up to its
//!    rim and leaves over the lowest point of it. It stops at the map edge or wherever the
//!    caller's test says (the sea, standing water, another river).
//! 2. [`smooth_path`]: the 8-connected cell path, smoothed and resampled every texel.
//! 3. [`long_profile`]: the bank level along the path, made to run downhill. A rise up to a
//!    breach depth is cut through; a higher one is left for the water to fill as a lake.
//! 4. [`rasterize`]: the channel and its banks, per texel, sized by [`ChannelModel`] for the
//!    bankfull discharge along the path, with step-pools on steep reaches, riffle-pools on
//!    gentle ones and point bars on the inside of bends.
//!
//! Grids are row-major, one `f32` per texel, `size` texels wide and high. Positions are in
//! texels with texel centres at `+0.5`. Heights are in the caller's height unit, which is
//! [`ChannelShape::cells_per_height`] texels tall.
//!
//! For heights in metres, as from an elevation model, set `cells_per_height` to
//! `1 / metres_per_texel`; depths and `max_depth` are then metres too. The minimum sizes (a
//! half-width of a texel, steps at least 3 samples apart, [`ChannelModel::min_width`]) stay in
//! texels on purpose: they are what the grid can show. A river comes out the same size in
//! metres at 1 m and 10 m texels (a test checks it), as long as the centreline runs near texel
//! centres. On a coarse grid a centreline along texel edges carves a shallower, wider channel,
//! since no texel sits on the thalweg.
//!
//! The code came from bevy_erosion's river brush and is shared with bevy_fft, which carves
//! rivers into elevation models (`docs/river_ecosystem.md` in bevy_fft).

use glam::{UVec2, Vec2};
use std::{cmp::Reverse, collections::BinaryHeap};

/// Local slope (height per texel, in texels) where step-pools take over from riffle-pools, and
/// where riffle-pools take over from a plain bed.
pub const STEP_POOL_SLOPE: (f32, f32) = (0.02, 0.04);
pub const RIFFLE_POOL_SLOPE: (f32, f32) = (0.001, 0.004);

/// No river at this texel.
pub const NO_RIVER: f32 = 1.0e6;

/// One texel of the river raster: the target height (or [`NO_RIVER`]), how far to blend back
/// toward the ground (0 in the channel, rising to 1 across the banks), the armour (0 to 1) and
/// the depth the armour protects below the carved bed (height units).
pub type RiverCell = [f32; 4];

/// River channels from their discharge: a catchment of A km² carries a bankfull discharge
/// Q = k · A^0.8 m³/s, and a channel for Q is a_W · Q^0.5 m wide and a_H · Q^0.4 m deep
/// (Leopold & Maddock 1953; Parker et al. 2007).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ChannelModel {
    /// k in Q = k · A^0.8: about 1–3 in humid regions, much less in arid ones.
    pub bankfull_k: f32,
    /// a_W in width = a_W · Q^0.5 (m).
    pub width_coef: f32,
    /// a_H in depth = a_H · Q^0.4 (m).
    pub depth_coef: f32,
    /// Narrowest channel (texels) the water can run in. Narrower ones are widened to it and made
    /// shallower so they carry the same flow (Manning conveyance W · H^(5/3)).
    pub min_width: f32,
}

impl Default for ChannelModel {
    fn default() -> Self {
        Self {
            bankfull_k: 1.5,
            width_coef: 4.4,
            depth_coef: 0.3,
            min_width: 3.0,
        }
    }
}

impl ChannelModel {
    /// Bankfull discharge (m³/s) of a catchment of `area` m².
    pub fn discharge(&self, area: f32) -> f32 {
        self.bankfull_k * (area.max(0.0) / 1.0e6).powf(0.8)
    }

    /// Width and depth (m) of the channel for `q` m³/s on texels `metres_per_texel` wide,
    /// widened to [`Self::min_width`] texels.
    pub fn section(&self, q: f32, metres_per_texel: f32) -> (f32, f32) {
        let q = q.max(1e-6);
        let (w, h) = (self.width_coef * q.sqrt(), self.depth_coef * q.powf(0.4));
        let min = self.min_width * metres_per_texel;
        if w < min {
            (min, h * (w / min).powf(0.6))
        } else {
            (w, h)
        }
    }

    /// Half-width (texels) and depth (height units) of the channel for `q` m³/s, with height
    /// units `cells_per_height` texels tall.
    pub fn half_width_and_depth(
        &self,
        q: f32,
        metres_per_texel: f32,
        cells_per_height: f32,
    ) -> (f32, f32) {
        let (w, h) = self.section(q, metres_per_texel);
        (
            0.5 * w / metres_per_texel,
            h / (cells_per_height * metres_per_texel),
        )
    }
}

/// The channel model, the grid's units and the cross-section's shape.
#[derive(Clone, Copy, Debug)]
pub struct ChannelShape {
    pub model: ChannelModel,
    /// Texel size in metres.
    pub metres_per_texel: f32,
    /// Texels per height unit: a height difference of 1 rises as far as this many texels run.
    pub cells_per_height: f32,
    /// Cross-section exponent: 2 is parabolic, larger is boxier.
    pub profile: f32,
    /// Depth (height units) the channel's depth eases toward and never passes.
    pub max_depth: f32,
    /// Width of the graded banks beside the channel (texels).
    pub bank_width: f32,
    /// Height the banks rise per texel away from the channel.
    pub bank_slope: f32,
}

impl ChannelShape {
    /// Half-width (texels) and depth (height units) of the channel for `q` m³/s.
    pub fn section(&self, q: f32) -> (f32, f32) {
        self.model
            .half_width_and_depth(q, self.metres_per_texel, self.cells_per_height)
    }
}

/// A river to rasterize: its centreline and the bank level and discharge along it.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct River {
    /// Centreline, one sample per texel, in texels.
    pub path: Vec<[f32; 2]>,
    /// Bank level (height units) at each path sample, running downhill.
    pub profile: Vec<f32>,
    /// Bankfull discharge (m³/s) at each path sample.
    pub flow: Vec<f32>,
    /// Seeds the step and pool spacing.
    pub seed: u32,
}

/// Cells from `source` (in texels) down to where the river ends: the map edge, or the first
/// cell `ends` accepts, given its index in `heights`. `ends` is not asked about the source's own
/// cell. With neither, the route ends at the lowest cell it reached.
pub fn route(
    heights: &[f32],
    size: [u32; 2],
    source: [f32; 2],
    mut ends: impl FnMut(usize) -> bool,
) -> Vec<[u32; 2]> {
    let size = UVec2::from(size);
    let source = Vec2::from(source);
    let n = (size.x * size.y) as usize;
    if heights.len() != n || size.min_element() < 3 {
        return Vec::new();
    }
    let at = |c: UVec2| (c.y * size.x + c.x) as usize;
    let start = source.floor().as_uvec2().min(size - 1);
    let mut ends = |c: UVec2| {
        let edge = c.x == 0 || c.y == 0 || c.x == size.x - 1 || c.y == size.y - 1;
        edge || ends(at(c))
    };
    let mut parent = vec![u32::MAX; n];
    let mut seen = vec![false; n];
    // Lowest first; ties in the order reached, so flats fill outward evenly.
    let mut open = BinaryHeap::new();
    let mut order = 0u32;
    seen[at(start)] = true;
    open.push(Reverse((
        sort_key(heights[at(start)]),
        order,
        start.x,
        start.y,
    )));
    let mut end = start;
    let mut lowest = f32::INFINITY;
    while let Some(Reverse((_, _, x, y))) = open.pop() {
        let c = UVec2::new(x, y);
        if heights[at(c)] < lowest {
            lowest = heights[at(c)];
            end = c;
        }
        if c != start && ends(c) {
            end = c;
            break;
        }
        for dy in -1i32..=1 {
            for dx in -1i32..=1 {
                let q = c.as_ivec2() + glam::IVec2::new(dx, dy);
                if (dx, dy) == (0, 0)
                    || q.cmplt(glam::IVec2::ZERO).any()
                    || q.cmpge(size.as_ivec2()).any()
                {
                    continue;
                }
                let q = q.as_uvec2();
                if !seen[at(q)] {
                    seen[at(q)] = true;
                    parent[at(q)] = at(c) as u32;
                    order += 1;
                    open.push(Reverse((sort_key(heights[at(q)]), order, q.x, q.y)));
                }
            }
        }
    }
    let mut cells = vec![end.to_array()];
    let mut i = at(end);
    while parent[i] != u32::MAX {
        i = parent[i] as usize;
        cells.push([i as u32 % size.x, i as u32 / size.x]);
    }
    cells.reverse();
    cells
}

/// `h`'s bits rearranged so they sort as the heights do, negative ones included; NaN sorts
/// above every height, so a route goes around holes in the data.
fn sort_key(h: f32) -> u32 {
    if h.is_nan() {
        return u32::MAX;
    }
    let bits = h.to_bits();
    if bits & 0x8000_0000 != 0 {
        !bits
    } else {
        bits | 0x8000_0000
    }
}

/// Texel centres of `cells`, smoothed (twice, over `radius` samples each side, ends kept) and
/// resampled one texel apart.
pub fn smooth_path(cells: &[[u32; 2]], radius: usize) -> Vec<[f32; 2]> {
    let mut points: Vec<Vec2> = cells
        .iter()
        .map(|&c| UVec2::from(c).as_vec2() + 0.5)
        .collect();
    if points.len() < 3 {
        return points.iter().map(|p| p.to_array()).collect();
    }
    for _ in 0..2 {
        let prev = points.clone();
        let last = prev.len() - 1;
        for (i, p) in points.iter_mut().enumerate().take(last).skip(1) {
            let r = radius.min(i).min(last - i);
            let sum: Vec2 = prev[i - r..=i + r].iter().sum();
            *p = sum / (2 * r + 1) as f32;
        }
    }
    let mut out = vec![points[0]];
    let mut carry = 0.0;
    for w in points.windows(2) {
        let (a, b) = (w[0], w[1]);
        let len = a.distance(b);
        let mut t = 1.0 - carry;
        while t <= len {
            out.push(a.lerp(b, t / len));
            t += 1.0;
        }
        carry = len - (t - 1.0);
    }
    if out
        .last()
        .is_some_and(|p| p.distance(*points.last().unwrap()) > 0.25)
    {
        out.push(*points.last().unwrap());
    }
    out.iter().map(|p| p.to_array()).collect()
}

/// Bank level along a path whose ground is `ground`: it never rises, except over a rise higher
/// than `breach_depth`, which it follows (the water fills a lake behind it and spills over).
/// A rise it cuts through gets a steady slope from where it starts to where the ground is back
/// below it.
pub fn long_profile(ground: &[f32], breach_depth: f32) -> Vec<f32> {
    let mut z = ground.to_vec();
    let mut i = 1;
    while i < z.len() {
        let level = z[i - 1];
        if ground[i] <= level {
            z[i] = ground[i];
            i += 1;
            continue;
        }
        let back = (i..ground.len()).find(|&j| ground[j] <= level);
        let peak = ground[i..back.unwrap_or(ground.len())]
            .iter()
            .fold(0.0f32, |m, &g| m.max(g - level));
        match back {
            Some(j) if peak <= breach_depth => {
                let end = ground[j];
                for (k, v) in z.iter_mut().enumerate().take(j + 1).skip(i) {
                    *v = level + (end - level) * (k + 1 - i) as f32 / (j + 1 - i) as f32;
                }
                i = j + 1;
            }
            _ => {
                z[i] = ground[i];
                i += 1;
            }
        }
    }
    z
}

/// `field` at a point in texels, between texel centres.
pub fn bilinear(field: &[f32], size: [u32; 2], p: [f32; 2]) -> f32 {
    let (size, p) = (UVec2::from(size), Vec2::from(p));
    if field.is_empty() || field.len() != (size.x * size.y) as usize {
        return 0.0;
    }
    let q = (p - 0.5).clamp(Vec2::ZERO, (size - 1).as_vec2());
    let i = q.floor().as_uvec2();
    let j = (i + 1).min(size - 1);
    let f = q - q.floor();
    let at = |x: u32, y: u32| field[(y * size.x + x) as usize];
    let a = at(i.x, i.y) + (at(j.x, i.y) - at(i.x, i.y)) * f.x;
    let b = at(i.x, j.y) + (at(j.x, j.y) - at(i.x, j.y)) * f.x;
    a + (b - a) * f.y
}

/// Deterministic 0..1 from a seed and an index.
fn hash01(seed: u32, i: u32) -> f32 {
    let mut x = seed.wrapping_mul(0x9e37_79b9) ^ i.wrapping_mul(0x85eb_ca6b);
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb_352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846c_a68b);
    x ^= x >> 16;
    x as f32 / u32::MAX as f32
}

fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Quintic ramp, as bevy_erosion's Shape Beds grades its banks.
fn s5(x: f32) -> f32 {
    let t = x.clamp(0.0, 1.0);
    t * t * t * (t * (6.0 * t - 15.0) + 10.0)
}

/// Per-sample channel along a river.
struct Section {
    /// Half-width (texels).
    width: f32,
    /// Depth below the bank level on the thalweg, for the discharge (height units).
    depth: f32,
    /// What steps and pools add to it on the thalweg (height units). It fades out toward the
    /// banks, so they do not become walls.
    extra: f32,
    /// How far the step lips bow downstream in the middle of the channel (channel widths).
    bow: f32,
    /// How sharply the channel bends, -1 to 1 (+ turning left). The thalweg moves toward the
    /// outer bank under a steep cut bank, and a gently sloping point bar builds on the inner one.
    bend: f32,
    tangent: Vec2,
}

fn sections(river: &River, path: &[Vec2], shape: &ChannelShape, detail: f32) -> Vec<Section> {
    let n = path.len();
    let soft_cap = |d: f32| {
        let m = shape.max_depth.max(1e-6);
        m * (1.0 - (-d / m).exp())
    };
    let flow = |i: usize| river.flow.get(i).copied().unwrap_or(0.0);
    let width = |i: usize| shape.section(flow(i)).0.max(1.0);
    let base = |i: usize| soft_cap(shape.section(flow(i)).1);
    let tangent = |i: usize| {
        let a = path[i.saturating_sub(1)];
        let b = path[(i + 1).min(n - 1)];
        (b - a).normalize_or(Vec2::X)
    };
    // Local slope of the bank level (texels of height per texel) over a few channel widths.
    let slope = |i: usize| {
        let k = (2.0 * width(i)).round().max(2.0) as usize;
        let (a, b) = (i.saturating_sub(k), (i + k).min(n - 1));
        ((river.profile[a] - river.profile[b]) * shape.cells_per_height / (b - a).max(1) as f32)
            .max(0.0)
    };

    // Steps, as depth added to the channel's (positive, a pool) or taken from it (a lip).
    // Between steps at `s` and `e` the bed is a nearly flat tread ending at the lip `e`; at each
    // lip the bed drops about 0.9 of the fall since the last one into a pool scoured about 0.6
    // of the fall deeper, so a step's height comes to about 1.5 × its spacing × the slope
    // (Abrahams et al. 1995). Steps about 1.5 × the channel depth high then sit
    // H / (1.5 S) apart, 1 to 4 channel widths (Chin 1999): closer on steeper reaches. Each lip
    // sits 0.7 of the channel depth below the banks, so the channel stays continuous and holds
    // its flow, and its pool hangs below that.
    let mut extra = vec![0.0f32; n];
    // How far each step's lip bows downstream in the middle, in channel widths.
    let mut bow = vec![0.4f32; n];
    let mut s = 0usize;
    let mut k = 0u32;
    while s + 1 < n {
        let w = 2.0 * width(s);
        let fall_rate = (slope(s) / shape.cells_per_height).max(1e-6);
        let jitter = 0.6 + 0.8 * hash01(river.seed, k);
        let spacing = (base(s) / fall_rate * jitter).clamp(w, 4.0 * w).max(3.0) as usize;
        let e = (s + spacing).min(n - 1);
        let steep = smoothstep(STEP_POOL_SLOPE.0, STEP_POOL_SLOPE.1, slope(s)) * detail;
        if steep > 0.0 && e > s {
            let fall = (river.profile[s] - river.profile[e]).max(0.0);
            let drop = (0.9 * (0.75 + 0.5 * hash01(river.seed ^ 0x51, k))).min(1.0);
            let reach = (0.25 * spacing as f32).max(1.0);
            let step: Vec<f32> = (s..e)
                .map(|j| {
                    let left = (e - j) as f32 / (e - s) as f32;
                    let tread = river.profile[e] + (1.0 - drop) * fall * left;
                    let pool = 0.6 * fall * (-((j - s) as f32 / reach).powi(2)).exp();
                    (river.profile[j] - tread).max(0.0) + pool
                })
                .collect();
            let lip = step.iter().fold(f32::MAX, |m, &d| m.min(d)) + 0.3 * base(s);
            for (v, d) in extra[s..e].iter_mut().zip(&step) {
                *v += steep * (d - lip);
            }
            bow[s..e].fill(0.2 + 0.5 * hash01(river.seed ^ 0x3c, k));
        }
        s = e;
        k += 1;
    }
    // Riffle-pools: a pool every 5 to 7 channel widths (Leopold et al. 1964), from about 0.1 %
    // slope to where step-pools take over. Each pool reaches about 0.8 of the channel depth
    // deeper and the riffles between are a little shallower, so pools come to about twice the
    // riffles' depth.
    let mut s = 0usize;
    while s + 1 < n {
        let w = width(s);
        let spacing = (2.0 * w * (5.0 + 2.0 * hash01(river.seed ^ 0xa7, k))).max(4.0) as usize;
        let e = (s + spacing).min(n - 1);
        let c = (s + e) / 2;
        let gentle = smoothstep(RIFFLE_POOL_SLOPE.0, RIFFLE_POOL_SLOPE.1, slope(c))
            * (1.0 - smoothstep(STEP_POOL_SLOPE.0, STEP_POOL_SLOPE.1, slope(c)))
            * detail;
        if gentle > 0.0 {
            let pool: Vec<f32> = (s..e)
                .map(|j| {
                    let t = (j as f32 - c as f32) / w;
                    0.8 * base(c) * (-t * t).exp()
                })
                .collect();
            let mean = pool.iter().sum::<f32>() / pool.len() as f32;
            for (v, d) in extra[s..e].iter_mut().zip(&pool) {
                *v += gentle * (d - mean);
            }
        }
        s = e;
        k += 1;
    }

    // A riser falls over a few texels rather than one (a cascade over a boulder line). No pool
    // goes deeper than 3 × the channel, and no lip comes closer than 0.7 of it to the banks.
    let extra: Vec<f32> = (0..n)
        .map(|i| {
            let (a, b) = (i.saturating_sub(1), (i + 1).min(n - 1));
            let mean = extra[a..=b].iter().sum::<f32>() / (b - a + 1) as f32;
            mean.clamp(-0.3 * base(i), 3.0 * base(i))
        })
        .collect();

    (0..n)
        .map(|i| {
            let w = width(i);
            // Curvature from the turn of the tangent over about a channel width each side.
            let r = (w.round() as usize).max(2);
            let (a, b) = (i.saturating_sub(r), (i + r).min(n - 1));
            let turn = tangent(a).perp_dot(tangent(b));
            let curvature = turn / (b - a).max(1) as f32;
            let bend = (curvature * w * 4.0 * detail).clamp(-1.0, 1.0);
            Section {
                // Pools are wider as well as deeper.
                width: w * (1.0 + 0.5 * (extra[i] / base(i).max(1e-6)).clamp(0.0, 1.0)),
                bow: bow[i],
                depth: base(i),
                extra: extra[i],
                bend,
                tangent: tangent(i),
            }
        })
        .collect()
}

/// [`Section::extra`] between samples, at a fractional sample index.
fn sample_extra(secs: &[Section], at: f32) -> f32 {
    let at = at.clamp(0.0, (secs.len() - 1) as f32);
    let i = at.floor() as usize;
    let j = (i + 1).min(secs.len() - 1);
    let f = at - i as f32;
    secs[i].extra + (secs[j].extra - secs[i].extra) * f
}

/// `values` between samples, at a fractional sample index.
fn sample(values: &[f32], at: f32) -> f32 {
    let at = at.clamp(0.0, (values.len() - 1) as f32);
    let i = at.floor() as usize;
    let j = (i + 1).min(values.len() - 1);
    values[i] + (values[j] - values[i]) * (at - i as f32)
}

/// The rivers' beds, banks and armour, one [`RiverCell`] per texel of a `size` map. `detail`
/// scales the steps, pools and point bars (0 to 1); `armour` is written into the channel and
/// fades out across the banks.
pub fn rasterize(
    rivers: &[River],
    shape: &ChannelShape,
    detail: f32,
    armour: f32,
    size: [u32; 2],
) -> Vec<RiverCell> {
    let size = UVec2::from(size);
    let n = (size.x * size.y) as usize;
    let mut cells = vec![[NO_RIVER, 1.0, 0.0, 0.0]; n];
    let bank = shape.bank_width.max(1.0);
    for river in rivers.iter().filter(|r| r.path.len() >= 2) {
        let path: Vec<Vec2> = river.path.iter().map(|&p| Vec2::from(p)).collect();
        let secs = sections(river, &path, shape, detail);
        // Nearest sample per texel, for this river.
        let mut nearest = vec![(f32::INFINITY, u32::MAX); n];
        for (i, (p, sec)) in path.iter().zip(&secs).enumerate() {
            let reach = sec.width + bank + 1.0;
            let lo = (*p - reach).floor().max(Vec2::ZERO).as_uvec2();
            let hi = (*p + reach).ceil().as_uvec2().min(size - 1);
            for y in lo.y..=hi.y {
                for x in lo.x..=hi.x {
                    let d = (Vec2::new(x as f32, y as f32) + 0.5).distance_squared(*p);
                    let j = (y * size.x + x) as usize;
                    if d < nearest[j].0 {
                        nearest[j] = (d, i as u32);
                    }
                }
            }
        }
        for (j, &(_, i)) in nearest.iter().enumerate() {
            if i == u32::MAX {
                continue;
            }
            let (i, sec) = (i as usize, &secs[i as usize]);
            let d =
                Vec2::new((j as u32 % size.x) as f32, (j as u32 / size.x) as f32) + 0.5 - path[i];
            // Past either end of the river.
            if d.dot(sec.tangent).abs() > 1.0 && (i == 0 || i == path.len() - 1) {
                continue;
            }
            // Across the channel from its centreline (+ to the left), and from the thalweg, which
            // a bend moves toward its outer bank. The banks stay at ±width from the centreline:
            // the outer side gets narrower and steep (the cut bank), the inner side wider and
            // shallow (the point bar).
            let centre = sec.tangent.perp_dot(d);
            let thalweg = -0.35 * sec.bend * sec.width;
            let across = centre - thalweg;
            let bar_side = across * sec.bend > 0.0;
            let half = if bar_side {
                sec.width + thalweg.abs()
            } else {
                sec.width - thalweg.abs()
            };
            let x = across.abs() / half;
            // At least about parabolic: a boxier section would stand its banks up as walls
            // wherever a pool deepens the channel.
            let rounded = shape.profile.max(1.2);
            let profile = if bar_side {
                rounded * (1.0 + 2.5 * sec.bend.abs())
            } else {
                rounded
            };
            // Bank level at the texel's own place along the path, not its nearest sample's, so
            // a steep bed does not come out in stripes.
            let level = sample(&river.profile, i as f32 + d.dot(sec.tangent));
            let cell = if x < 1.0 {
                let inner = (1.0 - x * x).max(0.0);
                // Step lips bow downstream in the middle of the channel.
                let along = i as f32 + d.dot(sec.tangent) - sec.bow * sec.width * inner;
                let extra = sample_extra(&secs, along);
                let cut = sec.depth * inner.powf(profile) + extra * inner.powf(2.0);
                [level - cut, 0.0, armour, sec.depth + extra]
            } else {
                let out = centre.abs() - sec.width;
                if out >= bank {
                    continue;
                }
                let blend = s5(out / bank);
                [
                    level + shape.bank_slope * out,
                    blend,
                    armour * (1.0 - blend),
                    sec.depth * (1.0 - blend),
                ]
            };
            let old = cells[j];
            // The channel wins over another river's bank; between two channels, the deeper.
            if cell[1] < old[1] || (cell[1] == old[1] && cell[0] < old[0]) {
                cells[j] = [cell[0], cell[1], cell[2].max(old[2]), cell[3].max(old[3])];
            } else {
                cells[j][2] = old[2].max(cell[2]);
            }
        }
    }
    cells
}

/// `ground` with `cells` from [`rasterize`] carved into it: the bed in the channel, a blend
/// back to the ground across the banks. It only ever lowers the ground, so a bank target above
/// it, or another river's bank, never raises it.
pub fn carve(ground: &[f32], cells: &[RiverCell]) -> Vec<f32> {
    ground
        .iter()
        .zip(cells)
        .map(|(&g, c)| {
            if c[0] >= 0.5 * NO_RIVER {
                g
            } else {
                g.min(c[0] + (g - c[0]) * c[1])
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const SIZE: [u32; 2] = [64, 64];

    fn shape() -> ChannelShape {
        ChannelShape {
            model: ChannelModel::default(),
            metres_per_texel: 10.0,
            cells_per_height: 100.0,
            profile: 0.8,
            max_depth: 0.025,
            bank_width: 4.0,
            bank_slope: 0.0015,
        }
    }

    /// A valley along y = 32 falling toward +x.
    fn valley(x: u32, y: u32) -> f32 {
        0.5 - 0.002 * x as f32 + 0.0005 * (y as f32 - 32.0).abs()
    }

    fn field(f: impl Fn(u32, u32) -> f32) -> Vec<f32> {
        (0..SIZE[0] * SIZE[1])
            .map(|i| f(i % SIZE[0], i / SIZE[0]))
            .collect()
    }

    #[test]
    fn route_runs_down_the_valley_to_the_edge() {
        let h = field(valley);
        let cells = route(&h, SIZE, [5.5, 20.5], |_| false);
        let end = *cells.last().unwrap();
        assert_eq!(end[0], SIZE[0] - 1, "ended at {end:?}");
        // It reaches the valley floor and follows it.
        assert!(cells.iter().any(|c| c[1] == 32));
        assert!((end[1] as i32 - 32).abs() <= 1);
    }

    #[test]
    fn route_spills_out_of_a_hollow() {
        // A pit at x = 20 in the valley: the route crosses it and carries on.
        let h = field(|x, y| valley(x, y) - 0.01 * (-((x as f32 - 20.0).powi(2)) / 8.0).exp());
        let cells = route(&h, SIZE, [5.5, 32.5], |_| false);
        assert_eq!(cells.last().unwrap()[0], SIZE[0] - 1);
    }

    #[test]
    fn route_stops_where_the_caller_says() {
        let h = field(valley);
        let lake = (32 * SIZE[0] + 40) as usize;
        let cells = route(&h, SIZE, [5.5, 32.5], |i| i == lake);
        assert_eq!(*cells.last().unwrap(), [40, 32]);
    }

    #[test]
    fn route_runs_below_sea_level_and_around_holes() {
        // The valley dropped 1 below zero, with a hole of missing data across it at x = 20.
        let h = field(|x, y| {
            if x == 20 && (28..=36).contains(&y) {
                f32::NAN
            } else {
                valley(x, y) - 1.0
            }
        });
        let cells = route(&h, SIZE, [5.5, 32.5], |_| false);
        assert_eq!(cells.last().unwrap()[0], SIZE[0] - 1);
        assert!(
            cells
                .iter()
                .all(|c| !h[(c[1] * SIZE[0] + c[0]) as usize].is_nan())
        );
        let mut keys: Vec<f32> = vec![-2.0, -0.5, -0.0, 0.0, 0.25, 3.0];
        keys.sort_by_key(|&h| sort_key(h));
        assert_eq!(keys, [-2.0, -0.5, -0.0, 0.0, 0.25, 3.0]);
    }

    #[test]
    fn smoothed_path_is_evenly_spaced() {
        let cells: Vec<[u32; 2]> = (0..20).map(|i| [i, i / 2]).collect();
        let path = smooth_path(&cells, 3);
        for w in path.windows(2).take(path.len() - 2) {
            assert!((Vec2::from(w[0]).distance(Vec2::from(w[1])) - 1.0).abs() < 1e-3);
        }
        assert_eq!(path[0], [0.5, 0.5]);
    }

    #[test]
    fn profile_cuts_low_rises_and_keeps_high_ones() {
        let ground = [1.0, 0.9, 0.95, 0.85, 0.8, 1.2, 0.7];
        let z = long_profile(&ground, 0.1);
        assert!(z.windows(2).take(4).all(|w| w[1] <= w[0]), "{z:?}");
        assert!(z.iter().zip(&ground).all(|(z, g)| z <= g));
        // The 0.4 rise at index 5 is higher than the breach depth: the bed follows it.
        assert_eq!(z[5], 1.2);
    }

    #[test]
    fn steep_rivers_get_steps_and_pools() {
        let path: Vec<[f32; 2]> = (0..200).map(|i| [i as f32 + 0.5, 32.5]).collect();
        let points: Vec<Vec2> = path.iter().map(|&p| Vec2::from(p)).collect();
        // 100 km² upstream.
        let q = shape().model.discharge(1e8);
        let steep = |s: f32| River {
            flow: vec![q; path.len()],
            seed: 7,
            path: path.clone(),
            // `s` texels of height per texel.
            profile: (0..200).map(|i| 1.0 - s * i as f32 / 100.0).collect(),
        };
        // Depth the steps and pools add to the plain bed.
        let depths = |r: &River| -> Vec<f32> {
            sections(r, &points, &shape(), 1.0)
                .iter()
                .map(|s| s.extra)
                .collect()
        };
        let base = shape().section(q).1;
        // From the shallowest lip to the deepest pool.
        let spread = |d: &[f32]| {
            d.iter().fold(f32::MIN, |m, &v| m.max(v)) - d.iter().fold(f32::MAX, |m, &v| m.min(v))
        };
        let step_pool = depths(&steep(0.1));
        let plain = depths(&steep(0.001));
        assert!(
            spread(&step_pool) > 0.5 * base,
            "steps vary the depth: {}",
            spread(&step_pool)
        );
        assert!(
            spread(&plain) < 0.1 * base,
            "a gentle bed stays plain: {}",
            spread(&plain)
        );
    }

    #[test]
    fn raster_cuts_a_channel_with_graded_banks() {
        let h = field(valley);
        let cells = route(&h, SIZE, [5.5, 32.5], |_| false);
        let path = smooth_path(&cells, 2);
        let ground: Vec<f32> = path.iter().map(|&p| bilinear(&h, SIZE, p)).collect();
        let q = shape().model.discharge(1e7);
        let river = River {
            flow: vec![q; path.len()],
            profile: long_profile(&ground, 0.02),
            seed: 1,
            path,
        };
        let cells = rasterize(&[river], &shape(), 1.0, 1.0, SIZE);
        let at = |x: u32, y: u32| cells[(y * SIZE[0] + x) as usize];
        // On the thalweg: below the ground, fully armoured, no blend.
        let mid = at(30, 32);
        assert!(mid[0] < valley(30, 32), "{mid:?}");
        assert_eq!(mid[1], 0.0);
        assert_eq!(mid[2], 1.0);
        // On the bank: blending back toward the ground; far away: untouched.
        let side = at(30, 32 + 5);
        assert!(side[1] > 0.0 && side[1] < 1.0, "{side:?}");
        assert_eq!(at(30, 5)[0], NO_RIVER);
    }

    /// A river of 50 m³/s down a 1 % valley in metres, carved at 1 m and at 10 m texels with
    /// heights in metres, comes out the same size in metres: depth on the thalweg and width of
    /// the channel.
    #[test]
    fn a_river_is_the_same_size_at_1_m_and_10_m_texels() {
        let q = 50.0;
        let measure = |metres_per_texel: f32| {
            let span = 600.0;
            let n = (span / metres_per_texel) as u32;
            let shape = ChannelShape {
                model: ChannelModel::default(),
                metres_per_texel,
                cells_per_height: 1.0 / metres_per_texel,
                profile: 2.0,
                max_depth: 100.0,
                bank_width: 1.0,
                bank_slope: 0.0,
            };
            // Falling 1 % along x, flat across.
            // On texel centres: a centreline on a texel edge at 10 m texels puts the deepest texel
            // 5 m off the thalweg, and the channel comes out 20 % shallower and a texel wider.
            let path: Vec<[f32; 2]> = (0..n)
                .map(|i| [i as f32 + 0.5, n as f32 / 2.0 + 0.5])
                .collect();
            let profile: Vec<f32> = path
                .iter()
                .map(|p| 10.0 - 0.01 * p[0] * metres_per_texel)
                .collect();
            let river = River {
                flow: vec![q; path.len()],
                profile: profile.clone(),
                seed: 1,
                path,
            };
            let cells = rasterize(&[river], &shape, 0.0, 0.0, [n, n]);
            let x = n / 2;
            let column: Vec<RiverCell> = (0..n).map(|y| cells[(y * n + x) as usize]).collect();
            let level = 10.0 - 0.01 * (x as f32 + 0.5) * metres_per_texel;
            let depth = column.iter().map(|c| level - c[0]).fold(0.0f32, f32::max);
            let width = column.iter().filter(|c| c[1] == 0.0).count() as f32 * metres_per_texel;
            (depth, width)
        };
        let (fine_depth, fine_width) = measure(1.0);
        let (coarse_depth, coarse_width) = measure(10.0);
        let (w, h) = ChannelModel::default().section(q, 1.0);
        println!(
            "model {w:.1} m x {h:.2} m; 1 m texels {fine_width} m x {fine_depth:.2} m; \
             10 m texels {coarse_width} m x {coarse_depth:.2} m"
        );
        assert!(
            (fine_depth - h).abs() < 0.05 * h,
            "{fine_depth} against {h}"
        );
        assert!(
            (coarse_depth - fine_depth).abs() < 0.1 * h,
            "{coarse_depth} against {fine_depth}"
        );
        assert!((fine_width - w).abs() <= 2.0, "{fine_width} against {w}");
        assert!(
            (coarse_width - fine_width).abs() <= 10.0,
            "{coarse_width} against {fine_width}"
        );
    }

    /// Lip-to-pool steps along the thalweg of a carved river: (H/L)/S and the spacing in channel
    /// widths, against Abrahams et al. (1995) and Chin (1999).
    fn steps(slope: f32, q: f32) -> (f32, f32, usize) {
        let mpt = 1.0;
        let n = 800u32;
        let shape = ChannelShape {
            model: ChannelModel::default(),
            metres_per_texel: mpt,
            cells_per_height: 1.0 / mpt,
            profile: 2.0,
            max_depth: 100.0,
            bank_width: 4.0,
            bank_slope: 0.0,
        };
        let y = 20.5;
        let path: Vec<[f32; 2]> = (0..n).map(|i| [i as f32 + 0.5, y]).collect();
        let level = |x: f32| 100.0 - slope * x;
        let river = River {
            flow: vec![q; path.len()],
            profile: path.iter().map(|p| level(p[0])).collect(),
            seed: 11,
            path,
        };
        let size = [n, 41];
        let ground: Vec<f32> = (0..size[0] * size[1])
            .map(|i| level((i % n) as f32 + 0.5))
            .collect();
        let cells = rasterize(&[river], &shape, 1.0, 0.0, size);
        let bed = carve(&ground, &cells);
        // The thalweg, away from the ends, detrended.
        let row = 20 * n as usize;
        let xs: Vec<usize> = (50..n as usize - 50).collect();
        let z: Vec<f32> = xs.iter().map(|&x| bed[row + x]).collect();
        let resid: Vec<f32> = xs
            .iter()
            .zip(&z)
            .map(|(&x, &b)| b - level(x as f32 + 0.5))
            .collect();
        let width = 2.0 * shape.section(q).0;
        let depth = shape.section(q).1;
        // Lips: highest points of the detrended bed within half a width each side, standing
        // at least a fifth of the channel depth above the lowest point before the next lip.
        let k = (0.5 * width).max(2.0) as usize;
        let lips: Vec<usize> = (k..resid.len() - k)
            .filter(|&i| (i - k..=i + k).all(|j| resid[j] <= resid[i]))
            .collect();
        let mut drops = Vec::new();
        for w in lips.windows(2) {
            let pool = z[w[0]..w[1]].iter().fold(f32::MAX, |m, &v| m.min(v));
            let h = z[w[0]] - pool;
            if h > 0.2 * depth {
                drops.push((h, (w[1] - w[0]) as f32));
            }
        }
        let count = drops.len().max(1) as f32;
        let h = drops.iter().map(|d| d.0).sum::<f32>() / count;
        let l = drops.iter().map(|d| d.1).sum::<f32>() / count;
        (h / l / slope, l / width, drops.len())
    }

    /// Measured on 2026-11-20: (H/L)/S 1.24 at 5 %, 1.22 at 8 % and 0.95 at 12 %, against
    /// Abrahams' 1 to 2; steps 0.9 to 1.2 widths apart, at the low end of Chin's 1 to 4. The
    /// generator's minimum spacing is a channel width, cut to whole samples, so narrow channels
    /// come out a little under it; the steepest reach falls just short of Abrahams' range.
    #[test]
    fn step_pools_have_abrahams_proportions() {
        for (slope, q, low) in [(0.05, 2.0, 1.0), (0.08, 5.0, 1.0), (0.12, 1.0, 0.9)] {
            let (hls, spacing, count) = steps(slope, q);
            println!(
                "slope {slope}, {q} m3/s: {count} steps, (H/L)/S {hls:.2}, {spacing:.1} widths apart"
            );
            assert!(count > 20, "{count} steps at slope {slope}");
            assert!((low..=2.0).contains(&hls), "(H/L)/S {hls} at slope {slope}");
            assert!(
                (0.8..=4.0).contains(&spacing),
                "{spacing} widths at slope {slope}"
            );
        }
    }

    #[test]
    fn carving_only_lowers_the_ground() {
        let h = field(valley);
        let cells = route(&h, SIZE, [5.5, 32.5], |_| false);
        let path = smooth_path(&cells, 2);
        let ground: Vec<f32> = path.iter().map(|&p| bilinear(&h, SIZE, p)).collect();
        let river = River {
            flow: vec![shape().model.discharge(1e7); path.len()],
            profile: long_profile(&ground, 0.02),
            seed: 1,
            path,
        };
        // Steep banks rise above the valley's ground beside the channel.
        let steep = ChannelShape {
            bank_slope: 0.05,
            ..shape()
        };
        let cells = rasterize(&[river], &steep, 1.0, 1.0, SIZE);
        assert!(
            cells
                .iter()
                .zip(&h)
                .any(|(c, g)| c[1] > 0.0 && c[1] < 1.0 && c[0] > *g)
        );
        let bed = carve(&h, &cells);
        assert!(bed.iter().zip(&h).all(|(b, g)| b <= g));
        assert!(bed.iter().zip(&h).any(|(b, g)| b < g));
    }
}
