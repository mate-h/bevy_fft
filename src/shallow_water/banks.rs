//! Checks a river's banks against its settled water.
//!
//! Elevation models record the water surface over rivers, so a baked channel is only as good as
//! the carving and the banks beside it. Run the river to a steady state, read the grid back, and
//! [`check_river_banks`] flags two problems before any shading work hides them: water that left
//! the channel, and stretches of bank whose crest is below the settled surface.

use std::collections::VecDeque;

use bevy::math::{UVec2, Vec2};

use super::RiverSpline;

/// Thresholds for [`check_river_banks`], in meters.
#[derive(Clone, Copy, Debug)]
pub struct RiverBankCheck {
    /// Distance past the channel's half width that still counts as channel, for the floodplain
    /// edge of the carve and the cells the bank crest is searched in.
    pub margin: f32,
    /// Height a bank crest must stand above the settled surface beside it.
    pub freeboard: f32,
    /// Depth below which a cell counts as dry.
    pub wet_depth: f32,
    /// Distances along the river to check, from its first point. Leave out the mouth, where the
    /// river meets open water on purpose.
    pub along: (f32, f32),
}

impl Default for RiverBankCheck {
    fn default() -> Self {
        Self {
            margin: 4.0,
            freeboard: 0.1,
            wet_depth: 0.01,
            along: (0.0, f32::INFINITY),
        }
    }
}

/// A settled grid read back from the GPU: bed and depth per cell in meters, row by row, with row
/// `j` along world +Z.
#[derive(Clone, Copy, Debug)]
pub struct SettledGrid<'a> {
    pub cells: UVec2,
    pub cell_size: f32,
    /// World XZ of the outer corner of cell (0, 0).
    pub corner: Vec2,
    pub bed: &'a [f32],
    pub depth: &'a [f32],
}

impl SettledGrid<'_> {
    fn index(&self, c: UVec2) -> usize {
        (c.y * self.cells.x + c.x) as usize
    }

    fn cell_at(&self, p: Vec2) -> Option<UVec2> {
        let c = ((p - self.corner) / self.cell_size).floor();
        (c.x >= 0.0 && c.y >= 0.0 && c.x < self.cells.x as f32 && c.y < self.cells.y as f32)
            .then(|| c.as_uvec2())
    }

    fn centre(&self, c: UVec2) -> Vec2 {
        self.corner + (c.as_vec2() + 0.5) * self.cell_size
    }
}

/// Water outside the channel, one connected region of wet cells.
#[derive(Clone, Debug)]
pub struct BankLeak {
    pub cells: usize,
    /// Wet area in m².
    pub area: f32,
    pub max_depth: f32,
    /// World XZ of the region's deepest cell.
    pub deepest: Vec2,
    /// Distances along the river over which the region meets the channel. A floodplain that
    /// fills from one breach can meet the channel along much more of its length, so look for the
    /// breach among the low banks in that range.
    pub along: (f32, f32),
}

/// A stretch of bank whose crest is lower than the settled surface plus the freeboard.
#[derive(Clone, Debug)]
pub struct LowBank {
    /// Distances along the river where the stretch starts and ends.
    pub along: (f32, f32),
    /// +1 on the left bank looking downstream, -1 on the right.
    pub side: f32,
    /// World XZ of the lowest crest in the stretch.
    pub at: Vec2,
    /// That crest's height and the settled surface on the centreline beside it.
    pub crest: f32,
    pub surface: f32,
}

#[derive(Clone, Debug, Default)]
pub struct RiverBankReport {
    pub leaks: Vec<BankLeak>,
    pub low_banks: Vec<LowBank>,
}

impl RiverBankReport {
    pub fn is_clean(&self) -> bool {
        self.leaks.is_empty() && self.low_banks.is_empty()
    }
}

/// Flags leaks and low banks along `river` in a settled `grid`, read back from a river run to a
/// steady state.
///
/// The channel is every cell within half the river's width plus [`RiverBankCheck::margin`] of
/// its centreline. A leak is a region of wet cells outside the channel that touches wet channel
/// cells within [`RiverBankCheck::along`]. A low bank is a stretch where, across the centreline
/// from half the width out to the margin, the highest bed stays below the settled surface on the
/// centreline plus [`RiverBankCheck::freeboard`].
pub fn check_river_banks(
    river: &RiverSpline,
    grid: &SettledGrid,
    check: &RiverBankCheck,
) -> RiverBankReport {
    let n = (grid.cells.x * grid.cells.y) as usize;
    assert!(grid.bed.len() == n && grid.depth.len() == n);
    let samples = arc_samples(river, 0.5 * grid.cell_size);
    let half = 0.5 * river.width;
    let reach = half + check.margin;
    let within = |s: f32| s >= check.along.0 && s <= check.along.1;
    let wet = |i: usize| grid.depth[i] > check.wet_depth;

    // Distance along the river of the nearest centreline sample, for every channel cell.
    let mut along = vec![f32::NAN; n];
    let mut nearest = vec![f32::INFINITY; n];
    let r = (reach / grid.cell_size).ceil() as i32 + 1;
    for &(p, s, _) in &samples {
        let Some(c) = grid.cell_at(p) else {
            continue;
        };
        for dj in -r..=r {
            for di in -r..=r {
                let q = c.as_ivec2() + bevy::math::IVec2::new(di, dj);
                if q.x < 0 || q.y < 0 || q.x >= grid.cells.x as i32 || q.y >= grid.cells.y as i32 {
                    continue;
                }
                let q = q.as_uvec2();
                let d2 = (grid.centre(q) - p).length_squared();
                let i = grid.index(q);
                if d2 <= reach * reach && d2 < nearest[i] {
                    nearest[i] = d2;
                    along[i] = s;
                }
            }
        }
    }
    let channel = |i: usize| !along[i].is_nan();

    let mut report = RiverBankReport::default();
    let mut seen = vec![false; n];
    let neighbours = |c: UVec2| {
        [(-1, 0), (1, 0), (0, -1), (0, 1)]
            .into_iter()
            .map(move |(di, dj)| c.as_ivec2() + bevy::math::IVec2::new(di, dj))
            .filter(|q| q.x >= 0 && q.y >= 0)
            .map(|q| q.as_uvec2())
            .filter(|q| q.x < grid.cells.x && q.y < grid.cells.y)
    };
    for start in 0..n {
        if seen[start] || channel(start) || !wet(start) {
            continue;
        }
        let mut queue = VecDeque::from([start]);
        seen[start] = true;
        let (mut cells, mut max_depth, mut deepest, mut contact) = (0, 0.0f32, start, None);
        while let Some(i) = queue.pop_front() {
            cells += 1;
            if grid.depth[i] > max_depth {
                (max_depth, deepest) = (grid.depth[i], i);
            }
            let c = UVec2::new(i as u32 % grid.cells.x, i as u32 / grid.cells.x);
            for q in neighbours(c) {
                let j = grid.index(q);
                if !wet(j) {
                    continue;
                }
                if channel(j) {
                    if within(along[j]) {
                        let a = along[j];
                        contact = Some(
                            contact.map_or((a, a), |(lo, hi): (f32, f32)| (lo.min(a), hi.max(a))),
                        );
                    }
                } else if !seen[j] {
                    seen[j] = true;
                    queue.push_back(j);
                }
            }
        }
        if let Some(along) = contact {
            let c = UVec2::new(deepest as u32 % grid.cells.x, deepest as u32 / grid.cells.x);
            report.leaks.push(BankLeak {
                cells,
                area: cells as f32 * grid.cell_size * grid.cell_size,
                max_depth,
                deepest: grid.centre(c),
                along,
            });
        }
    }

    // Bank crests across each wet centreline sample, merged into stretches per side.
    let step = 0.5 * grid.cell_size;
    for side in [1.0f32, -1.0] {
        let mut run: Option<LowBank> = None;
        for &(p, s, tangent) in &samples {
            let low = (|| {
                if !within(s) {
                    return None;
                }
                let i = grid.index(grid.cell_at(p)?);
                if !wet(i) {
                    return None;
                }
                let surface = grid.bed[i] + grid.depth[i];
                let normal = side * tangent.perp();
                let mut crest = (f32::NEG_INFINITY, p);
                let mut d = half;
                while d <= reach {
                    let q = p + normal * d;
                    let bed = grid.bed[grid.index(grid.cell_at(q)?)];
                    if bed > crest.0 {
                        crest = (bed, q);
                    }
                    d += step;
                }
                (crest.0 < surface + check.freeboard).then_some(LowBank {
                    along: (s, s),
                    side,
                    at: crest.1,
                    crest: crest.0,
                    surface,
                })
            })();
            run = match (run, low) {
                (Some(mut current), Some(low)) => {
                    current.along.1 = s;
                    if low.crest - low.surface < current.crest - current.surface {
                        (current.at, current.crest, current.surface) =
                            (low.at, low.crest, low.surface);
                    }
                    Some(current)
                }
                (None, low) => low,
                (Some(current), None) => {
                    report.low_banks.push(current);
                    None
                }
            };
        }
        report.low_banks.extend(run);
    }
    report
}

/// Points along the spline about `spacing` apart, with the distance along it and the unit
/// tangent.
fn arc_samples(river: &RiverSpline, spacing: f32) -> Vec<(Vec2, f32, Vec2)> {
    let fine = river.sample(32);
    let mut out = Vec::new();
    let mut s = 0.0;
    let mut next = 0.0;
    for pair in fine.windows(2) {
        let (a, b) = (pair[0], pair[1]);
        let len = (b - a).length();
        if len <= 0.0 {
            continue;
        }
        let tangent = (b - a) / len;
        while next <= s + len {
            out.push((a + tangent * (next - s), next, tangent));
            next += spacing;
        }
        s += len;
    }
    out
}

/// A waterfall lip where water falling the drop would outrun the solver's velocity clamp.
#[derive(Clone, Copy, Debug)]
pub struct ClampedFall {
    /// World XZ of the face between the lip cell and the cell below it.
    pub at: Vec2,
    /// Bed drop across the face in meters.
    pub drop: f32,
    /// √(2 g drop), the speed the water reaches at the foot, in m/s.
    pub fall_speed: f32,
}

/// Wet lips in a settled `grid` whose bed drops by more than `slope_cap` cell widths to a
/// neighbour, as the waterfall test finds them, and where free fall down that drop reaches more
/// than `velocity_clamp`, the solver's 0.5 Δx/Δt. Faster water than that is clipped and piles up
/// at the foot of the fall instead of running on. Sorted by the drop, largest first.
pub fn clamped_falls(
    grid: &SettledGrid,
    gravity: f32,
    velocity_clamp: f32,
    slope_cap: f32,
    wet_depth: f32,
) -> Vec<ClampedFall> {
    let mut falls = Vec::new();
    let (nx, ny) = (grid.cells.x, grid.cells.y);
    for j in 0..ny {
        for i in 0..nx {
            let c = UVec2::new(i, j);
            let k = grid.index(c);
            if grid.depth[k] <= wet_depth {
                continue;
            }
            for (di, dj) in [(1i32, 0i32), (-1, 0), (0, 1), (0, -1)] {
                let (a, b) = (i as i32 + di, j as i32 + dj);
                if a < 0 || b < 0 || a >= nx as i32 || b >= ny as i32 {
                    continue;
                }
                let n = grid.index(UVec2::new(a as u32, b as u32));
                let drop = grid.bed[k] - grid.bed[n];
                // The lower cell filled to the lip is a pool, not a fall.
                if drop <= slope_cap * grid.cell_size || grid.bed[n] + grid.depth[n] >= grid.bed[k]
                {
                    continue;
                }
                let fall_speed = (2.0 * gravity * drop).sqrt();
                if fall_speed > velocity_clamp {
                    let face =
                        grid.centre(c) + 0.5 * grid.cell_size * Vec2::new(di as f32, dj as f32);
                    falls.push(ClampedFall {
                        at: face,
                        drop,
                        fall_speed,
                    });
                }
            }
        }
    }
    falls.sort_by(|a, b| b.drop.total_cmp(&a.drop));
    falls
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A straight channel along +x on 1 m cells, 6 m wide with its bed at -1 m between banks
    /// 1.5 m high, a floodplain at -0.6 m behind them, and water 0.8 m deep in it. A notch cuts
    /// the left bank to -0.5 m, and with `spill` water stands on the floodplain behind it.
    fn channel(notch: Option<std::ops::Range<u32>>, spill: bool) -> (Vec<f32>, Vec<f32>) {
        let cells = UVec2::new(64, 32);
        let mut bed = vec![0.0; 64 * 32];
        let mut depth = vec![0.0; 64 * 32];
        for j in 0..cells.y {
            for i in 0..cells.x {
                let k = (j * cells.x + i) as usize;
                let z = j as f32 + 0.5 - 16.0;
                bed[k] = if z.abs() < 3.0 {
                    -1.0
                } else if z.abs() < 6.0 {
                    1.5
                } else {
                    -0.6
                };
                if z.abs() < 3.0 {
                    depth[k] = 0.8;
                }
                if let Some(range) = &notch
                    && range.contains(&i)
                    && (3.0..6.0).contains(&z)
                {
                    bed[k] = -0.5;
                    depth[k] = 0.3;
                }
                if spill
                    && notch.as_ref().is_some_and(|r| r.contains(&i))
                    && (6.0..9.0).contains(&z)
                {
                    depth[k] = 0.1;
                }
            }
        }
        (bed, depth)
    }

    fn river() -> RiverSpline {
        RiverSpline {
            points: vec![Vec2::new(-40.0, 0.0), Vec2::new(40.0, 0.0)],
            width: 6.0,
            discharge: 1.0,
            outflow_slope: 0.0,
        }
    }

    fn report(bed: &[f32], depth: &[f32]) -> RiverBankReport {
        let grid = SettledGrid {
            cells: UVec2::new(64, 32),
            cell_size: 1.0,
            corner: Vec2::new(-32.0, -16.0),
            bed,
            depth,
        };
        check_river_banks(
            &river(),
            &grid,
            &RiverBankCheck {
                margin: 3.0,
                ..Default::default()
            },
        )
    }

    #[test]
    fn sound_banks_pass() {
        let (bed, depth) = channel(None, false);
        let report = report(&bed, &depth);
        assert!(report.is_clean(), "{report:?}");
    }

    #[test]
    fn a_notched_bank_is_low_and_leaks() {
        let (bed, depth) = channel(Some(20..26), true);
        let report = report(&bed, &depth);
        assert_eq!(report.leaks.len(), 1, "{report:?}");
        let leak = &report.leaks[0];
        assert_eq!(leak.cells, 18);
        // Cells 20..26 are world x -12..-6, 28 to 34 m along the river from x = -40.
        assert!(leak.along.0 >= 27.0 && leak.along.1 <= 35.0, "{leak:?}");
        assert_eq!(report.low_banks.len(), 1, "{report:?}");
        let low = &report.low_banks[0];
        assert_eq!(low.side, 1.0);
        assert!(low.along.0 >= 27.5 && low.along.1 <= 34.5, "{low:?}");
        assert!(low.crest < low.surface);
    }

    /// A 3 m drop on 1 m cells: free fall reaches 7.7 m/s, past the clamp of 6.25 m/s at
    /// dt 0.08, and within the clamp of 30 m/s at dt 1/60.
    #[test]
    fn clamped_falls_flags_drops_faster_than_the_clamp() {
        let cells = UVec2::new(8, 1);
        let bed: Vec<f32> = (0..8).map(|i| if i < 4 { 3.0 } else { 0.0 }).collect();
        let depth: Vec<f32> = (0..8).map(|i| if i < 4 { 0.2 } else { 0.1 }).collect();
        let grid = SettledGrid {
            cells,
            cell_size: 1.0,
            corner: Vec2::ZERO,
            bed: &bed,
            depth: &depth,
        };
        let clamp = |dt: f32| 0.5 * 1.0 / dt;
        let falls = clamped_falls(&grid, 9.81, clamp(0.08), 1.5, 0.01);
        assert_eq!(falls.len(), 1, "{falls:?}");
        assert!((falls[0].drop - 3.0).abs() < 1e-6);
        assert!((falls[0].at - Vec2::new(4.0, 0.5)).length() < 1e-6);
        assert!(clamped_falls(&grid, 9.81, clamp(1.0 / 60.0), 1.5, 0.01).is_empty());
    }
}
