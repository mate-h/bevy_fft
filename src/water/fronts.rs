//! Breaking wave fronts traced as lines. See [`WaterBreakingFronts`].

use std::collections::{HashMap, HashSet, VecDeque};

use bevy::{
    asset::RenderAssetUsages,
    ecs::system::lifetimeless::Read,
    prelude::*,
    render::{
        RenderApp,
        extract_component::ExtractComponent,
        gpu_readback::{Readback, ReadbackComplete},
        render_asset::RenderAssets,
        render_resource::{binding_types::*, *},
        renderer::{RenderContext, RenderDevice, RenderQueue},
        storage::{GpuShaderBuffer, ShaderBuffer},
        sync_component::SyncComponent,
        texture::GpuImage,
    },
};

use super::heightfield::{WATER_HEIGHTFIELD_SHADER, WaterHeightfield, WaterHeightfieldParams};

/// Bytes of the `FrontCells` header in `heightfield.wesl`.
const HEADER_BYTES: usize = 16;
/// Bytes of one `FrontCell` in `heightfield.wesl`.
const CELL_BYTES: usize = 32;
/// Gravity in m/s² for the wave speed that moves lines between readbacks.
const GRAVITY: f32 = 9.81;
/// Share of a new line's points, or of an old line's, that must lie near the other line for the
/// new line to keep the old line's id.
const MATCH_SHARE: f32 = 0.3;

/// How [`WaterBreakingFronts`] matched the lines of its readbacks, to tell where new ids come from.
/// [`WaterBreakingFronts::tracking`] adds up every readback.
#[derive(Clone, Copy, Debug, Default, PartialEq, Reflect)]
pub struct FrontTrackingStats {
    /// Readbacks counted.
    pub readbacks: u32,
    /// New lines that kept the id of a line in the readback before.
    pub kept: u32,
    /// New lines that took the id of a line missing from the readbacks since, within
    /// [`WaterBreakingFronts::keep_lost`]: a front that flickered out and back.
    pub revived: u32,
    /// New lines whose best match went to another new line, as when a front splits.
    pub split: u32,
    /// New lines near an old line that shared too few points with it.
    pub weak: u32,
    /// New lines with no old line near them.
    pub fresh: u32,
    /// Old lines no new line matched.
    pub lost: u32,
    /// Seconds the lost lines had lived, summed, for their mean age.
    pub lost_age: f32,
}

/// One point of a [`FrontLine`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Reflect)]
pub struct FrontPoint {
    /// World position of the front face, at the water surface.
    pub position: Vec3,
    /// Unit direction on XZ the front travels in, down the surface gradient.
    pub direction: Vec2,
    /// Steepness of the front face, as the magnitude of the surface gradient.
    pub slope: f32,
    /// Depth-averaged water velocity in m/s on XZ.
    pub velocity: Vec2,
    /// Water depth in meters.
    pub depth: f32,
    /// Breaking intensity from 0 to 1, as the LOD writer puts it in the foam lane.
    pub intensity: f32,
}

/// A breaking front traced along its crest.
#[derive(Clone, Debug, Default, PartialEq, Reflect)]
pub struct FrontLine {
    /// Stays the same while the front is tracked from readback to readback.
    pub id: u32,
    /// Seconds since the front was first traced.
    pub age: f32,
    /// The line closes into a loop, as a ring wave does.
    pub closed: bool,
    /// Points about one cell apart, in order along the front.
    pub points: Vec<FrontPoint>,
}

/// Traces the breaking fronts of the [`WaterHeightfield`] on the same entity into [`Self::lines`],
/// after Thuerey et al. 2007.
///
/// Each frame a compute pass marks the grid cells where a front breaks: the CMF10 tests of the
/// LOD writer pass, and the flow runs against the surface gradient, so the front face counts and
/// the back of the wave does not. The cells are read back and linked into [`FrontLine`]s along
/// the wave crest, as Thuerey et al. describe in their section 4:
///
/// 1. The front cells are broadened by [`WaterBreakingFronts::link_cells`] to close small gaps,
///    and the broadened set is split into connected regions.
/// 2. From the strongest unclaimed cell of a region, a line steps one cell at a time along the
///    front's tangent, in both directions. Each step snaps to the steepest front cell across the
///    front, so the line stays on the front face. It ends where it leaves the region, or closes
///    into a loop when it comes back to its start.
/// 3. Cells near a finished line are claimed, and the next unclaimed cell starts another line. A
///    branching front gives two lines.
///
/// Thuerey et al. trace lines once and then advect them. Here the solver runs on the GPU, so the
/// lines are traced again from every readback and matched to the previous lines instead. A line
/// that overlaps one from the last readback, moved forward at the shallow water wave speed √(gh),
/// keeps its [`FrontLine::id`] and [`FrontLine::age`]. Lines overlap when a share of either one's
/// points lies near the other, so a front that grows or shrinks keeps its id. A line that goes
/// missing is carried forward for [`WaterBreakingFronts::keep_lost`] seconds, so a front that
/// flickers below [`WaterBreakingFronts::min_intensity`] for a moment keeps its id too. That gives
/// spray, foam and the breaker curl a stable line to follow while the front itself always comes
/// from the current solver state. [`WaterBreakingFronts::tracking`] counts how lines were matched.
///
/// Results arrive a few frames late, like [`OceanQueries`](crate::ocean::OceanQueries).
/// [`Self::frame`] counts the readbacks.
#[derive(Component, Clone, Debug, Reflect)]
pub struct WaterBreakingFronts {
    /// Cells weaker than this, from 0 to 1, are not front cells.
    pub min_intensity: f32,
    /// Gaps up to this many cells between front cells still link into one line. Thuerey et al.
    /// use 2.
    pub link_cells: f32,
    /// Lines through fewer distinct front cells are dropped.
    pub min_points: usize,
    /// Most front cells read back per frame. Further cells are ignored.
    pub capacity: u32,
    /// Seconds a line missing from the readbacks is still carried forward, so a front that
    /// flickers out for a moment, below [`Self::min_intensity`], keeps its id when it comes back.
    pub keep_lost: f32,
    /// How the readbacks so far were matched.
    pub tracking: FrontTrackingStats,
    /// Fronts from the latest readback.
    pub lines: Vec<FrontLine>,
    /// Readbacks taken so far. It changes when [`Self::lines`] does.
    pub frame: u32,
    /// Front cells found in the latest readback, which can exceed [`Self::capacity`].
    pub cell_count: u32,
    #[reflect(ignore)]
    buffer: Handle<ShaderBuffer>,
    /// Cells the current buffer holds.
    buffer_capacity: u32,
    next_id: u32,
    /// Lines missing from the latest readbacks, carried forward, with the seconds they have been
    /// missing.
    #[reflect(ignore)]
    missing: Vec<(FrontLine, f32)>,
    submitted: u32,
    last_time: Option<f32>,
}

impl Default for WaterBreakingFronts {
    fn default() -> Self {
        Self {
            min_intensity: 0.2,
            link_cells: 2.0,
            min_points: 4,
            capacity: 8192,
            keep_lost: 0.5,
            tracking: FrontTrackingStats::default(),
            lines: Vec::new(),
            frame: 0,
            cell_count: 0,
            buffer: Handle::default(),
            buffer_capacity: 0,
            next_id: 1,
            missing: Vec::new(),
            submitted: 0,
            last_time: None,
        }
    }
}

/// Marks the entity that reads back the front cells of `owner`.
#[derive(Component)]
pub(super) struct FrontsReadback {
    owner: Entity,
}

/// Creates the readback buffer of each new [`WaterBreakingFronts`], and counts frames.
pub(super) fn prepare_breaking_fronts(
    mut commands: Commands,
    mut fronts: Query<(Entity, &mut WaterBreakingFronts)>,
    readbacks: Query<(Entity, &FrontsReadback)>,
    mut buffers: ResMut<Assets<ShaderBuffer>>,
) {
    for (entity, mut fronts) in &mut fronts {
        if fronts.buffer == Handle::default() || fronts.buffer_capacity != fronts.capacity {
            let size = (HEADER_BYTES + fronts.capacity as usize * CELL_BYTES) as u64;
            let buffer = buffers.add(ShaderBuffer::with_size(
                size,
                RenderAssetUsages::RENDER_WORLD,
            ));
            fronts.buffer = buffer.clone();
            fronts.buffer_capacity = fronts.capacity;
            for (readback, link) in &readbacks {
                if link.owner == entity {
                    commands.entity(readback).despawn();
                }
            }
            commands.spawn((
                FrontsReadback { owner: entity },
                Readback::buffer(buffer),
                Name::new("Breaking front readback"),
            ));
        }
        fronts.submitted = fronts.submitted.wrapping_add(1).max(1);
    }
    for (readback, link) in &readbacks {
        if !fronts.contains(link.owner) {
            commands.entity(readback).despawn();
        }
    }
}

pub(super) fn accept_front_cells(
    event: On<ReadbackComplete>,
    readbacks: Query<&FrontsReadback>,
    mut fronts: Query<(&mut WaterBreakingFronts, &WaterHeightfield)>,
    time: Res<Time>,
) {
    let Ok(link) = readbacks.get(event.entity) else {
        return;
    };
    let Ok((mut fronts, heightfield)) = fronts.get_mut(link.owner) else {
        return;
    };
    let Some(header) = event.data.get(..HEADER_BYTES) else {
        return;
    };
    let frame = read_u32(header, 0);
    // A zero frame is a buffer the pass has not written yet.
    if frame == 0 {
        return;
    }
    let found = read_u32(header, 4);
    let body = &event.data[HEADER_BYTES..];
    let count = (found as usize)
        .min(fronts.buffer_capacity as usize)
        .min(body.len() / CELL_BYTES);
    let cells: Vec<FrontCell> = (0..count)
        .map(|i| FrontCell::read(&body[i * CELL_BYTES..(i + 1) * CELL_BYTES]))
        .collect();
    let now = time.elapsed_secs();
    let dt = fronts.last_time.map_or(0.0, |t| (now - t).max(0.0));
    fronts.last_time = Some(now);

    let grid = Grid::of(heightfield);
    let traced = link_front_cells(&cells, fronts.link_cells, fronts.min_points);
    let lines: Vec<(bool, Vec<FrontPoint>)> = traced
        .iter()
        .map(|line| (line.closed, grid.points(line, &cells)))
        .collect();
    let tolerance = (fronts.link_cells + 1.0) * heightfield.cell_size;
    let fronts = &mut *fronts;
    let mut tracker = FrontTracker {
        next_id: fronts.next_id,
        missing: std::mem::take(&mut fronts.missing),
        keep_lost: fronts.keep_lost,
        stats: FrontTrackingStats::default(),
    };
    fronts.lines = tracker.track(&fronts.lines, lines, dt, tolerance);
    fronts.next_id = tracker.next_id;
    fronts.missing = tracker.missing;
    let t = &mut fronts.tracking;
    let s = tracker.stats;
    t.readbacks += 1;
    t.kept += s.kept;
    t.revived += s.revived;
    t.split += s.split;
    t.weak += s.weak;
    t.fresh += s.fresh;
    t.lost += s.lost;
    t.lost_age += s.lost_age;
    fronts.frame = fronts.frame.wrapping_add(1);
    fronts.cell_count = found;
}

fn read_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
}

fn read_f32(bytes: &[u8], offset: usize) -> f32 {
    f32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
}

/// A front cell as `heightfield.wesl` writes it.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct FrontCell {
    pub cell: IVec2,
    pub intensity: f32,
    /// Free-surface gradient.
    pub gradient: Vec2,
    pub velocity: Vec2,
    /// Free-surface height in meters above the source's zero level.
    pub eta: f32,
    pub depth: f32,
}

impl FrontCell {
    fn read(bytes: &[u8]) -> Self {
        let packed = read_u32(bytes, 0);
        let f = |lane: usize| read_f32(bytes, lane * 4);
        Self {
            cell: IVec2::new((packed & 0xffff) as i32, (packed >> 16) as i32),
            intensity: f(1),
            gradient: Vec2::new(f(2), f(3)),
            velocity: Vec2::new(f(4), f(5)),
            eta: f(6),
            depth: f(7),
        }
    }
}

/// Where the grid sits in the world.
struct Grid {
    center: Vec2,
    cells: Vec2,
    cell_size: f32,
    datum: f32,
}

impl Grid {
    fn of(heightfield: &WaterHeightfield) -> Self {
        Self {
            center: heightfield.center,
            cells: heightfield.cells.as_vec2(),
            cell_size: heightfield.cell_size,
            datum: heightfield.datum,
        }
    }

    /// World XZ of grid coordinate `g`, where cell `i` spans `i..i + 1` and rows run along +Z.
    fn world(&self, g: Vec2) -> Vec2 {
        self.center + (g - 0.5 * self.cells) * self.cell_size
    }

    fn points(&self, line: &TracedLine, cells: &[FrontCell]) -> Vec<FrontPoint> {
        line.points
            .iter()
            .map(|&(g, i)| {
                let c = &cells[i];
                let xz = self.world(g);
                FrontPoint {
                    position: Vec3::new(xz.x, self.datum + c.eta, xz.y),
                    direction: (-c.gradient).normalize_or_zero(),
                    slope: c.gradient.length(),
                    velocity: c.velocity,
                    depth: c.depth,
                    intensity: c.intensity,
                }
            })
            .collect()
    }
}

/// A line in grid coordinates. Each point keeps the index of the front cell it takes its data from.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct TracedLine {
    pub points: Vec<(Vec2, usize)>,
    pub closed: bool,
}

/// Links front cells into lines along the front, as [`WaterBreakingFronts`] describes.
pub(crate) fn link_front_cells(
    cells: &[FrontCell],
    link_cells: f32,
    min_points: usize,
) -> Vec<TracedLine> {
    let reach = link_cells.max(1.0);
    let r = reach.ceil() as i32;
    let index: HashMap<IVec2, usize> = cells.iter().enumerate().map(|(i, c)| (c.cell, i)).collect();
    let disc: Vec<IVec2> = (-r..=r)
        .flat_map(|y| (-r..=r).map(move |x| IVec2::new(x, y)))
        .filter(|d| (d.length_squared() as f32) <= reach * reach)
        .collect();

    // The broadened set, split into 8-connected regions.
    let mut region: HashMap<IVec2, u32> = cells
        .iter()
        .flat_map(|c| disc.iter().map(move |d| (c.cell + *d, u32::MAX)))
        .collect();
    let mut regions = 0;
    let keys: Vec<IVec2> = region.keys().copied().collect();
    for start in keys {
        if region[&start] != u32::MAX {
            continue;
        }
        let mut queue = VecDeque::from([start]);
        region.insert(start, regions);
        while let Some(p) = queue.pop_front() {
            for d in [
                IVec2::X,
                IVec2::NEG_X,
                IVec2::Y,
                IVec2::NEG_Y,
                IVec2::ONE,
                IVec2::NEG_ONE,
                IVec2::new(1, -1),
                IVec2::new(-1, 1),
            ] {
                let q = p + d;
                if region.get(&q) == Some(&u32::MAX) {
                    region.insert(q, regions);
                    queue.push_back(q);
                }
            }
        }
        regions += 1;
    }

    let nearest = |g: Vec2| -> Option<usize> {
        let c = g.floor().as_ivec2();
        disc.iter()
            .filter_map(|d| index.get(&(c + *d)).copied())
            .min_by(|&a, &b| {
                let da = (cells[a].cell.as_vec2() + 0.5).distance_squared(g);
                let db = (cells[b].cell.as_vec2() + 0.5).distance_squared(g);
                da.total_cmp(&db)
            })
    };

    let mut seeds: Vec<usize> = (0..cells.len())
        .filter(|&i| cells[i].gradient.length_squared() > 0.0)
        .collect();
    seeds.sort_by(|&a, &b| cells[b].intensity.total_cmp(&cells[a].intensity));
    let mut claimed = vec![false; cells.len()];
    let mut lines = Vec::new();
    for seed in seeds {
        if claimed[seed] {
            continue;
        }
        let in_region = region[&cells[seed].cell];
        let inside = |g: Vec2| region.get(&g.floor().as_ivec2()) == Some(&in_region);
        let start = cells[seed].cell.as_vec2() + 0.5;
        let tangent = cells[seed].gradient.perp().normalize();
        let max_steps = region.len();

        // One half of the line, from the seed along `direction`.
        let walk = |direction: Vec2, line: &[(Vec2, usize)]| -> (Vec<(Vec2, usize)>, bool) {
            let mut out: Vec<(Vec2, usize)> = Vec::new();
            let mut x = start;
            let mut heading = direction;
            let mut gradient = cells[seed].gradient;
            for _ in 0..max_steps {
                if let Some(i) = nearest(x) {
                    gradient = cells[i].gradient;
                }
                let mut t = gradient.perp().normalize_or_zero();
                if t == Vec2::ZERO {
                    break;
                }
                if t.dot(heading) < 0.0 {
                    t = -t;
                }
                let ahead = x + t;
                if !inside(ahead) {
                    break;
                }
                // Snap across the front to its steepest cell (Thuerey et al. eq. 7).
                let across = gradient.normalize();
                let mut next = ahead;
                let mut steepest = -1.0;
                for k in -2 * r..=2 * r {
                    let q = ahead + across * (k as f32 * 0.5);
                    if let Some(&i) = index.get(&q.floor().as_ivec2()) {
                        let s = cells[i].gradient.length();
                        if s > steepest {
                            steepest = s;
                            next = cells[i].cell.as_vec2() + 0.5;
                        }
                    }
                }
                if next.distance(x) < 0.5 {
                    break;
                }
                // Only the first half can close a loop. The second starts next to the seed.
                if line.is_empty() && out.len() as f32 > 2.0 * reach && next.distance(start) < reach
                {
                    return (out, true);
                }
                let revisits = out
                    .iter()
                    .rev()
                    .skip(2)
                    .any(|(p, _)| p.distance(next) < 0.5)
                    || line.iter().skip(2).any(|(p, _)| p.distance(next) < 0.5);
                if revisits {
                    break;
                }
                let Some(i) = nearest(next) else {
                    break;
                };
                out.push((next, i));
                heading = (next - x).normalize();
                x = next;
            }
            // Steps through the broadened gap past the last front cell are not on the front.
            while out
                .last()
                .is_some_and(|(p, _)| !index.contains_key(&p.floor().as_ivec2()))
            {
                out.pop();
            }
            (out, false)
        };

        let (forward, closed) = walk(tangent, &[]);
        let mut points = vec![(start, seed)];
        points.extend(forward);
        if !closed {
            let (backward, _) = walk(-tangent, &points);
            points.splice(0..0, backward.into_iter().rev());
        }

        for &(g, _) in &points {
            let c = g.floor().as_ivec2();
            for d in &disc {
                if let Some(&i) = index.get(&(c + *d)) {
                    claimed[i] = true;
                }
            }
        }
        claimed[seed] = true;
        // Points borrow the data of the nearest front cell, so count the distinct cells. A lone
        // cell would otherwise give a short line through the gap it broadens into.
        let distinct: HashSet<usize> = points.iter().map(|&(_, i)| i).collect();
        if distinct.len() >= min_points.max(2) {
            lines.push(TracedLine { points, closed });
        }
    }
    lines
}

/// Matches the lines of each readback to those of the readbacks before.
pub(crate) struct FrontTracker {
    pub next_id: u32,
    /// Lines missing from the latest readbacks, with the seconds they have been missing.
    pub missing: Vec<(FrontLine, f32)>,
    /// Seconds a missing line is kept.
    pub keep_lost: f32,
    pub stats: FrontTrackingStats,
}

/// `line` moved forward by `dt` at the shallow water wave speed √(gh) of each point.
fn advance(line: &FrontLine, dt: f32) -> FrontLine {
    let mut moved = line.clone();
    for p in &mut moved.points {
        let speed = (GRAVITY * p.depth.max(0.0)).sqrt();
        let step = p.direction * speed * dt;
        p.position += Vec3::new(step.x, 0.0, step.y);
    }
    moved
}

/// Share of `points` within `tolerance` of any of `other`.
fn near_share(points: &[Vec2], other: &[Vec2], tolerance: f32) -> f32 {
    let near = points
        .iter()
        .filter(|p| {
            other
                .iter()
                .any(|q| q.distance_squared(**p) <= tolerance * tolerance)
        })
        .count();
    near as f32 / points.len().max(1) as f32
}

impl FrontTracker {
    /// Gives each new line the id and age of the old line it overlaps most: a line of the readback
    /// before, `previous`, or one missing since, all moved forward by `dt` at the shallow water
    /// wave speed. They overlap when a share of either line's points lies near the other, so a
    /// short piece of a long front keeps its id, and so does a front that grew. Lines without a
    /// match get new ids. Old lines left without a match are kept as missing for
    /// [`Self::keep_lost`] seconds.
    pub fn track(
        &mut self,
        previous: &[FrontLine],
        lines: Vec<(bool, Vec<FrontPoint>)>,
        dt: f32,
        tolerance: f32,
    ) -> Vec<FrontLine> {
        let old: Vec<FrontLine> = previous
            .iter()
            .map(|line| advance(line, dt))
            .chain(self.missing.iter().map(|(line, _)| advance(line, dt)))
            .collect();
        let old_xz: Vec<Vec<Vec2>> = old
            .iter()
            .map(|l| l.points.iter().map(|p| p.position.xz()).collect())
            .collect();
        let mut pairs = Vec::new();
        let mut near_any = vec![false; lines.len()];
        for (n, (_, points)) in lines.iter().enumerate() {
            let new_xz: Vec<Vec2> = points.iter().map(|p| p.position.xz()).collect();
            for (o, old_points) in old_xz.iter().enumerate() {
                let forward = near_share(&new_xz, old_points, tolerance);
                if forward > 0.0 {
                    near_any[n] = true;
                }
                let share = forward.max(near_share(old_points, &new_xz, tolerance));
                if share >= MATCH_SHARE {
                    // Lines of the last readback come before missing ones of the same share.
                    pairs.push((share, o >= previous.len(), n, o));
                }
            }
        }
        pairs.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        let mut matched_new = vec![None; lines.len()];
        let mut had_pair = vec![false; lines.len()];
        let mut used_old = HashSet::new();
        for (_, _, n, o) in pairs {
            had_pair[n] = true;
            if matched_new[n].is_none() && used_old.insert(o) {
                matched_new[n] = Some(o);
            }
        }
        for n in 0..lines.len() {
            match matched_new[n] {
                Some(o) if o < previous.len() => self.stats.kept += 1,
                Some(_) => self.stats.revived += 1,
                None if had_pair[n] => self.stats.split += 1,
                None if near_any[n] => self.stats.weak += 1,
                None => self.stats.fresh += 1,
            }
        }
        // Old lines no new line took: lines of the last readback go missing, and missing ones
        // age until they are dropped.
        let mut missing = Vec::new();
        for (o, line) in old.iter().enumerate() {
            if used_old.contains(&o) {
                continue;
            }
            let (gone, age) = if o < previous.len() {
                self.stats.lost += 1;
                self.stats.lost_age += line.age;
                (0.0, line.age)
            } else {
                (self.missing[o - previous.len()].1, line.age)
            };
            let gone = gone + dt;
            if gone <= self.keep_lost {
                missing.push((
                    FrontLine {
                        age: age + dt,
                        ..line.clone()
                    },
                    gone,
                ));
            }
        }
        let result = lines
            .into_iter()
            .zip(matched_new)
            .map(|((closed, points), o)| match o {
                Some(o) => FrontLine {
                    id: old[o].id,
                    age: old[o].age + dt,
                    closed,
                    points,
                },
                None => {
                    let id = self.next_id;
                    self.next_id = self.next_id.wrapping_add(1).max(1);
                    FrontLine {
                        id,
                        age: 0.0,
                        closed,
                        points,
                    }
                }
            })
            .collect();
        self.missing = missing;
        result
    }
}

/// The render-world copy of a [`WaterBreakingFronts`]: its readback buffer and pass settings.
#[derive(Component, Clone)]
pub struct ExtractedFronts {
    buffer: Handle<ShaderBuffer>,
    frame: u32,
    capacity: u32,
    min_intensity: f32,
}

impl SyncComponent<RenderApp> for WaterBreakingFronts {
    type Target = ExtractedFronts;
}

impl ExtractComponent<RenderApp> for WaterBreakingFronts {
    type QueryData = Read<Self>;
    type QueryFilter = ();
    type Out = ExtractedFronts;

    fn extract_component(
        item: bevy::ecs::query::QueryItem<'_, '_, Self::QueryData>,
    ) -> Option<Self::Out> {
        (item.buffer != Handle::default()).then(|| ExtractedFronts {
            buffer: item.buffer.clone(),
            frame: item.submitted,
            capacity: item.buffer_capacity,
            min_intensity: item.min_intensity,
        })
    }
}

/// Uniform `FrontParams` in `heightfield.wesl`.
#[derive(Clone, Copy, Default, ShaderType)]
struct FrontParams {
    frame: u32,
    capacity: u32,
    min_intensity: f32,
    _pad: f32,
}

#[derive(Resource)]
pub(super) struct WaterFrontsPipeline {
    layout: BindGroupLayoutDescriptor,
    pipeline: CachedComputePipelineId,
}

impl FromWorld for WaterFrontsPipeline {
    fn from_world(world: &mut World) -> Self {
        let pipeline_cache = world.resource::<PipelineCache>();
        let asset_server = world.resource::<AssetServer>();
        let unfiltered = TextureSampleType::Float { filterable: false };
        let entries = BindGroupLayoutEntries::with_indices(
            ShaderStages::COMPUTE,
            (
                (0, uniform_buffer::<WaterHeightfieldParams>(false)),
                (8, texture_2d(unfiltered)),
                (9, texture_2d(unfiltered)),
                (22, storage_buffer_sized(false, None)),
                (23, uniform_buffer::<FrontParams>(false)),
            ),
        );
        let layout = BindGroupLayoutDescriptor::new("water_front_cells_layout", &entries);
        let pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("water_front_cells".into()),
            layout: vec![layout.clone()],
            immediate_size: 0,
            shader: asset_server.load(WATER_HEIGHTFIELD_SHADER),
            shader_defs: vec![],
            entry_point: Some("water_front_cells".into()),
            zero_initialize_workgroup_memory: false,
            constants: vec![],
        });
        Self { layout, pipeline }
    }
}

struct FrontsPass {
    group: BindGroup,
    buffer: Buffer,
    dispatch: UVec2,
}

#[derive(Resource, Default)]
pub(super) struct WaterFrontsPasses {
    passes: Vec<FrontsPass>,
    uniforms: Vec<(
        UniformBuffer<WaterHeightfieldParams>,
        UniformBuffer<FrontParams>,
    )>,
}

#[allow(clippy::too_many_arguments)]
pub(super) fn prepare_water_fronts(
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    pipeline_cache: Res<PipelineCache>,
    pipeline: Res<WaterFrontsPipeline>,
    fronts: Query<(&WaterHeightfield, &ExtractedFronts)>,
    images: Res<RenderAssets<GpuImage>>,
    buffers: Res<RenderAssets<GpuShaderBuffer>>,
    mut state: ResMut<WaterFrontsPasses>,
) {
    let state = &mut *state;
    state.passes.clear();
    let layout = pipeline_cache.get_bind_group_layout(&pipeline.layout);
    for (heightfield, fronts) in &fronts {
        let (Some(bed_water), Some(velocity), Some(output)) = (
            images.get(&heightfield.bed_water),
            images.get(&heightfield.velocity),
            buffers.get(&fronts.buffer),
        ) else {
            continue;
        };
        let index = state.passes.len();
        if state.uniforms.len() <= index {
            state.uniforms.push(default());
        }
        let (grid, params) = &mut state.uniforms[index];
        grid.set(WaterHeightfieldParams::grid(heightfield));
        grid.write_buffer(&device, &queue);
        params.set(FrontParams {
            frame: fronts.frame,
            capacity: fronts.capacity,
            min_intensity: fronts.min_intensity,
            _pad: 0.0,
        });
        params.write_buffer(&device, &queue);
        let (Some(grid), Some(params)) = (grid.binding(), params.binding()) else {
            continue;
        };
        let group = device.create_bind_group(
            "water_front_cells_bind_group",
            &layout,
            &BindGroupEntries::with_indices((
                (0, grid),
                (8, &bed_water.texture_view),
                (9, &velocity.texture_view),
                (22, output.buffer.as_entire_binding()),
                (23, params),
            )),
        );
        state.passes.push(FrontsPass {
            group,
            buffer: output.buffer.clone(),
            dispatch: heightfield.cells.map(|n| n.div_ceil(8)),
        });
    }
}

pub(super) fn run_water_fronts(
    mut ctx: RenderContext,
    pipeline: Res<WaterFrontsPipeline>,
    cache: Res<PipelineCache>,
    state: Res<WaterFrontsPasses>,
) {
    if state.passes.is_empty() {
        return;
    }
    let Some(compute) = cache.get_compute_pipeline(pipeline.pipeline) else {
        return;
    };
    let encoder = ctx.command_encoder();
    for pass in &state.passes {
        encoder.clear_buffer(&pass.buffer, 0, Some(HEADER_BYTES as u64));
        let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("water_front_cells_pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(compute);
        compute_pass.set_bind_group(0, &pass.group, &[]);
        compute_pass.dispatch_workgroups(pass.dispatch.x, pass.dispatch.y, 1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Front cells along a straight front through the origin whose crest runs along `along`,
    /// with the surface falling toward `+normal`.
    fn straight_front(along: Vec2, length: i32, slope: f32) -> Vec<FrontCell> {
        let along = along.normalize();
        let normal = -along.perp();
        let mut cells: Vec<FrontCell> = Vec::new();
        for s in -length * 2..=length * 2 {
            let p = 100.0 + along * (s as f32 * 0.5);
            let cell = p.floor().as_ivec2();
            if cells.iter().all(|c| c.cell != cell) {
                cells.push(FrontCell {
                    cell,
                    // Strongest in the middle, so lines start there and trace both ways.
                    intensity: 1.0 - (s as f32).abs() / (4 * length) as f32,
                    gradient: -normal * slope,
                    velocity: normal,
                    eta: 0.5,
                    depth: 1.0,
                });
            }
        }
        cells
    }

    #[test]
    fn a_straight_front_links_into_one_line() {
        let cells = straight_front(Vec2::Y, 20, 0.3);
        let lines = link_front_cells(&cells, 2.0, 4);
        assert_eq!(lines.len(), 1);
        let line = &lines[0];
        assert!(!line.closed);
        assert!(line.points.len() >= 38, "{} points", line.points.len());
        let ends = (line.points[0].0, line.points.last().unwrap().0);
        assert!((ends.0.y - ends.1.y).abs() > 38.0, "{ends:?}");
        for w in line.points.windows(2) {
            let step = w[0].0.distance(w[1].0);
            assert!((0.5..=1.6).contains(&step), "step {step}");
        }
    }

    #[test]
    fn a_front_two_cells_wide_links_into_one_line() {
        // An axis-aligned face is often two columns of equally steep cells.
        let mut cells = straight_front(Vec2::Y, 20, 0.3);
        cells.extend(straight_front(Vec2::Y, 20, 0.3).into_iter().map(|mut c| {
            c.cell.x += 1;
            c
        }));
        let lines = link_front_cells(&cells, 2.0, 4);
        assert_eq!(
            lines.len(),
            1,
            "{:?}",
            lines.iter().map(|l| l.points.len()).collect::<Vec<_>>()
        );
        assert!(lines[0].points.len() >= 38, "{}", lines[0].points.len());
    }

    #[test]
    fn a_rotated_front_links_into_one_line() {
        // Thuerey et al. rotate their test front by ten degrees.
        let along = Vec2::from_angle(10f32.to_radians()).rotate(Vec2::Y);
        let cells = straight_front(along, 20, 0.3);
        let lines = link_front_cells(&cells, 2.0, 4);
        assert_eq!(lines.len(), 1);
        assert!(lines[0].points.len() >= 35, "{}", lines[0].points.len());
    }

    #[test]
    fn small_gaps_close_and_separate_fronts_stay_apart() {
        let mut cells = straight_front(Vec2::Y, 20, 0.3);
        // A two-cell gap in the middle of the front.
        cells.retain(|c| !(c.cell.y == 99 || c.cell.y == 100));
        // A second front 30 cells away.
        cells.extend(straight_front(Vec2::Y, 10, 0.3).into_iter().map(|mut c| {
            c.cell.x += 30;
            c
        }));
        let lines = link_front_cells(&cells, 2.0, 4);
        assert_eq!(lines.len(), 2, "{lines:?}");
    }

    #[test]
    fn a_ring_front_closes_into_a_loop() {
        let mut cells: Vec<FrontCell> = Vec::new();
        for a in 0..720 {
            let dir = Vec2::from_angle((a as f32 * 0.5).to_radians());
            let cell = (100.0 + dir * 20.0).floor().as_ivec2();
            if cells.iter().all(|c| c.cell != cell) {
                cells.push(FrontCell {
                    cell,
                    intensity: 1.0,
                    // The ring spreads outward, so the surface falls outward.
                    gradient: -dir * 0.3,
                    velocity: dir,
                    eta: 0.5,
                    depth: 1.0,
                });
            }
        }
        let lines = link_front_cells(&cells, 2.0, 4);
        assert_eq!(lines.len(), 1, "{} lines", lines.len());
        assert!(lines[0].closed);
        assert!(lines[0].points.len() > 100, "{}", lines[0].points.len());
    }

    #[test]
    fn isolated_cells_make_no_line() {
        let cells = vec![FrontCell {
            cell: IVec2::new(5, 5),
            intensity: 1.0,
            gradient: Vec2::X,
            velocity: Vec2::NEG_X,
            eta: 0.0,
            depth: 1.0,
        }];
        assert!(link_front_cells(&cells, 2.0, 4).is_empty());
    }

    fn line_at(x: f32, direction: Vec2) -> Vec<FrontPoint> {
        (0..20)
            .map(|z| FrontPoint {
                position: Vec3::new(x, 0.0, z as f32),
                direction,
                depth: 1.0,
                ..default()
            })
            .collect()
    }

    fn tracker(keep_lost: f32) -> FrontTracker {
        FrontTracker {
            next_id: 1,
            missing: Vec::new(),
            keep_lost,
            stats: FrontTrackingStats::default(),
        }
    }

    #[test]
    fn tracking_keeps_ids_of_fronts_that_move_on() {
        let mut t = tracker(0.0);
        let first = t.track(&[], vec![(false, line_at(0.0, Vec2::X))], 0.0, 3.0);
        assert_eq!(first[0].id, 1);
        // √(g·1 m) ≈ 3.1 m/s, so 0.5 s carries the front about 1.6 m.
        let second = t.track(
            &first,
            vec![
                (false, line_at(1.6, Vec2::X)),
                (false, line_at(50.0, Vec2::X)),
            ],
            0.5,
            3.0,
        );
        assert_eq!(second[0].id, 1);
        assert_eq!(second[0].age, 0.5);
        assert_eq!(second[1].id, 2);
        assert_eq!(second[1].age, 0.0);
        assert_eq!((t.stats.kept, t.stats.fresh), (1, 2));
    }

    #[test]
    fn a_front_that_flickers_out_keeps_its_id() {
        let mut t = tracker(0.5);
        let first = t.track(&[], vec![(false, line_at(0.0, Vec2::X))], 0.0, 3.0);
        // Gone for one readback, then back where it would have moved to.
        let gap = t.track(&first, Vec::new(), 0.1, 3.0);
        assert!(gap.is_empty());
        let back = t.track(&gap, vec![(false, line_at(0.62, Vec2::X))], 0.1, 3.0);
        assert_eq!(back[0].id, 1);
        assert_eq!(t.stats.revived, 1);
        // Gone for longer than `keep_lost`, it comes back as a new front.
        let mut t = tracker(0.5);
        let first = t.track(&[], vec![(false, line_at(0.0, Vec2::X))], 0.0, 3.0);
        let mut lines = first;
        for _ in 0..6 {
            lines = t.track(&lines, Vec::new(), 0.1, 3.0);
        }
        let late = t.track(&lines, vec![(false, line_at(2.2, Vec2::X))], 0.1, 3.0);
        assert_eq!(late[0].id, 2);
    }

    #[test]
    fn a_short_piece_of_a_long_front_keeps_its_id() {
        let mut t = tracker(0.0);
        let long: Vec<FrontPoint> = (0..20)
            .map(|z| FrontPoint {
                position: Vec3::new(z as f32, 0.0, 0.0),
                direction: Vec2::Y,
                ..default()
            })
            .collect();
        let first = t.track(&[], vec![(false, long.clone())], 0.0, 3.0);
        // Most of the front stops breaking; the part left lies on the old line, so it keeps the id
        // whichever way the share is taken. A longer line that grew from a short one shares few of
        // its points with it, and keeps the id through the old line's share.
        let grown: Vec<FrontPoint> = (0..80)
            .map(|z| FrontPoint {
                position: Vec3::new(z as f32 - 30.0, 0.0, 0.0),
                direction: Vec2::Y,
                ..default()
            })
            .collect();
        let second = t.track(&first, vec![(false, grown)], 0.0, 3.0);
        assert_eq!(second[0].id, 1);
    }

    #[test]
    fn a_split_front_keeps_its_id_on_one_part() {
        let mut t = tracker(0.0);
        let first = t.track(&[], vec![(false, line_at(0.0, Vec2::X))], 0.0, 3.0);
        let whole = line_at(0.0, Vec2::X);
        let parts = vec![(false, whole[..12].to_vec()), (false, whole[12..].to_vec())];
        let second = t.track(&first, parts, 0.0, 3.0);
        let ids: Vec<u32> = second.iter().map(|l| l.id).collect();
        assert!(ids.contains(&1) && ids.contains(&2), "{ids:?}");
        assert_eq!(t.stats.split, 1);
    }

    #[test]
    fn front_cells_parse_the_shader_layout() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(7u32 | (9u32 << 16)).to_le_bytes());
        for v in [0.75f32, 0.1, -0.2, 1.5, -0.5, 0.3, 0.8] {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let c = FrontCell::read(&bytes);
        assert_eq!(c.cell, IVec2::new(7, 9));
        assert_eq!(c.intensity, 0.75);
        assert_eq!(c.gradient, Vec2::new(0.1, -0.2));
        assert_eq!(c.velocity, Vec2::new(1.5, -0.5));
        assert_eq!((c.eta, c.depth), (0.3, 0.8));
    }
}
