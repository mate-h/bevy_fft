//! Rivers through a shallow water grid: per-face inflow and outflow borders, and the spline that
//! places them.

use bevy::{
    asset::RenderAssetUsages,
    image::Image,
    math::{UVec2, Vec2},
    reflect::Reflect,
    render::render_resource::{Extent3d, TextureDimension, TextureFormat, TextureUsages},
};

use super::ShallowWaterPlacement;

/// One edge of the grid. Left and right run along grid y, bottom and top along grid x.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Reflect)]
pub enum GridEdge {
    /// x = 0.
    Left,
    /// x = cells_x.
    Right,
    /// y = 0.
    Bottom,
    /// y = cells_y.
    Top,
}

impl GridEdge {
    fn row(self) -> usize {
        self as usize
    }
}

/// What one face on the grid's edge does, overriding its edge's
/// [`ShallowWaterBorder`](super::ShallowWaterBorder).
#[derive(Clone, Copy, PartialEq, Debug, Default, Reflect)]
pub enum ShallowWaterBorderFace {
    /// The edge's own border applies.
    #[default]
    Edge,
    /// A fixed discharge flows in, in m²/s per meter of face (cell units squared per second
    /// without a placement). The depth update takes it exactly, whatever the depth beside the
    /// face, so a dry channel fills at the given rate.
    Inflow { discharge_per_width: f32 },
    /// Water leaves at Manning's normal-depth speed `h^(2/3) √S / n` for bed slope `S`, so the
    /// river holds the depth it would have in a long channel downstream. Needs
    /// [`ShallowWaterController::manning_n`](super::ShallowWaterController::manning_n). A zero
    /// slope makes it a zero-gradient edge, which copies the outward speed of the face inside.
    Outflow { slope: f32 },
    /// Closed.
    Wall,
}

impl ShallowWaterBorderFace {
    fn texel(self) -> [f32; 4] {
        match self {
            Self::Edge => [0.0; 4],
            Self::Inflow {
                discharge_per_width,
            } => [1.0, discharge_per_width, 0.0, 0.0],
            Self::Outflow { slope } => [2.0, 0.0, slope.max(0.0), 0.0],
            Self::Wall => [3.0, 0.0, 0.0, 0.0],
        }
    }
}

/// Per-face border overrides for [`ShallowWaterController::border_profile`](super::ShallowWaterController::border_profile).
///
/// The image is `Rgba32Float`, `max(cells_x, cells_y)` wide and four rows high, one row per
/// [`GridEdge`] in declaration order and one texel per face along the edge: face `j` of the left
/// and right edges is beside cell row `j`, face `i` of the bottom and top beside cell column `i`.
/// `.x` is the face kind (0 edge, 1 inflow, 2 outflow, 3 wall), `.y` the inflow's discharge per
/// unit width and `.z` the outflow's bed slope.
#[derive(Clone, Debug)]
pub struct ShallowWaterBorderProfile {
    cells: UVec2,
    faces: Vec<ShallowWaterBorderFace>,
}

impl ShallowWaterBorderProfile {
    /// A profile that leaves every face to its edge.
    pub fn new(cells: UVec2) -> Self {
        let width = cells.x.max(cells.y) as usize;
        Self {
            cells,
            faces: vec![ShallowWaterBorderFace::Edge; 4 * width],
        }
    }

    /// Number of faces along `edge`.
    pub fn edge_len(&self, edge: GridEdge) -> u32 {
        match edge {
            GridEdge::Left | GridEdge::Right => self.cells.y,
            GridEdge::Bottom | GridEdge::Top => self.cells.x,
        }
    }

    fn slot(&self, edge: GridEdge, index: u32) -> usize {
        assert!(index < self.edge_len(edge), "face {index} is off {edge:?}");
        edge.row() * self.cells.x.max(self.cells.y) as usize + index as usize
    }

    pub fn get(&self, edge: GridEdge, index: u32) -> ShallowWaterBorderFace {
        self.faces[self.slot(edge, index)]
    }

    pub fn set(&mut self, edge: GridEdge, index: u32, face: ShallowWaterBorderFace) {
        let slot = self.slot(edge, index);
        self.faces[slot] = face;
    }

    /// Adds a river's inflow where its spline enters the grid and its outflow where it leaves.
    /// Inflow onto a face that already has one adds to its discharge.
    pub fn add_river(&mut self, river: &RiverSpline, placement: &ShallowWaterPlacement) {
        let cells = self.cells;
        let path: Vec<Vec2> = river
            .sample(8)
            .into_iter()
            .map(|p| placement.world_to_cells(p, cells))
            .collect();
        let crossings = border_crossings(&path, cells.as_vec2());
        let width = river.width / placement.cell_size;
        if let Some(entry) = crossings.iter().find(|c| c.entering) {
            let weights = self.channel_weights(entry, width);
            let total: f32 = weights.iter().map(|(_, w)| w).sum();
            for (index, w) in weights {
                let q = river.discharge * w / (total * placement.cell_size);
                let face = match self.get(entry.edge, index) {
                    ShallowWaterBorderFace::Inflow {
                        discharge_per_width,
                    } => discharge_per_width + q,
                    _ => q,
                };
                self.set(
                    entry.edge,
                    index,
                    ShallowWaterBorderFace::Inflow {
                        discharge_per_width: face,
                    },
                );
            }
        }
        if let Some(exit) = crossings.iter().rev().find(|c| !c.entering) {
            for (index, _) in self.channel_weights(exit, width) {
                self.set(
                    exit.edge,
                    index,
                    ShallowWaterBorderFace::Outflow {
                        slope: river.outflow_slope,
                    },
                );
            }
        }
    }

    /// Faces a channel `width` cells wide covers where it crosses the edge, with parabolic
    /// weights that peak on the centreline. The channel is wider along an edge it crosses at a
    /// slant. A channel narrower than a face takes the nearest face.
    fn channel_weights(&self, crossing: &Crossing, width: f32) -> Vec<(u32, f32)> {
        let (along, normal) = match crossing.edge {
            GridEdge::Left | GridEdge::Right => (crossing.at.y, Vec2::X),
            GridEdge::Bottom | GridEdge::Top => (crossing.at.x, Vec2::Y),
        };
        let cos = crossing.direction.normalize_or_zero().dot(normal).abs();
        let half = 0.5 * width / cos.max(0.25);
        let len = self.edge_len(crossing.edge);
        let weights: Vec<(u32, f32)> = (0..len)
            .filter_map(|i| {
                let s = (i as f32 + 0.5 - along) / half;
                (s.abs() < 1.0).then_some((i, 1.0 - s * s))
            })
            .collect();
        if weights.is_empty() {
            let nearest = (along.floor().max(0.0) as u32).min(len - 1);
            return vec![(nearest, 1.0)];
        }
        weights
    }

    /// The profile as an image for [`ShallowWaterController::border_profile`](super::ShallowWaterController::border_profile).
    pub fn image(&self) -> Image {
        let width = self.cells.x.max(self.cells.y);
        let mut image = Image::new_fill(
            Extent3d {
                width,
                height: 4,
                depth_or_array_layers: 1,
            },
            TextureDimension::D2,
            &[0u8; 16],
            TextureFormat::Rgba32Float,
            RenderAssetUsages::default(),
        );
        image.texture_descriptor.usage = TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST;
        image.data = Some(
            self.faces
                .iter()
                .flat_map(|f| f.texel())
                .flat_map(f32::to_le_bytes)
                .collect(),
        );
        image
    }
}

/// A river's centreline through the grid, in world XZ meters, drawn as a Catmull-Rom spline
/// through `points`. It should start upstream outside the grid and end downstream outside it:
/// where it first enters, [`ShallowWaterBorderProfile::add_river`] lets in `discharge`, and where
/// it last leaves, the water flows out at normal depth for `outflow_slope`.
#[derive(Clone, Debug, Reflect)]
pub struct RiverSpline {
    pub points: Vec<Vec2>,
    /// Channel width at the grid's edges in meters. The discharge is spread over it with a
    /// parabolic profile, deepest on the centreline.
    pub width: f32,
    /// Discharge in m³/s.
    pub discharge: f32,
    /// Bed slope of the channel downstream of the grid, for the normal-depth outflow.
    pub outflow_slope: f32,
}

impl RiverSpline {
    /// Points along the spline, `per_segment` per span between control points, ends included.
    pub fn sample(&self, per_segment: u32) -> Vec<Vec2> {
        let p = &self.points;
        if p.len() < 2 {
            return p.clone();
        }
        let at = |i: isize| p[i.clamp(0, p.len() as isize - 1) as usize];
        let mut out = Vec::new();
        for i in 0..p.len() as isize - 1 {
            let (p0, p1, p2, p3) = (at(i - 1), at(i), at(i + 1), at(i + 2));
            for k in 0..per_segment {
                let t = k as f32 / per_segment as f32;
                let (t2, t3) = (t * t, t * t * t);
                out.push(
                    0.5 * (2.0 * p1
                        + (p2 - p0) * t
                        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
                        + (3.0 * p1 - p0 - 3.0 * p2 + p3) * t3),
                );
            }
        }
        out.push(*p.last().unwrap());
        out
    }
}

/// Where a path crosses the edge of the grid, in cells.
struct Crossing {
    edge: GridEdge,
    at: Vec2,
    direction: Vec2,
    entering: bool,
}

fn border_crossings(path: &[Vec2], size: Vec2) -> Vec<Crossing> {
    let inside = |p: Vec2| p.x > 0.0 && p.y > 0.0 && p.x < size.x && p.y < size.y;
    let mut out = Vec::new();
    for pair in path.windows(2) {
        let (a, b) = (pair[0], pair[1]);
        let entering = inside(b);
        if inside(a) == entering {
            continue;
        }
        // Bisect for the crossing, keeping `lo` on `a`'s side.
        let (mut lo, mut hi) = (0.0f32, 1.0f32);
        for _ in 0..32 {
            let mid = 0.5 * (lo + hi);
            if inside(a.lerp(b, mid)) == entering {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        let at = a.lerp(b, 0.5 * (lo + hi));
        let distances = [
            (GridEdge::Left, at.x.abs()),
            (GridEdge::Right, (at.x - size.x).abs()),
            (GridEdge::Bottom, at.y.abs()),
            (GridEdge::Top, (at.y - size.y).abs()),
        ];
        let edge = distances
            .into_iter()
            .min_by(|x, y| x.1.total_cmp(&y.1))
            .unwrap()
            .0;
        out.push(Crossing {
            edge,
            at: at.clamp(Vec2::ZERO, size),
            direction: b - a,
            entering,
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn placement() -> ShallowWaterPlacement {
        ShallowWaterPlacement {
            center: Vec2::new(100.0, -40.0),
            cell_size: 2.0,
            datum: 0.0,
            blend_cells: 0.0,
        }
    }

    fn discharge(profile: &ShallowWaterBorderProfile, edge: GridEdge, cell_size: f32) -> f32 {
        (0..profile.edge_len(edge))
            .map(|i| match profile.get(edge, i) {
                ShallowWaterBorderFace::Inflow {
                    discharge_per_width,
                } => discharge_per_width * cell_size,
                _ => 0.0,
            })
            .sum()
    }

    #[test]
    fn a_straight_river_enters_left_and_leaves_right() {
        let cells = UVec2::new(64, 32);
        let p = placement();
        // The grid spans x 36..164 and z -72..-8 in meters. The river runs along z = -40.
        let river = RiverSpline {
            points: vec![Vec2::new(0.0, -40.0), Vec2::new(200.0, -40.0)],
            width: 16.0,
            discharge: 30.0,
            outflow_slope: 0.001,
        };
        let mut profile = ShallowWaterBorderProfile::new(cells);
        profile.add_river(&river, &p);
        let q = discharge(&profile, GridEdge::Left, p.cell_size);
        assert!((q - 30.0).abs() < 1e-3, "inflow carries {q} m³/s");
        // Eight cells wide, centred on row 16: faces 12..20.
        let inflow: Vec<u32> = (0..32)
            .filter(|&j| {
                matches!(
                    profile.get(GridEdge::Left, j),
                    ShallowWaterBorderFace::Inflow { .. }
                )
            })
            .collect();
        assert_eq!(inflow, (12..20).collect::<Vec<_>>());
        let outflow: Vec<u32> = (0..32)
            .filter(|&j| {
                profile.get(GridEdge::Right, j) == ShallowWaterBorderFace::Outflow { slope: 0.001 }
            })
            .collect();
        assert_eq!(outflow, inflow);
        assert!(
            (0..64).all(|i| profile.get(GridEdge::Top, i) == ShallowWaterBorderFace::Edge),
            "the top edge is untouched"
        );
    }

    #[test]
    fn a_bending_river_leaves_through_the_top() {
        let cells = UVec2::new(64, 64);
        let p = ShallowWaterPlacement {
            center: Vec2::ZERO,
            cell_size: 1.0,
            ..placement()
        };
        let river = RiverSpline {
            points: vec![
                Vec2::new(-50.0, -10.0),
                Vec2::new(0.0, -5.0),
                Vec2::new(10.0, 20.0),
                Vec2::new(12.0, 60.0),
            ],
            width: 6.0,
            discharge: 5.0,
            outflow_slope: 0.0,
        };
        let mut profile = ShallowWaterBorderProfile::new(cells);
        profile.add_river(&river, &p);
        assert!((discharge(&profile, GridEdge::Left, 1.0) - 5.0).abs() < 1e-4);
        let top: Vec<u32> = (0..64)
            .filter(|&i| {
                matches!(
                    profile.get(GridEdge::Top, i),
                    ShallowWaterBorderFace::Outflow { .. }
                )
            })
            .collect();
        assert!(!top.is_empty(), "no outflow on the top edge");
        assert!(
            top.iter().all(|&i| (38..48).contains(&i)),
            "outflow at {top:?}, near x = 12 m"
        );
    }

    #[test]
    fn the_image_holds_one_row_per_edge() {
        let mut profile = ShallowWaterBorderProfile::new(UVec2::new(8, 4));
        profile.set(
            GridEdge::Top,
            7,
            ShallowWaterBorderFace::Inflow {
                discharge_per_width: 0.5,
            },
        );
        let image = profile.image();
        assert_eq!(image.texture_descriptor.size.width, 8);
        let data = image.data.unwrap();
        let o = (3 * 8 + 7) * 16;
        let lane =
            |k: usize| f32::from_le_bytes(data[o + 4 * k..o + 4 * k + 4].try_into().unwrap());
        assert_eq!((lane(0), lane(1)), (1.0, 0.5));
    }
}
