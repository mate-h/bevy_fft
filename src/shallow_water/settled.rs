//! A cache of a settled shallow water grid: the bed and depth of every cell and the velocity of
//! every face, saved once a river has reached its steady state and loaded over the preset on the
//! next reset, so the grid starts settled instead of filling for minutes.

use std::{
    io::{self, Read, Write},
    path::Path,
    sync::{Arc, Mutex},
};

use bevy::{
    prelude::*,
    render::{
        gpu_readback::{Readback, ReadbackComplete},
        render_resource::{
            Extent3d, Texture, TextureDescriptor, TextureDimension, TextureFormat, TextureUsages,
        },
        renderer::{RenderDevice, RenderQueue},
    },
};

use super::{ShallowWaterController, ShallowWaterPlacement};

const MAGIC: &[u8; 8] = b"SWSETTL1";

/// The state of a settled grid. [`ShallowWaterController::settled`] loads it on a reset.
#[derive(Clone, Debug, PartialEq)]
pub struct ShallowWaterSettled {
    pub cells: UVec2,
    /// The grid's cell width in meters, or 1 for a grid without a placement. A cache loads only
    /// over a grid with the same cells, cell width and centre.
    pub cell_size: f32,
    pub center: Vec2,
    /// `bed_water` per cell, four floats each: bed, depth and two unused lanes.
    pub bed_water: Vec<f32>,
    /// Velocity of each face across x, `cells.x + 1` per row.
    pub face_u: Vec<f32>,
    /// Velocity of each face across z, `cells.x` per row, `cells.y + 1` rows.
    pub face_w: Vec<f32>,
}

/// The floats of a texture readback, without the padding that takes each row to 256 bytes.
fn unpad(bytes: &[u8], width: u32, height: u32, lanes: u32) -> Option<Vec<f32>> {
    let row = (width * lanes * 4) as usize;
    let stride = row.next_multiple_of(256);
    let stride = if height > 1 { stride } else { row };
    if bytes.len() < stride * (height as usize - 1) + row {
        return None;
    }
    let mut out = Vec::with_capacity((width * height * lanes) as usize);
    for y in 0..height as usize {
        let start = y * stride;
        out.extend(
            bytes[start..start + row]
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap())),
        );
    }
    Some(out)
}

impl ShallowWaterSettled {
    /// Builds the cache from texture readbacks of a grid's `bed_water`, `flow_x` and `flow_y`
    /// images, as [`capture_shallow_water_settled`] takes them.
    pub fn from_readbacks(
        cells: UVec2,
        placement: Option<&ShallowWaterPlacement>,
        bed_water: &[u8],
        face_u: &[u8],
        face_w: &[u8],
    ) -> Option<Self> {
        Some(Self {
            cells,
            cell_size: placement.map_or(1.0, |p| p.cell_size),
            center: placement.map_or(Vec2::ZERO, |p| p.center),
            bed_water: unpad(bed_water, cells.x, cells.y, 4)?,
            face_u: unpad(face_u, cells.x + 1, cells.y, 1)?,
            face_w: unpad(face_w, cells.x, cells.y + 1, 1)?,
        })
    }

    /// Whether the cache was taken from a grid laid out like `controller`'s.
    pub fn fits(&self, controller: &ShallowWaterController) -> bool {
        let placement = controller.placement.as_ref();
        self.cells == UVec2::new(controller.cells_x, controller.cells_y)
            && self.cell_size == placement.map_or(1.0, |p| p.cell_size)
            && self.center == placement.map_or(Vec2::ZERO, |p| p.center)
    }

    pub fn write(&self, mut out: impl Write) -> io::Result<()> {
        out.write_all(MAGIC)?;
        for v in [self.cells.x, self.cells.y] {
            out.write_all(&v.to_le_bytes())?;
        }
        for v in [self.cell_size, self.center.x, self.center.y] {
            out.write_all(&v.to_le_bytes())?;
        }
        for lane in [&self.bed_water, &self.face_u, &self.face_w] {
            out.write_all(bytemuck::cast_slice(lane))?;
        }
        Ok(())
    }

    pub fn read(mut input: impl Read) -> io::Result<Self> {
        let bad = |what: &str| io::Error::new(io::ErrorKind::InvalidData, what.to_string());
        let mut magic = [0u8; 8];
        input.read_exact(&mut magic)?;
        if &magic != MAGIC {
            return Err(bad("not a settled shallow water cache"));
        }
        let mut word = [0u8; 4];
        let mut next = |input: &mut dyn Read| -> io::Result<[u8; 4]> {
            input.read_exact(&mut word)?;
            Ok(word)
        };
        let cells = UVec2::new(
            u32::from_le_bytes(next(&mut input)?),
            u32::from_le_bytes(next(&mut input)?),
        );
        let cell_size = f32::from_le_bytes(next(&mut input)?);
        let center = Vec2::new(
            f32::from_le_bytes(next(&mut input)?),
            f32::from_le_bytes(next(&mut input)?),
        );
        if cells.x == 0 || cells.y == 0 || cells.max_element() > 1 << 14 {
            return Err(bad("implausible grid size"));
        }
        let mut lane = |count: u32| -> io::Result<Vec<f32>> {
            let mut bytes = vec![0u8; count as usize * 4];
            input.read_exact(&mut bytes)?;
            Ok(bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect())
        };
        Ok(Self {
            bed_water: lane(cells.x * cells.y * 4)?,
            face_u: lane((cells.x + 1) * cells.y)?,
            face_w: lane(cells.x * (cells.y + 1))?,
            cells,
            cell_size,
            center,
        })
    }

    pub fn save(&self, path: impl AsRef<Path>) -> io::Result<()> {
        let mut bytes = Vec::new();
        self.write(&mut bytes)?;
        std::fs::write(path, bytes)
    }

    pub fn load(path: impl AsRef<Path>) -> io::Result<Self> {
        Self::read(std::fs::read(path)?.as_slice())
    }

    /// Uploads the three lanes as textures shaped like the grid's, to copy from on a reset.
    pub(crate) fn upload(&self, device: &RenderDevice, queue: &RenderQueue) -> [Texture; 3] {
        let texture = |label: &'static str, width, height, format, data: &[f32]| {
            device.create_texture_with_data(
                queue,
                &TextureDescriptor {
                    label: Some(label),
                    size: Extent3d {
                        width,
                        height,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: TextureDimension::D2,
                    format,
                    usage: TextureUsages::COPY_SRC | TextureUsages::COPY_DST,
                    view_formats: &[],
                },
                Default::default(),
                bytemuck::cast_slice(data),
            )
        };
        let c = self.cells;
        [
            texture(
                "shallow_water_settled_bed_water",
                c.x,
                c.y,
                TextureFormat::Rgba32Float,
                &self.bed_water,
            ),
            texture(
                "shallow_water_settled_face_u",
                c.x + 1,
                c.y,
                TextureFormat::R32Float,
                &self.face_u,
            ),
            texture(
                "shallow_water_settled_face_w",
                c.x,
                c.y + 1,
                TextureFormat::R32Float,
                &self.face_w,
            ),
        ]
    }
}

/// Reads the controller's grid back and hands the settled state to `done` once all three lanes
/// have arrived, a few frames later. Run it once the river has settled, then
/// [`ShallowWaterSettled::save`] the state.
pub fn capture_shallow_water_settled(
    commands: &mut Commands,
    controller: &ShallowWaterController,
    done: impl FnOnce(ShallowWaterSettled) + Send + Sync + 'static,
) {
    let cells = UVec2::new(controller.cells_x, controller.cells_y);
    let placement = controller.placement;
    let parts: Arc<Mutex<[Option<Vec<u8>>; 3]>> = Arc::default();
    let done = Arc::new(Mutex::new(Some(done)));
    for (lane, image) in [
        controller.bed_water.clone(),
        controller.flow_x.clone(),
        controller.flow_y.clone(),
    ]
    .into_iter()
    .enumerate()
    {
        let parts = parts.clone();
        let done = done.clone();
        commands.spawn(Readback::texture(image)).observe(
            move |event: On<ReadbackComplete>, mut commands: Commands| {
                commands.entity(event.entity).despawn();
                let mut parts = parts.lock().unwrap();
                if parts[lane].is_some() {
                    return;
                }
                parts[lane] = Some(event.data.clone());
                if let [Some(b), Some(u), Some(w)] = &*parts
                    && let Some(settled) =
                        ShallowWaterSettled::from_readbacks(cells, placement.as_ref(), b, u, w)
                    && let Some(done) = done.lock().unwrap().take()
                {
                    done(settled);
                }
            },
        );
    }
}
