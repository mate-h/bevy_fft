#!/usr/bin/env python3
"""Carves the beach example's river channel into its baked bevy_terrain height tiles.

The beach lowers its shallow water bed along `CHANNEL` in examples/beach/main.rs, where the
LiDAR recorded the river's water surface instead of its bed. The terrain mesh draws the tiles
under assets/terrains/kauai_shoreline/height, which still hold that surface, so the water runs
under the drawn ground. This applies the same carve to every tile of every LOD, in place:

  /opt/homebrew/bin/python3 tools/carve_terrain_tiles.py

It reads `CHANNEL`, `CHANNEL_WIDTH` and `CHANNEL_BANK_RISE` from the example's source, so the
two stay in step. The carve only lowers heights, so running it twice changes nothing. A full
re-bake with tools/bake_topobathy.py replaces the tiles and needs this run again. Copy the
height folder first if you want to undo it.
"""
import argparse
import re
from pathlib import Path

import numpy as np
from osgeo import gdal

gdal.UseExceptions()
ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'examples/beach/main.rs'
TILES = ROOT / 'assets/terrains/kauai_shoreline/height'
SIDE = 2048.0
BORDER = 1


def channel(source):
    text = source.read_text()
    block = text[text.index('const CHANNEL: ['):]
    block = block[:block.index('];')]
    points = np.array([[float(v) for v in m] for m in re.findall(
        r'Vec3::new\(([-\d.]+), ([-\d.]+), ([-\d.]+)\)', block)])
    width = float(re.search(r'const CHANNEL_WIDTH: f32 = ([\d.]+);', text).group(1))
    rise = float(re.search(r'const CHANNEL_BANK_RISE: f32 = ([\d.]+);', text).group(1))
    return points, width, rise


def carve(heights, x, z, points, width, rise):
    """The example's `carve_channel`: a parabolic section along each segment, never raising."""
    half = width / 2
    out = heights.copy()
    for a, b in zip(points, points[1:]):
        ab = b[:2] - a[:2]
        t = np.clip(((x - a[0]) * ab[0] + (z - a[1]) * ab[1]) / ab.dot(ab), 0, 1)
        d = np.hypot(x - (a[0] + t * ab[0]), z - (a[1] + t * ab[1]))
        inside = d < half
        bed = a[2] + t * (b[2] - a[2]) + rise * (d / half) ** 2
        out[inside] = np.minimum(out[inside], bed[inside])
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--tiles', type=Path, default=TILES)
    args = parser.parse_args()
    points, width, rise = channel(SOURCE)
    lo, hi = points[:, :2].min(0) - width, points[:, :2].max(0) + width
    for lod_dir in sorted(p for p in args.tiles.iterdir() if p.is_dir()):
        lod = int(lod_dir.name)
        count = 2 ** lod
        for path in sorted((lod_dir / '0_0').glob('*.tif')):
            tx, ty = (int(v) for v in path.stem.split('_')[2:4])
            # The tile's world square, x east and z as the beach's TerrainBed reads it.
            x0, z0 = (tx / count - 0.5) * SIDE, (ty / count - 0.5) * SIDE
            span = SIDE / count
            if x0 > hi[0] or x0 + span < lo[0] or z0 > hi[1] or z0 + span < lo[1]:
                continue
            data = gdal.Open(str(path))
            heights = data.GetRasterBand(1).ReadAsArray()
            size = heights.shape[0]
            centre = size - 2 * BORDER
            texel = (np.arange(size) + 0.5 - BORDER) / centre * span
            x, z = np.meshgrid(x0 + texel, z0 + texel)
            carved = carve(heights, x, z, points, width, rise)
            lowered = int((carved < heights - 1e-4).sum())
            data = None
            if lowered == 0:
                continue
            tmp = path.with_suffix('.tmp.tif')
            out = gdal.GetDriverByName('GTiff').Create(
                str(tmp), size, size, 1, gdal.GDT_Float32, ['TILED=YES', 'BLOCKXSIZE=512', 'BLOCKYSIZE=512'])
            out.GetRasterBand(1).WriteArray(carved.astype(np.float32))
            out = None
            tmp.replace(path)
            print(f'{path.relative_to(args.tiles)}: lowered {lowered} texels, '
                  f'by up to {float((heights - carved).max()):.2f} m')


if __name__ == '__main__':
    main()
