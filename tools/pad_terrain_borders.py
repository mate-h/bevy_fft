#!/usr/bin/env python3
"""Pads the outer edge of a baked bevy_terrain height tile set with the data inside it.

The bevy_terrain processor fills the area past the source raster with 0 m and resamples that into
the last texels of the tiles along the map's edge, so the terrain rises or sinks to 0 m in a thin
line around the map: at sea, a ridge up to the water surface. This replaces the outermost
`--texels` texels of every edge tile, on every LOD, with the first texel inside them.

  python3 tools/pad_terrain_borders.py assets/terrains/kauai_shoreline

Run it after each bake; tools/prepare_beach_materials.py and tools/bake_topobathy.py do. Needs
GDAL (Homebrew's python3 has it).
"""
import argparse
from pathlib import Path

from osgeo import gdal

gdal.UseExceptions()

# The processor's zero fill reaches two texels in at the finest LOD; three leaves a margin.
DEFAULT_TEXELS = 3


def pad_tile(path, west, east, south, north, texels):
    """Copies the first good row or column over the outer ones on the given sides. Rows run
    toward +Z, the map's north."""
    dataset = gdal.Open(str(path), gdal.GA_Update)
    band = dataset.GetRasterBand(1)
    heights = band.ReadAsArray()
    k = texels
    if west:
        heights[:, :k] = heights[:, k:k + 1]
    if east:
        heights[:, -k:] = heights[:, -k - 1:-k]
    if south:
        heights[:k, :] = heights[k:k + 1, :]
    if north:
        heights[-k:, :] = heights[-k - 1:-k, :]
    band.WriteArray(heights)
    dataset.FlushCache()


def pad_terrain(terrain, texels=DEFAULT_TEXELS):
    """Pads every LOD of the height tiles under `terrain`, which holds `height/<lod>/0_0/`."""
    padded = 0
    for lod_dir in sorted((Path(terrain) / 'height').iterdir()):
        if not lod_dir.name.isdigit():
            continue
        lod = int(lod_dir.name)
        tiles = 1 << lod
        for x in range(tiles):
            for y in range(tiles):
                edges = (x == 0, x == tiles - 1, y == 0, y == tiles - 1)
                if not any(edges):
                    continue
                path = lod_dir / '0_0' / f'0_{lod}_{x}_{y}.tif'
                if path.exists():
                    pad_tile(path, *edges, texels)
                    padded += 1
    return padded


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('terrain', type=Path, help='Terrain folder, such as assets/terrains/kauai_shoreline')
    parser.add_argument('--texels', type=int, default=DEFAULT_TEXELS, help='Outer texels to replace on each edge')
    args = parser.parse_args()
    count = pad_terrain(args.terrain, args.texels)
    print(f'Padded {count} edge tiles in {args.terrain}')


if __name__ == '__main__':
    main()
