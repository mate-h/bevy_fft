#!/usr/bin/env python3
"""Bake terrain and sea-floor depth for a bay or river area from Esri TopoBathy.

Fetches a square around a latitude and longitude through the ImageServer `exportImage` operation
as 32-bit float tiles, which the server reprojects to the local UTM zone. Mosaics them, carves
river channels along splines, and writes:

  <out>/source/height.tif    carved heights, a north-up float32 GeoTIFF in local UTM
  <out>/seafloor_depth.r32   meters of water below sea level (0 on land), little-endian float32,
                             rows increasing with world Z, for the shallow water bed and the LOD data
  <out>/manifest.json        CRS, geotransform, size, world mapping and sources

With `--bake`, it also runs the neighboring bevy_terrain processor on the carved heights, which
writes `config.tc.ron` and the height tiles in `<out>`. The world mapping is the planar one that
processor uses: raster north maps to +Z, and the square's centre is the origin.

The source is WorldElevation/TopoBathy on elevation.arcgis.com, at most 5000 by 5000 per request.
It needs an ArcGIS API key with elevation access in `ARCGIS_API_KEY`, which goes in a request
header, never the URL. The public WorldElevation3D/TopoBathy3D service is no substitute: its
exportImage answers with a near-constant 1.5-2.4 m everywhere, mountains and open sea alike.
`--input` skips the fetch and carves an existing GeoTIFF instead, such as a local LiDAR survey.

Elevation models record the water surface over rivers, not the bed. Pass `--rivers rivers.json`
to cut channels to a depth set along each spline:

  [{"width": 30, "points": [[x, z, depth], ...]}, ...]

Points are local meters (x east, z north, from the square's centre). Depth is meters below the
model's surface along the centerline and is interpolated along the spline. The cross-section is
parabolic: full depth on the centerline, zero at the banks.

Baked data is cached and written under assets/terrains, which git ignores. Requires Python,
NumPy and GDAL (Homebrew's python3 has both).
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
import urllib.parse
import urllib.request

import numpy as np
from osgeo import gdal, osr

gdal.UseExceptions()
ROOT = Path(__file__).resolve().parents[1]
SERVICE = 'https://elevation.arcgis.com/arcgis/rest/services/WorldElevation/TopoBathy/ImageServer'
MAX_REQUEST = 5000
USER_AGENT = 'bevy_fft topobathy bake'


def utm_epsg(lat, lon):
    zone = int((lon + 180) // 6) % 60 + 1
    return (32600 if lat >= 0 else 32700) + zone


def to_utm(epsg, lat, lon):
    wgs, utm = osr.SpatialReference(), osr.SpatialReference()
    wgs.ImportFromEPSG(4326)
    utm.ImportFromEPSG(epsg)
    wgs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    utm.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    x, y, _ = osr.CoordinateTransformation(wgs, utm).TransformPoint(lon, lat)
    return x, y


def request(url, key, retries=4):
    headers = {'User-Agent': USER_AGENT}
    if key:
        headers['X-Esri-Authorization'] = f'Bearer {key}'
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=300) as response:
                data, kind = response.read(), response.headers.get('Content-Type', '')
            if 'json' in kind or data[:1] == b'{':
                raise RuntimeError(f'Service error: {data[:300].decode(errors="replace")}')
            return data
        except (OSError, RuntimeError) as error:
            if attempt == retries - 1:
                raise
            print(f'  retrying after {error}', flush=True)
            time.sleep(2 ** attempt)


def fetch_piece(key, epsg, bbox, size, cache):
    """One exportImage request, cached by its parameters. Returns float32 rows, north first."""
    params = {
        'bbox': ','.join(f'{v:.3f}' for v in bbox), 'bboxSR': epsg, 'imageSR': epsg,
        'size': f'{size[0]},{size[1]}', 'format': 'tiff', 'pixelType': 'F32',
        'interpolation': 'RSP_BilinearInterpolation', 'noData': -99999, 'f': 'image',
    }
    url = f'{SERVICE}/exportImage?{urllib.parse.urlencode(params)}'
    path = cache / f'topobathy_{hashlib.sha1(url.encode()).hexdigest()[:16]}.tif'
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        part = path.with_suffix('.part')
        part.write_bytes(request(url, key))
        part.rename(path)
    data = gdal.Open(str(path)).ReadAsArray().astype(np.float32)
    if data.shape != (size[1], size[0]):
        raise ValueError(f'{path.name}: expected {size[1]}x{size[0]}, got {data.shape}')
    return data


def fetch(key, epsg, centre, side, resolution, cache):
    n = int(round(side / resolution))
    pieces = math.ceil(n / MAX_REQUEST)
    print(f'TopoBathy: {n}x{n} px at {resolution:g} m in EPSG:{epsg}, {pieces * pieces} requests', flush=True)
    x0, y1 = centre[0] - side / 2, centre[1] + side / 2
    heights = np.empty((n, n), dtype=np.float32)
    edges = [round(i * n / pieces) for i in range(pieces + 1)]
    for r in range(pieces):
        for c in range(pieces):
            cols, rows = (edges[c], edges[c + 1]), (edges[r], edges[r + 1])
            bbox = (x0 + cols[0] * resolution, y1 - rows[1] * resolution, x0 + cols[1] * resolution, y1 - rows[0] * resolution)
            print(f'  piece {r * pieces + c + 1}/{pieces * pieces}', flush=True)
            heights[rows[0]:rows[1], cols[0]:cols[1]] = fetch_piece(
                key, epsg, bbox, (cols[1] - cols[0], rows[1] - rows[0]), cache)
    missing = heights <= -99990
    if missing.any():
        print(f'  {missing.mean():.2%} of pixels have no data; filling them from neighbors', flush=True)
        heights = fill_missing(heights, missing)
    return heights


def warp(path, epsg, centre, side, resolution):
    bounds = (centre[0] - side / 2, centre[1] - side / 2, centre[0] + side / 2, centre[1] + side / 2)
    n = int(round(side / resolution))
    dataset = gdal.Warp('', str(path), format='MEM', dstSRS=f'EPSG:{epsg}', outputBounds=bounds, width=n, height=n,
                        resampleAlg='bilinear', dstNodata=-99999, outputType=gdal.GDT_Float32)
    heights = dataset.ReadAsArray().astype(np.float32)
    missing = heights <= -99990
    if missing.all():
        raise ValueError(f'{path} does not cover the square')
    return fill_missing(heights, missing) if missing.any() else heights


def fill_missing(heights, missing):
    dataset = gdal.GetDriverByName('MEM').Create('', heights.shape[1], heights.shape[0], 1, gdal.GDT_Float32)
    band = dataset.GetRasterBand(1)
    band.WriteArray(np.where(missing, -99999, heights))
    band.SetNoDataValue(-99999)
    gdal.FillNodata(band, None, maxSearchDist=max(heights.shape), smoothingIterations=0)
    return band.ReadAsArray()


def catmull_rom(points, spacing):
    """Samples a uniform Catmull-Rom spline through `points` (N x 3)."""
    p = np.asarray(points, dtype=np.float64)
    if len(p) < 2:
        raise ValueError('A river spline needs at least two points')
    p = np.vstack((2 * p[0] - p[1], p, 2 * p[-1] - p[-2]))
    out = []
    for i in range(1, len(p) - 2):
        a, b, c, d = p[i - 1], p[i], p[i + 1], p[i + 2]
        steps = max(int(np.linalg.norm(c[:2] - b[:2]) / spacing), 1)
        t = np.linspace(0, 1, steps, endpoint=False)[:, None]
        out.append(0.5 * (2 * b + (c - a) * t + (2 * a - 5 * b + 4 * c - d) * t ** 2 + (3 * b - a - 3 * c + d) * t ** 3))
    out.append(p[-2][None])
    return np.vstack(out)


def carve(heights, rivers, side, resolution):
    """Lowers heights along each spline. The model's height on the centerline is the water surface."""
    n = heights.shape[0]
    original = heights.copy()
    # Pixel centre (row, col) -> local x east, z north.
    coord = (np.arange(n) + 0.5) * resolution - side / 2
    for index, river in enumerate(rivers):
        half = float(river['width']) / 2
        line = catmull_rom(river['points'], resolution / 2)
        cols = np.clip(((line[:, 0] + side / 2) / resolution).astype(int), 0, n - 1)
        rows = np.clip(((side / 2 - line[:, 1]) / resolution).astype(int), 0, n - 1)
        surface = original[rows, cols]
        reach = int(math.ceil(half / resolution)) + 1
        # Each pixel takes its bed from the nearest centerline sample, so a sloping river surface
        # doesn't let downstream samples cut the bed lower.
        nearest = np.full(heights.shape, np.inf, dtype=np.float32)
        bed = np.full(heights.shape, np.inf, dtype=np.float32)
        for (x, z, depth), level in zip(line, surface):
            c, r = int((x + side / 2) / resolution), int((side / 2 - z) / resolution)
            r0, r1, c0, c1 = max(r - reach, 0), min(r + reach + 1, n), max(c - reach, 0), min(c + reach + 1, n)
            if r0 >= r1 or c0 >= c1:
                continue
            dx = coord[c0:c1][None, :] - x
            dz = (side / 2 - (np.arange(r0, r1) + 0.5) * resolution)[:, None] - z
            d2 = dx * dx + dz * dz
            closer = (d2 < half * half) & (d2 < nearest[r0:r1, c0:c1])
            nearest[r0:r1, c0:c1][closer] = d2[closer]
            bed[r0:r1, c0:c1][closer] = (level - depth * (1 - d2 / (half * half)))[closer]
        np.minimum(heights, bed, out=heights)
        print(f'  river {index}: {len(river["points"])} points, {half * 2:g} m wide, '
              f'lowered by up to {float((original - heights).max()):.2f} m', flush=True)
    return heights


def write_geotiff(path, data, epsg, transform):
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset = gdal.GetDriverByName('GTiff').Create(str(path), data.shape[1], data.shape[0], 1, gdal.GDT_Float32,
                                                   options=['COMPRESS=DEFLATE', 'PREDICTOR=3', 'TILED=YES'])
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg)
    dataset.SetProjection(srs.ExportToWkt())
    dataset.SetGeoTransform(transform)
    dataset.GetRasterBand(1).WriteArray(data)
    dataset.FlushCache()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--name', required=True, help='Terrain folder under assets/terrains')
    parser.add_argument('--center', required=True, help='Square centre as "lat,lon" in degrees')
    parser.add_argument('--side', type=float, default=2048.0, help='Square side in meters (default 2048)')
    parser.add_argument('--resolution', type=float, default=1.0, help='Meters per pixel (default 1)')
    parser.add_argument('--sea-level', type=float, default=0.0, help='Height of the sea surface in meters')
    parser.add_argument('--rivers', type=Path, help='JSON list of river splines to carve')
    parser.add_argument('--input', type=Path, help='Carve this GeoTIFF instead of fetching; it is warped to the square')
    parser.add_argument('--bake', action='store_true', help='Run the bevy_terrain processor on the carved heights')
    parser.add_argument('--processor', type=Path, default=ROOT.parent / 'bevy_terrain/target/release/bevy_terrain_preprocess')
    args = parser.parse_args()

    key = os.environ.get('ARCGIS_API_KEY') or None
    if not key and not args.input:
        parser.error('set ARCGIS_API_KEY to fetch TopoBathy, or pass --input')
    lat, lon = (float(v) for v in args.center.split(','))
    side = round(args.side / args.resolution) * args.resolution
    epsg = utm_epsg(lat, lon)
    centre = to_utm(epsg, lat, lon)
    out = ROOT / 'assets/terrains' / args.name

    if args.input:
        heights = warp(args.input, epsg, centre, side, args.resolution)
    else:
        heights = fetch(key, epsg, centre, side, args.resolution, out / 'cache')
    rivers = json.loads(args.rivers.read_text()) if args.rivers else []
    if rivers:
        heights = carve(heights, rivers, side, args.resolution)
    transform = (centre[0] - side / 2, args.resolution, 0.0, centre[1] + side / 2, 0.0, -args.resolution)
    write_geotiff(out / 'source/height.tif', heights, epsg, transform)
    depth = np.maximum(args.sea_level - heights, 0).astype('<f4')
    # North-up raster rows run toward -Z; the depth file's rows increase with world Z.
    (out / 'seafloor_depth.r32').write_bytes(np.flipud(depth).tobytes())
    n = heights.shape[0]
    manifest = {
        'source': str(args.input) if args.input else SERVICE, 'center_lat_lon': [lat, lon], 'crs': f'EPSG:{epsg}',
        'geotransform': transform, 'size': [n, n], 'side_m': side, 'resolution_m': args.resolution,
        'sea_level_m': args.sea_level, 'height_range_m': [float(heights.min()), float(heights.max())],
        'wet_fraction': float((depth > 0).mean()), 'rivers': rivers,
        'world_mapping': f'x = (column + 0.5) * {args.resolution:g} - {side / 2:g}, z = {side / 2:g} - (row + 0.5) * {args.resolution:g} '
                         'for source/height.tif rows. seafloor_depth.r32 rows increase with world Z.',
    }
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Heights {manifest["height_range_m"][0]:.1f} to {manifest["height_range_m"][1]:.1f} m, '
          f'{manifest["wet_fraction"]:.1%} below sea level. Wrote {out.relative_to(ROOT)}', flush=True)

    if args.bake:
        if not args.processor.exists():
            raise SystemExit(f'No processor at {args.processor}. Build it with '
                             '`cargo build --release -p bevy_terrain_preprocess` in ../bevy_terrain.')
        subprocess.run([str(args.processor), str(out / 'source/height.tif'), '--terrain-path', str(out.relative_to(ROOT)),
                        '--overwrite', '--data-type', 'float32', '--format', 'r32f', '--planar', '--side-length', f'{side:g}'],
                       cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
