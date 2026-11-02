#!/usr/bin/env python3
"""Prepare CC0 PBR arrays and co-registered coastline masks for the beach example.

Requires Python, NumPy, and GDAL. Downloads are cached. Source rasters stay unchanged.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import urllib.request

import numpy as np
from osgeo import gdal, osr

gdal.UseExceptions()
ROOT = Path(__file__).resolve().parents[1]
GRID = 'kauai_lumahai_hanalei_shoreline_lidar_1m_2048x2048'
# API dimensions are millimeters. These repeats are meters, in material layer order.
LAYERS = [
    ('sand_01', 1.5), ('rock_boulder_dry', 1.8), ('grass_ground', 2.51),
    ('leafy_grass', 2.0), ('mossy_rock', 3.0), ('brown_mud_03', 1.3),
    ('coral_ground_02', 2.0), ('asphalt_02', 3.0),
]


def fetch(url, path, md5=None):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(url, headers={'User-Agent': 'bevy_fft beach material preparation'})
        with urllib.request.urlopen(request) as response:
            data = response.read()
        if md5 and hashlib.md5(data).hexdigest() != md5:
            raise ValueError(f'Checksum mismatch: {url}')
        path.write_bytes(data)
    if md5 and hashlib.md5(path.read_bytes()).hexdigest() != md5:
        raise ValueError(f'Invalid cached asset: {path}')
    return path


def json_asset(url, path):
    return json.loads(fetch(url, path).read_text())


def read_map(path):
    data = gdal.Open(str(path)).ReadAsArray()
    if data.ndim == 2:
        data = data[None, ...]
    maximum = np.iinfo(data.dtype).max
    return np.moveaxis(data.astype(np.float32) / maximum, 0, -1)


def shrink(data):
    h, w, c = data.shape
    return data.reshape(h // 2, 2, w // 2, 2, c).mean(axis=(1, 3))


def dds_array(path, layers, srgb):
    """Write a DX10 RGBA8 DDS array, with filtered mip chains for every layer."""
    size = layers[0].shape[0]
    mip_count = size.bit_length()
    header = [124, 0x2100F, size, size, size * 4, 0, mip_count] + [0] * 11
    header += [32, 4, int.from_bytes(b'DX10', 'little'), 0, 0, 0, 0, 0]
    header += [0x401008, 0, 0, 0, 0]
    assert len(header) == 31
    with path.open('wb') as file:
        file.write(b'DDS ' + struct.pack('<31I', *header))
        file.write(struct.pack('<5I', 29 if srgb else 28, 3, 0, len(layers), 0))
        for data in layers:
            data = data.copy()
            for mip in range(mip_count):
                file.write(np.round(np.clip(data, 0, 1) * 255).astype(np.uint8).tobytes())
                if mip == mip_count - 1:
                    break
                if srgb:
                    linear = np.where(data[..., :3] <= .04045, data[..., :3] / 12.92, ((data[..., :3] + .055) / 1.055) ** 2.4)
                    data = shrink(np.concatenate((linear, data[..., 3:]), axis=2))
                    data[..., :3] = np.where(data[..., :3] <= .0031308, data[..., :3] * 12.92, 1.055 * data[..., :3] ** (1 / 2.4) - .055)
                else:
                    normals = data[..., :3] * 2 - 1
                    data = shrink(np.concatenate((normals, data[..., 3:] ** 2), axis=2))
                    data[..., :3] /= np.maximum(np.linalg.norm(data[..., :3], axis=2, keepdims=True), 1e-6)
                    data[..., :3] = data[..., :3] * .5 + .5
                    data[..., 3] = np.sqrt(data[..., 3])


def png(path, data):
    dataset = gdal.GetDriverByName('MEM').Create('', data.shape[1], data.shape[0], 1, gdal.GDT_Byte)
    dataset.GetRasterBand(1).WriteArray(data)
    gdal.GetDriverByName('PNG').CreateCopy(str(path), dataset, options=['ZLEVEL=9'])


def masks(source, output):
    classes = gdal.Open(str(source / 'surface_classes' / f'{GRID}_surface_classes_12.tif'))
    heights = gdal.Open(str(source / 'geotiff_1m_lidar' / f'{GRID}.tif'))
    class_crs, height_crs = osr.SpatialReference(wkt=classes.GetProjection()), osr.SpatialReference(wkt=heights.GetProjection())
    if not class_crs.IsSame(height_crs) or classes.GetGeoTransform() != heights.GetGeoTransform() or classes.RasterXSize != heights.RasterXSize or classes.RasterYSize != heights.RasterYSize:
        raise ValueError('Class and elevation rasters must be co-registered. Reproject classes with nearest-neighbor sampling first.')
    ids, bed = classes.ReadAsArray(), heights.ReadAsArray()
    if ids.shape != (2048, 2048) or not np.isin(ids, np.arange(1, 13)).all():
        raise ValueError('Expected the 2048-square shoreline raster with class IDs 1 through 12')
    water = (bed <= 0).astype(np.uint8)
    seed = gdal.GetDriverByName('MEM').Create('', 2048, 2048, 1, gdal.GDT_Byte)
    seed.GetRasterBand(1).WriteArray(water)
    distance = gdal.GetDriverByName('MEM').Create('', 2048, 2048, 1, gdal.GDT_Float32)
    gdal.ComputeProximity(seed.GetRasterBand(1), distance.GetRasterBand(1), ['VALUES=1', 'DISTUNITS=PIXEL'])
    t = np.clip((bed - .15) / .85, 0, 1)
    wetness = np.isin(ids, [4, 9]) * np.exp(-distance.ReadAsArray() / 3.0) * (1 - t * t * (3 - 2 * t))
    # The planar bevy_terrain processor maps raster north to +Z. Solver rows increase with Z.
    png(output / 'surface_classes.png', np.flipud(ids))
    png(output / 'sand_capillary.png', np.flipud(np.round(wetness * 255).astype(np.uint8)))
    return {'grid': GRID, 'crs': classes.GetProjection(), 'geotransform': classes.GetGeoTransform(), 'world_mapping': 'x = column - 1024, z = 1024 - row. Exported PNG rows increase with world Z.', 'class_counts': {str(i): int((ids == i).sum()) for i in range(1, 13)}, 'capillary_width_m': 3.0, 'capillary_height_m': [0.15, 1.0]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, default=Path.home() / 'Downloads/hawaii_topobathy')
    parser.add_argument('--bake-terrain', action='store_true', help='Rebuild height tiles with the neighboring bevy_terrain processor')
    parser.add_argument('--processor', type=Path, default=ROOT.parent / 'bevy_terrain/target/release/bevy_terrain_preprocess')
    args = parser.parse_args()
    output = ROOT / 'assets/beach_materials'
    output.mkdir(parents=True, exist_ok=True)
    manifest = {'license': 'CC0-1.0', 'license_url': 'https://polyhaven.com/license', 'layers': [], 'coastline': masks(args.source_root, output)}
    colors, normals = [], []
    for asset, repeat in LAYERS:
        info = json_asset(f'https://api.polyhaven.com/info/{asset}', output / 'cache' / f'{asset}_info.json')
        files = json_asset(f'https://api.polyhaven.com/files/{asset}', output / 'cache' / f'{asset}_files.json')
        physical = np.array(info['dimensions']) / 1000
        if not np.allclose(physical, repeat, atol=.01):
            raise ValueError(f'Physical dimensions changed for {asset}: {physical}')
        maps, sources = {}, {}
        for name, options in [('color', ['diff', 'diffuse']), ('normal', ['nor_gl']), ('roughness', ['rough', 'roughness'])]:
            key = next(k for k in files if k.lower() in options)
            formats = files[key]['1k']
            format_name = 'png' if name == 'normal' else 'jpg'
            entry = formats.get(format_name) or formats['png']
            path = output / 'cache' / Path(entry['url']).name
            maps[name] = read_map(fetch(entry['url'], path, entry['md5']))
            sources[name] = entry
        colors.append(np.concatenate((maps['color'][..., :3], np.ones((1024, 1024, 1), dtype=np.float32)), axis=2))
        normals.append(np.concatenate((maps['normal'][..., :3], maps['roughness'][..., :1]), axis=2))
        manifest['layers'].append({'asset': asset, 'page': f'https://polyhaven.com/a/{asset}', 'authors': info['authors'], 'repeat_m': physical.tolist(), 'files': sources})
        print(f'{asset}: repeat {repeat:g} m', flush=True)
    dds_array(output / 'albedo.dds', colors, True)
    dds_array(output / 'normal_roughness.dds', normals, False)
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    if args.bake_terrain:
        subprocess.run([str(args.processor), str(args.source_root / 'geotiff_1m_lidar' / f'{GRID}.tif'), '--terrain-path', 'assets/terrains/kauai_shoreline', '--overwrite', '--data-type', 'float32', '--format', 'r32f', '--planar', '--side-length', '2048'], cwd=ROOT, check=True)
    print(f'Prepared material arrays and coastline masks in {output}')


if __name__ == '__main__':
    main()
