# Materials for the Kauai coastline

The beach ground uses the 1 m classification raster for the same 2048 m Lumahaʻi shoreline square as the terrain. Material boundaries follow class IDs rather than elevation. Sand and basalt classes continue across sea level. Forest classes describe the ground beneath vegetation. The material does not add trees, shrubs, buildings, or coral geometry.

Run `python3 tools/prepare_beach_materials.py` from the repository root. The script needs NumPy and GDAL's Python bindings. It reads `~/Downloads/hawaii_topobathy` by default. `--source-root` selects another dataset folder. Downloads, generated texture arrays, masks, and their detailed provenance manifest live in `assets/beach_materials`, which is ignored by git.

The script checks that the class and elevation rasters have the same CRS, geotransform, and dimensions. It preserves integer class IDs and flips raster rows to match the existing planar terrain processor, where north points along positive world Z. One world unit is one meter. The class map stays at its native 1 m resolution. No additional reprojection is needed for this co-registered square.

The existing height tiles remain valid. To rebuild them from this source dataset, pass `--bake-terrain`. That runs `../bevy_terrain/target/release/bevy_terrain_preprocess` with the same planar 2048 m extent and float height format. `--processor` selects another processor executable.

## Texture sources and scale

These Poly Haven assets are [CC0](https://polyhaven.com/license). The preparation script downloads albedo, OpenGL normal, and roughness maps from the public API. It checks file MD5 hashes and records authors, URLs, and scan dimensions in `manifest.json`. Repeat sizes below come from the API's physical dimensions, converted from millimeters to meters.

| Classes | Ground layer | Repeat |
| --- | --- | --- |
| 4, 9: beach and submerged sand | [Sand 01](https://polyhaven.com/a/sand_01) | 1.5 m |
| 3, 10: exposed and submerged rock | [Rock Boulder Dry](https://polyhaven.com/a/rock_boulder_dry), darkened for volcanic rock | 1.8 m |
| 5: grassland | [Grass Ground](https://polyhaven.com/a/grass_ground) | 2.51 m |
| 6: coastal scrub | [Leafy Grass](https://polyhaven.com/a/leafy_grass) | 2 m |
| 7: forest ground | [Mossy Rock](https://polyhaven.com/a/mossy_rock), for moss-covered forest ground | 3 m |
| 8: river and wetland bed | [Brown Mud 03](https://polyhaven.com/a/brown_mud_03) | 1.3 m |
| 11, 12: reef and reef rubble | [Coral Ground 02](https://polyhaven.com/a/coral_ground_02) | 2 m |
| 1: paved roads | [Asphalt 02](https://polyhaven.com/a/asphalt_02) | 3 m |

Class 2 marks structures. Without building geometry, the terrain beneath them uses the rock layer. The classification schema is read from the supplied `surface_classes_schema.json`. This is a material interpretation of that dataset, not an independent verification of its land-cover or benthic labels.

The arrays contain 1K maps with a full mip chain. Color mips are filtered in linear light. Normal mips are renormalized, and roughness is packed into the normal texture's alpha channel. Triplanar projection keeps texture scale consistent on cliffs. Translated scan patches blend on a triangular lattice to reduce visible repetition while retaining the same physical scale. Rock albedo is darkened toward basalt, and vegetation layers receive a green tint for coastal ground cover. Adjacent classes blend their shaded materials across one source texel, without interpolating class IDs.

## Sand moisture

The baked capillary mask applies only to sand classes. It fades over a 3 m distance from cells at or below sea level and from 0.15 m to 1 m above sea level. These are appearance parameters for the bank, not measured soil properties.

A persistent GPU buffer then records inundation from the shallow-water solver. Water columns between 5 mm and 25 mm saturate the mask. Exposed cells dry with a 120-second half-life toward the capillary baseline. Wet sand darkens, its roughness drops, and its fine normal detail softens. Pausing the solver freezes drying.

The Beach panel's Ground view shows the material, surface classes, or sand wetness. For headless captures, set `BEVY_FFT_GROUND_VIEW=classes` or `wetness` alongside the screenshot variables.
