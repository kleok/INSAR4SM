#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Water bodies from ESA WorldCover (10 m, https://esa-worldcover.org), to skip grid cells where soil moisture has no meaning.

Only the AOI window of the cloud optimized GeoTIFF tiles is read (no account needed).
"""
from __future__ import annotations
import os
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.merge import merge
from rasterio.windows import from_bounds

WORLDCOVER_URL = 'https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_{}_Map.tif'
WATER_CLASS = 80


def worldcover_tile(lon:float, lat:float)->str:
    """Name of the 3x3 degree WorldCover tile that contains a point (e.g. N42E000).
    """
    lat0, lon0 = int(np.floor(lat / 3) * 3), int(np.floor(lon / 3) * 3)
    return '{}{:02d}{}{:03d}'.format('N' if lat0 >= 0 else 'S', abs(lat0), 'E' if lon0 >= 0 else 'W', abs(lon0))


def get_worldcover(AOI_File:str, out_file:str, margin_deg:float = 0.01)->str:
    """Reads ESA WorldCover around the AOI, if not already on disk.

    Args:
        AOI_File (str): path of vector AOI file
        out_file (str): output GeoTIFF path
        margin_deg (float, optional): margin around the AOI bounds in degrees. Defaults to 0.01.

    Returns:
        str: path of the WorldCover GeoTIFF
    """
    if not os.path.exists(out_file):
        lon_min, lat_min, lon_max, lat_max = gpd.read_file(AOI_File).to_crs(epsg=4326).total_bounds
        bounds = (lon_min - margin_deg, lat_min - margin_deg, lon_max + margin_deg, lat_max + margin_deg)
        tiles = sorted({worldcover_tile(lon, lat) for lon in np.arange(bounds[0], bounds[2] + 3, 3).clip(max=bounds[2])
                                                  for lat in np.arange(bounds[1], bounds[3] + 3, 3).clip(max=bounds[3])})
        sources = [rasterio.open('/vsicurl/' + WORLDCOVER_URL.format(tile)) for tile in tiles]
        landcover, transform = merge(sources, bounds=bounds)
        profile = sources[0].profile
        for src in sources:
            src.close()
        profile.update(driver='GTiff', height=landcover.shape[1], width=landcover.shape[2], transform=transform,
                       compress='deflate', tiled=False)
        profile.pop('blockxsize', None), profile.pop('blockysize', None)
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        with rasterio.open(out_file, 'w', **profile) as dst:
            dst.write(landcover)
    return out_file


def water_fraction(sm_polygons:gpd.GeoSeries, worldcover_file:str)->np.array:
    """Fraction of each grid cell covered by permanent water bodies.

    Args:
        sm_polygons (gpd.GeoSeries): borders of the grid cells in WGS84 (see create_grid_xy)
        worldcover_file (str): WorldCover GeoTIFF (see get_worldcover)

    Returns:
        np.array: 1D array [grid cells] of water fractions (0-1)
    """
    with rasterio.open(worldcover_file) as src:
        water = src.read(1) == WATER_CLASS
        transform = src.transform
    fractions = np.zeros(len(sm_polygons))
    for ind, bounds in enumerate(sm_polygons.bounds.values):
        window = from_bounds(*bounds, transform=transform).round_offsets().round_lengths()
        cell = water[max(window.row_off, 0):window.row_off + window.height, max(window.col_off, 0):window.col_off + window.width]
        fractions[ind] = cell.mean() if cell.size else 0
    return fractions
