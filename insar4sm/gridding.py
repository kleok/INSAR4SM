#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import geopandas as gpd
import pandas as pd
import rasterio
from shapely.geometry import Point
import numpy as np
import os
import pyproj

def WGS84_to_UTM(lon:float, lat:float)->str:
    """Finds the best WGS84 UTM projection given a lon, lat.

    Args:
        lon (float): longitude
        lat (float): latitude

    Returns:
        str: the code of CRS (corresponds to WGS84 UTM projection)
    """
    utm_zone = int(np.floor((lon + 180) / 6) + 1)
    if lat>0:
        hemisphere='north'
    else:
        hemisphere='south'
    utm_crs_str = '+proj=utm +zone={} +{} +ellps=WGS84 +datum=WGS84 +units=m +no_defs'.format(utm_zone,hemisphere)

    utm_crs_epsg = pyproj.CRS(utm_crs_str).to_epsg()
    
    return utm_crs_epsg

def create_grid_xy(Outdir:str, AOI:str, res:int=250)->tuple[gpd.GeoDataFrame,gpd.GeoDataFrame]:
    """Creates a grid (centroids and borders) for given AOI and grid size.

    Args:
        Outdir (str): the output directory that SM_point_grid and sm_polygons_latlon will be saved
        AOI (str): the path of vector AOI file
        res (int, optional): the grid size in meters. Defaults to 250.

    Returns:
        tuple[gpd.GeoDataFrame,gpd.GeoDataFrame]: centroids of each grid cell, borders of each grid cell
    """
    # Read geometry information
    AOI_wgs84 = gpd.read_file(AOI)
    AOI_wgs84.crs = "EPSG:4326"
    
    polygon = AOI_wgs84.explode(index_parts=False).geometry.iloc[0]

    # find the UTM projection to work with
    utm_crs_epsg = WGS84_to_UTM(polygon.centroid.x, polygon.centroid.y)

    # change WGS84 (lat/lon) to WGS UTM projected reference system (x,y)
    polygon = AOI_wgs84.to_crs(epsg=utm_crs_epsg).iloc[0].geometry
    
    
    # Read x,y information
    minx, miny, maxx, maxy = polygon.bounds

    # construct a rectangular mesh
    points = []
    for y in np.arange(minx+res/2, maxx, res):
        for x in np.arange(miny+res/2, maxy, res):
            points.append(Point((round(y,2), round(x,2))))
    
    # validate if each point falls inside AOI shape 
    valid_points = [pt for pt in points if pt.within(polygon)]
    
    SM_ID = np.arange(len(valid_points))
    sm_points_xy = gpd.GeoDataFrame(SM_ID,
                                    geometry = valid_points,
                                    columns=['ID'])

    sm_points_xy.crs = "EPSG:{}".format(utm_crs_epsg)
    # create polygons from points
    sm_polygons_xy = sm_points_xy.buffer(res, cap_style = 3)

    # reproject back to WGS84 lat/lon
    
    sm_polygons_latlon = sm_polygons_xy.to_crs(epsg=4326)
    sm_points_latlon = sm_polygons_latlon.centroid
    
    sm_points_latlon.to_file(os.path.join(Outdir,'SM_point_grid.geojson'),
                             driver='GeoJSON')
    
    sm_polygons_latlon.to_file(os.path.join(Outdir,'SM_polygons.geojson'),
                             driver='GeoJSON')
    
    return sm_points_latlon, sm_polygons_latlon
def sm_to_geotiff(sm_df:pd.DataFrame, sm_points:gpd.GeoSeries, res:int, out_file:str)->str:
    """Writes the soil moisture estimations of the grid cells to a GeoTIFF (one band per SAR acquisition).

    The grid cells (see create_grid_xy) are centred on a UTM lattice with spacing res, so each
    cell becomes one pixel of res meters.

    Args:
        sm_df (pd.DataFrame): soil moisture estimations, one row per grid cell (same order as sm_points) and one column per acquisition
        sm_points (gpd.GeoSeries): centroids of the grid cells in WGS84
        res (int): the grid size in meters (as in create_grid_xy)
        out_file (str): path of output GeoTIFF

    Returns:
        str: path of output GeoTIFF
    """
    utm_crs_epsg = WGS84_to_UTM(sm_points.union_all().centroid.x, sm_points.union_all().centroid.y)
    points_xy = sm_points.to_crs(epsg=utm_crs_epsg)
    x0, y1 = points_xy.x.min(), points_xy.y.max()
    cols = np.round((points_xy.x.values - x0) / res).astype(int)
    rows = np.round((y1 - points_xy.y.values) / res).astype(int)

    sm_maps = np.full((sm_df.shape[1], rows.max() + 1, cols.max() + 1), np.nan, dtype=np.float32)
    sm_maps[:, rows, cols] = sm_df.values.T
    transform = rasterio.transform.from_origin(x0 - res / 2, y1 + res / 2, res, res)
    with rasterio.open(out_file, 'w', driver='GTiff', width=sm_maps.shape[2], height=sm_maps.shape[1],
                       count=sm_maps.shape[0], dtype='float32', crs='EPSG:{}'.format(utm_crs_epsg),
                       transform=transform, nodata=np.nan, compress='deflate') as dst:
        dst.write(sm_maps)
        for band, date in enumerate(sm_df.columns, start=1):
            dst.set_band_description(band, str(date))
    return out_file
