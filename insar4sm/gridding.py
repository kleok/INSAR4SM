#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import geopandas as gpd
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
    
    polygon = AOI_wgs84.explode().geometry[0][0]

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