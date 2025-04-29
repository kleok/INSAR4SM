#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import geopandas as gpd
import rioxarray
import numpy as np
import xarray as xr
from shapely.geometry import box
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

def convert_to_df(meteo_xr_filename:str, AOI_File:str) -> pd.DataFrame:
    """Convert xarray meteo dataset to pandas dataframe at daily basis.

    TODO: 
        Check if the netCDF4.num2date functionality can output datetime.datetime object
        
    Args:
        meteo_xr_filename (str): the path of provided meteo dataset
        AOI_File (str): the path of vector AOI

    Returns:
        pd.DataFrame: meteo_df has three columns (Datetimes, tp__m, skt__K) and contains daily information.
    """
    meteo_xr = xr.open_dataset(meteo_xr_filename, decode_coords='all')
    meteo_xr = meteo_xr.rio.write_crs("EPSG:4326")

    # Resolution (assumed 0.1°)
    res = 0.1
    half = res / 2

    # Get all grid cell polygons
    x_vals = meteo_xr.x.values
    y_vals = meteo_xr.y.values

    pixel_polygons = []
    pixel_centers = []

    for y in y_vals:
        for x in x_vals:
            pixel = box(x - half, y - half, x + half, y + half)
            pixel_polygons.append(pixel)
            pixel_centers.append((x, y))

    # Create GeoDataFrame of all grid pixels
    pixel_gdf = gpd.GeoDataFrame(pixel_centers, columns=["x", "y"], geometry=pixel_polygons, crs="EPSG:4326")

    # Intersect with AOI
    intersecting_pixels = pixel_gdf[pixel_gdf.intersects(gpd.read_file(AOI_File).unary_union)]

    # Get matching x/y values
    x_selected = np.unique(intersecting_pixels["x"])
    y_selected = np.unique(intersecting_pixels["y"])

    # Clip dataset
    ds_clipped = meteo_xr.sel(x=x_selected, y=y_selected)

    # mean values
    ds_mean = ds_clipped.mean(dim=["x", "y"])

    # to dataframe
    df = ds_mean.to_dataframe().reset_index()

    # Set time as index and drop from columns
    df.set_index('time', inplace=True)

    # Drop 'spatial_ref' column if it exists
    if 'spatial_ref' in df.columns:
        df.drop(columns=['spatial_ref'], inplace=True)

    return df