#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rasterio
from rasterio.windows import Window

def get_soilgrids_value(soilgrid_tiff:str, lon:float, lat:float)->float:
    """Extracts the value from given tiff file at lon, lat

    Args:
        soilgrid_tiff (str): The path of soilgrids tiff file
        lon (float): Longitude (WGS84)
        lat (float): Latitude (WGS84)

    Returns:
        float: The extracted percentage value (from 0-100)
    """
    with rasterio.open(soilgrid_tiff) as src:
        meta = src.meta
        # Use the transform in the metadata and your coordinates
        rowcol = rasterio.transform.rowcol(meta['transform'], xs=lon, ys=lat)
        value = src.read(1, window=Window(rowcol[1], rowcol[0], 1, 1))
        # convert g/kg to percent
        value_pct = float(value/10)
    return value_pct
