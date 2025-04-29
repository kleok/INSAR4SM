import os
import itertools
import geopandas as gpd
from shapely.geometry import Point
import xarray as xr
import warnings
warnings.simplefilter(action='ignore')

def create_ERA5_cell_polygons(ERA5_merged_filename, AOI_filename, ERA5_dir, intersection_percent_thres, grid_size = 0.05):
        
    merged_ERA5_xr = xr.open_dataset(ERA5_merged_filename)

    AOI_geometry = gpd.read_file(AOI_filename).geometry
    ERA5_lats = merged_ERA5_xr.variables['y'][:]
    ERA5_lons = merged_ERA5_xr.variables['x'][:]

    ERA5_centers = list(itertools.product(ERA5_lats.data, ERA5_lons.data))
    ERA5_grid_cells = {}

    for ERA5_center in ERA5_centers:

        grid_center_lat, grid_center_lon = ERA5_center
        centroid_gdf = gpd.GeoDataFrame(index=[0], crs="EPSG:4326", geometry=[Point(grid_center_lon, grid_center_lat)])
        grid_cell_gdf = centroid_gdf.buffer(grid_size, cap_style = 3)
        if AOI_geometry.intersects(grid_cell_gdf)[0]:
            intersection_percent = (grid_cell_gdf.intersection(AOI_geometry).area/grid_cell_gdf.area)[0]*100
            
            if intersection_percent > intersection_percent_thres:
                cell_name = 'ERA5_grid_cell_lat_{:.2f}_lon_{:.2f}'.format(grid_center_lat, grid_center_lon)
                filename = os.path.join(os.path.join(ERA5_dir,'{}.geojson'.format(cell_name)))
                
                ERA5_grid_cells[cell_name] = filename
                grid_cell_gdf.to_file(filename, driver="GeoJSON")

def get_missing_time_spans(request_start, request_end, downloaded_start, downloaded_end):
    """
    Identifies the time spans that need to be downloaded by comparing the requested time range 
    with an already downloaded time range.

    Parameters:
        request_start (pd.Timestamp): Start of the requested time range.
        request_end (pd.Timestamp): End of the requested time range.
        downloaded_start (pd.Timestamp): Start of the already downloaded time range.
        downloaded_end (pd.Timestamp): End of the already downloaded time range.

    Returns:
        List[Tuple[pd.Timestamp, pd.Timestamp]]: List of non-overlapping time spans 
        that need to be downloaded.
    """
    missing_spans = []

    # Time span before the downloaded range
    if request_start < downloaded_start:
        missing_spans.append((request_start, min(request_end, downloaded_start)))

    # Time span after the downloaded range
    if request_end > downloaded_end:
        missing_spans.append((max(downloaded_start, downloaded_end), request_end))

    return missing_spans