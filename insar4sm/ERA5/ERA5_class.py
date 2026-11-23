#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import datetime
import geopandas as gpd
import pandas as pd
from shapely import force_2d
import xarray as xr

# ERA5 functionalities
from insar4sm.ERA5.Download_ERA5_land import Get_ERA5_data, get_bbox_cdsapi
from insar4sm.ERA5.Preprocess_ERA5_data import merge_ERA5_land_datasets
from insar4sm.ERA5.utils import get_missing_time_spans, create_ERA5_cell_polygons

class ERA5:
    """
    Processing workflow for preparing ERA5-Land meteorological data.
    """
    
    def __init__(self, params_dict:dict):

        # Project Definition
        self.projectfolder  = params_dict['projectfolder']

        # temporal information
        self.request_start_datetime = pd.to_datetime(datetime.datetime.strptime(params_dict['time_start'],'%Y%m%dT%H%M%S'))
        self.request_end_datetime   = pd.to_datetime(datetime.datetime.strptime(params_dict['time_end'],'%Y%m%dT%H%M%S'))
        # ERA5-Land is available with ~5 days latency. Floored to the day, so reruns on the same day request the same data
        self.ERA5_last_datetime = min(self.request_end_datetime,
                                      pd.to_datetime(datetime.datetime.now() - datetime.timedelta(days=6)).floor('D'))

        # spatial information
        self.AOI_filename       = params_dict['AOI_File']
        self.AOI_geometry       = gpd.read_file(self.AOI_filename)['geometry'].explode(index_parts=True)
        self.AOI_polygon        = gpd.read_file(self.AOI_filename)['geometry'].explode(index_parts=True).iloc[0]
        self.AOI_polygon        = force_2d(self.AOI_polygon)
        
        # Data access and processing
        self.ERA5_variables       = params_dict['ERA5_variables']
        
        #-------------------------------------------------------------------
        #------->             Creating directory structure
        #-------------------------------------------------------------------
        self.ERA5_dir = os.path.join(self.projectfolder,'ERA5')
        self.era5_merged_filename =  os.path.join(self.ERA5_dir,'merged_ERA5.nc')

        for dir in [self.ERA5_dir]:
            if not os.path.exists(dir): os.makedirs(dir)
    

    def get_download_dates(self):
        # identify which dates are already downloaded
        self.time_spans = [(self.request_start_datetime, self.ERA5_last_datetime)]
        self.merged_is_valid = False
        if os.path.exists(self.era5_merged_filename):
            with xr.open_dataset(self.era5_merged_filename) as downloaded_data:
                # the merged file is reused only if it covers the ERA5-Land area of the current AOI
                # and holds hourly tp
                lat_max, lon_min, lat_min, lon_max = get_bbox_cdsapi(self.AOI_filename)
                self.merged_is_valid = np.allclose([downloaded_data.y.max(), downloaded_data.x.min(),
                                                    downloaded_data.y.min(), downloaded_data.x.max()],
                                                   [lat_max, lon_min, lat_min, lon_max], atol=0.01)
                self.merged_is_valid &= downloaded_data.tp.attrs.get('units') == 'mm/hour'
                if self.merged_is_valid:
                    self.time_spans = get_missing_time_spans(self.request_start_datetime,
                                                             self.ERA5_last_datetime,
                                                             pd.to_datetime(downloaded_data.time[0].values),
                                                             pd.to_datetime(downloaded_data.time[-1].values))
                else:
                    print('{} covers a different area or is from an older version, it will be replaced'.format(self.era5_merged_filename))

        return self.time_spans

    def download_ERA5_data(self):
        if not self.time_spans:
            print('ERA5-Land data from {} to {} are already downloaded'.format(self.request_start_datetime,
                                                                               self.ERA5_last_datetime))
            return

        ERA5_datasets = []
        for start_t, end_t in self.time_spans:
            print("ERA5-Land data will be downloaded from {} to {}".format(start_t, end_t))
            # one hour earlier, as hourly tp is the difference from the previous hour's accumulation
            ERA5_datasets += Get_ERA5_data(ERA5_variables = self.ERA5_variables,
                                           start_datetime = start_t - pd.Timedelta(hours=1),
                                           end_datetime = end_t,
                                           AOI_filename = self.AOI_filename,
                                           ERA5_dir = self.ERA5_dir)

        merged_era5_xr = merge_ERA5_land_datasets(ERA5_datasets, None).drop_vars('spatial_ref', errors='ignore')
        if self.merged_is_valid:
            # append new data to the merged file, keeping each time once
            with xr.open_dataset(self.era5_merged_filename) as downloaded_era5_xr:
                merged_era5_xr = xr.concat([downloaded_era5_xr.load(), merged_era5_xr], dim='time')
            _, unique_indices = np.unique(merged_era5_xr['time'].values, return_index=True)
            merged_era5_xr = merged_era5_xr.isel(time=unique_indices)

        # write to a temporary file, so an interrupted write does not leave a broken merged file
        tmp_filename = self.era5_merged_filename.replace('.nc', '_tmp.nc')
        merged_era5_xr.to_netcdf(tmp_filename)
        os.replace(tmp_filename, self.era5_merged_filename)

    def create_ERA5_polygons(self, intersection_percent_thres = 0):
        create_ERA5_cell_polygons(self.era5_merged_filename, self.AOI_filename, self.ERA5_dir, intersection_percent_thres)