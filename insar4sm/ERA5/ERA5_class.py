#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, glob
import datetime
import geopandas as gpd
import pandas as pd
from shapely import force_2d
import xarray as xr

# ERA5 functionalities
from insar4sm.ERA5.Download_ERA5_land import Get_ERA5_data
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
        if  (pd.to_datetime(datetime.datetime.now()) - self.request_end_datetime).days < 6 :
            self.ERA5_last_datetime = pd.to_datetime(datetime.datetime.now() - datetime.timedelta(days=6))
        else:
            self.ERA5_last_datetime = self.request_end_datetime

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
        if os.path.exists(self.era5_merged_filename):
            downloaded_data = xr.open_dataset(self.era5_merged_filename)
            downloaded_start_datetime = pd.to_datetime(downloaded_data.time[0].values)
            downloaded_end_datetime = pd.to_datetime(downloaded_data.time[-1].values)
            self.time_spans = get_missing_time_spans(self.request_start_datetime,
                                                self.request_end_datetime,
                                                downloaded_start_datetime,
                                                downloaded_end_datetime)
        else:
            self.time_spans = [(self.request_start_datetime, self.request_end_datetime)]

        return self.time_spans

    def download_ERA5_data(self):
        for time_span in self.time_spans:

            start_t, end_t = time_span
            print('Downloading data for {}'.format(time_span))

            if end_t < self.ERA5_last_datetime:
                print("ERA5-Land data will be downloaded from {} to {}".format(start_t, end_t))

            elif start_t < self.ERA5_last_datetime:
                
                if os.path.exists(self.era5_merged_filename):
                    # update request times
                    downloaded_era5_xr = xr.open_dataset(self.era5_merged_filename)
                    era5_time_spans = get_missing_time_spans(request_start = start_t,
                                                            request_end = self.ERA5_last_datetime,
                                                            downloaded_start = pd.to_datetime(downloaded_era5_xr.time[0].values),
                                                            downloaded_end = pd.to_datetime(downloaded_era5_xr.time[-1].values))
                    # get ERA5 data
                    for era5_time_span in era5_time_spans:
                        print("ERA5-Land data will be downloaded from {} to {}".format(era5_time_span[0], era5_time_span[1]))
                        ERA5_datasets = Get_ERA5_data(ERA5_variables = self.ERA5_variables,
                                                        start_datetime = era5_time_span[0],
                                                        end_datetime = era5_time_span[1],
                                                        AOI_filename = self.AOI_filename,
                                                        ERA5_dir = self.ERA5_dir)
                        
                    # append new data to era5_merged_file and overwrite file
                    new_ERA5_data  = merge_ERA5_land_datasets(ERA5_datasets, None)
                    updated_era5_xr = xr.concat([new_ERA5_data,downloaded_era5_xr], dim='time').sortby('time').copy()
                    downloaded_era5_xr.close()
                    os.remove(self.era5_merged_filename)
                    updated_era5_xr.to_netcdf(self.era5_merged_filename)

                else:
                    print("ERA5-Land data will be downloaded from {} to {}".format(start_t, self.ERA5_last_datetime))
                    # get ERA5 data
                    ERA5_datasets = Get_ERA5_data(ERA5_variables = self.ERA5_variables,
                                                    start_datetime = start_t,
                                                    end_datetime = self.ERA5_last_datetime,
                                                    AOI_filename = self.AOI_filename,
                                                    ERA5_dir = self.ERA5_dir)
                    
                    # merge dataset and save to file
                    merge_ERA5_land_datasets(ERA5_datasets = glob.glob(os.path.join(self.ERA5_dir,'ERA5_*.nc')),
                                            ERA5_merged_filename = self.era5_merged_filename)
                    

    def create_ERA5_polygons(self, intersection_percent_thres = 0):
        create_ERA5_cell_polygons(self.era5_merged_filename, self.AOI_filename, self.ERA5_dir, intersection_percent_thres)