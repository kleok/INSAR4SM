#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cdsapi
import datetime
import pandas as pd
import geopandas as gpd
import numpy as np

def retrieve_ERA5_land_data(ERA5_variables:list,
                            year_str:str,
                            month_str:str,
                            days_list:list,
                            time_list:list,
                            bbox_cdsapi:list,
                            export_filename:str,
                            overwrite = False )->str: 
    """
    Gets data from ERA-5 (ECMWF) reanalysis-era5-single-levels (hourly ~9km) resolution 0.1 degrees
    Args:
        ERA5_variables (list): List of ERA5 variables e.g. 'volumetric_soil_water_layer_1'
        year_str (str): Year
        month_str (str): Month
        days_list (list): days
        time_list (list): times (UTC)
        bbox_cdsapi (list): geographical borders e.g. [ 41.23, 24.5, 40.79, 25.9,]
        export_filename (str): full path of dataset
    Returns:
        export_filename (string): Full path of dataset
    References:
        https://confluence.ecmwf.int/display/CKB/ERA5-Land%3A+data+documentation
        https://essd.copernicus.org/articles/13/4349/2021/

    """

    if overwrite or not os.path.exists(export_filename):

        c = cdsapi.Client()

        request_dict = {
                'variable': ERA5_variables,
                'year': year_str,
                'month': month_str,
                'day': days_list,
                'time': time_list,
                'area': bbox_cdsapi,
                # 'format': 'netcdf',
                'format':'grib',
                # "download_format": "unarchived",
                "download_format": "zip", }
        print(request_dict)
        
        c.retrieve('reanalysis-era5-land', request_dict, export_filename)
        
    return export_filename


def Get_ERA5_data(ERA5_variables:list,
                 start_datetime:datetime.datetime,
                 end_datetime:datetime.datetime,
                 AOI_filename:str,
                ERA5_dir:str,) -> pd.DataFrame:
    
    """Downloads ERA5 datasets between two given dates.
    
    ToDo: There is an issue if the scale_factor and offset changes across 
          nc files to be merged.

    Args:
        ERA5_variables (list): list of ERA5 variables e.g. ['total_precipitation',]
        start_datetime (datetime.datetime): Starting Datetime e.g.  datetime.datetime(2021, 12, 2, 0, 0)
        end_datetime (datetime.datetime): Ending Datetime e.g.  datetime.datetime(2022, 2, 8, 0, 0)
        AOI_filename (str): vector polygon file of the AOI.
        ERA5_dir (str): Path that ERA5 data will be saved.
    Returns:
        ERA5_datasets (list): ERA5 downloaded datasets
    """

    lon_min, lat_min, lon_max, lat_max = np.squeeze(gpd.read_file(AOI_filename).bounds.values)
    half_res = 0.1
    bbox_cdsapi =  [np.ceil(lat_max*10)/10 + half_res,
                    np.floor(lon_min*10)/10 - half_res,
                    np.floor(lat_min*10)/10 - half_res,
                    np.ceil(lon_max*10)/10 + half_res,]

    request_start_time = start_datetime.strftime("%Y%m%dT%H%M%S")
    request_end_time = end_datetime.strftime("%Y%m%dT%H%M%S")
    bbox_cdsapi_str='_'.join(str(round(e,3)) for e in bbox_cdsapi)

    ERA5_datasets = []

    df = pd.date_range(start=start_datetime, end=end_datetime, freq='H').to_frame(name='Datetime')
    
    df['year'] = df['Datetime'].dt.year
    df["year_str"] = ['{:02d}'.format(year) for year in df['year']]
    
    df['month'] = df['Datetime'].dt.month
    df["month_str"] = ['{:02d}'.format(month) for month in df['month']]
    
    df['day'] = df['Datetime'].dt.day
    df["day_str"] = ['{:02d}'.format(day) for day in df['day']]

    df['hour'] = df['Datetime'].dt.hour
    df["hour_str"] = ['{:02d}'.format(hour) for hour in df['hour']]
    
    # for the last datetime we do a single request
    
    last_day_df = df.sort_values(by = 'Datetime').iloc[-1]
    last_day_times = np.arange(last_day_df.hour+1)
    last_day_times_str = ['{:02d}'.format(last_day_time) for last_day_time in last_day_times]
    #print("Downloading precipitation for the flood date: {}".format(last_day_df.Datetime.strftime("%Y-%m-%d")))

    export_filename = os.path.join(ERA5_dir,'ERA5_Last_day_{}_{}_{}.nc'.format(request_start_time,
                                                                          request_end_time,
                                                                          bbox_cdsapi_str))

    last_day_dataset = retrieve_ERA5_land_data(ERA5_variables = ERA5_variables,
                                                year_str = last_day_df.year_str,
                                                month_str = last_day_df.month_str,
                                                days_list = last_day_df.day_str,
                                                time_list = last_day_times_str,
                                                bbox_cdsapi = bbox_cdsapi,
                                                export_filename = export_filename)
    
    ERA5_datasets.append(last_day_dataset)
    
    # For each month we do a request
    df2 = df.truncate(after=datetime.datetime(last_day_df.year, last_day_df.month, last_day_df.day, 0)).iloc[:-1]
    
    for year in np.unique(df2["year_str"].values):
        df_year = df2[df2['year_str']==year]
        year_request = year
        for month in np.unique(df_year["month_str"].values):
            month_request = month
            print("Downloading {} for month {} of year {} ...".format(ERA5_variables, month_request, year_request))
            df_month = df_year[df_year['month_str']==month]
            days_request = np.unique(df_month['day_str']).tolist()
            hours_request = np.unique(df_month['hour_str']).tolist()
            
            export_filename = os.path.join(ERA5_dir,'ERA5_{}_{}_{}_{}_{}.nc'.format(month_request,
                                                                                    year_request,
                                                                                    request_start_time,
                                                                                    request_end_time,
                                                                                    bbox_cdsapi_str))

            monthly_dataset = retrieve_ERA5_land_data(ERA5_variables = ERA5_variables,
                                                    year_str = year_request,
                                                    month_str = month_request,
                                                    days_list = days_request,
                                                    time_list = hours_request,
                                                    bbox_cdsapi = bbox_cdsapi,
                                                    export_filename = export_filename)
            
            ERA5_datasets.append(monthly_dataset)

    return ERA5_datasets