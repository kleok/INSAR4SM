#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import geopandas as gpd
import datetime 
import netCDF4
import numpy as np
import cftime
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

def cftime_to_datetime(cfdatetime:cftime.datetime)->datetime.datetime:
    """coverts cftime datetime object to datetime.datetime object

    Args:
        cfdatetime (cftime.datetime): provided cftime datetime object

    Returns:
        datetime.datetime: calculated datetime.datetime object
    """
    year=cfdatetime.year
    month=cfdatetime.month
    day=cfdatetime.day
    hour=cfdatetime.hour
    minute=cfdatetime.minute
    second=cfdatetime.second
    return datetime.datetime(year,month,day,hour,minute,second)

def convert_to_df(meteo_file:str, SM_AOI:str, ERA5_flag:bool)->pd.DataFrame:
    """Convert meteo dataset to pandas dataframe at daily basis.

    TODO: 
        Check if the netCDF4.num2date functionality can output datetime.datetime object
        
    Args:
        meteo_file (str): the path of provided meteo dataset
        SM_AOI (str): the path of vector AOI
        ERA5_flag (bool): True if it is an ERA5-Land dataset

    Returns:
        pd.DataFrame: meteo_df has three columns (Datetimes, tp__m, skt__K) and contains daily information.
    """
    if ERA5_flag:
        # get approximate values of location of ISMN stations based on consider buffer
        p_lon = gpd.read_file(SM_AOI)['geometry'].centroid[0].x
        p_lat = gpd.read_file(SM_AOI)['geometry'].centroid[0].y
        
        # get the indices of the most representative ERA5 pixel that is going to be used
        
        ERA5_data=netCDF4.Dataset(meteo_file)
        ERA5_variables = list(ERA5_data.variables.keys())
        
        ERA5_lons = ERA5_data.variables['longitude'][:].data
        ERA5_lats = ERA5_data.variables['latitude'][:].data
    
        ERA_pixel_ind1 = np.argmin(np.abs(ERA5_lats-p_lat))
        ERA_pixel_ind2 = np.argmin(np.abs(ERA5_lons-p_lon))
      
        # create a dataframe with the ERA5 variables for the specific ISMN station
        df_dict={}
        
        for ERA5_variable in ERA5_variables:
            
            if ERA5_variable in ['longitude',  'latitude']:
                pass
            elif ERA5_variable=='time':
                time_var=ERA5_data.variables[ERA5_variable]
                t_cal = ERA5_data.variables[ERA5_variable].calendar
                dtime = netCDF4.num2date(time_var[:],time_var.units, calendar = t_cal)
                dtime_datetime=[cftime_to_datetime(cfdatetime) for cfdatetime in dtime.data]
                df_dict['Datetimes']=dtime_datetime
                
            elif ERA5_variable!='expver':
                temp_name=ERA5_variable+'__'+ERA5_data[ERA5_variable].units
                temp_dataset=ERA5_data[ERA5_variable][:][:,ERA_pixel_ind1,ERA_pixel_ind2]
                df_dict[temp_name]=np.squeeze(temp_dataset)
            else:
                pass
             
        # create a dataframe
        meteo_df = pd.DataFrame(df_dict)
        meteo_df.index = pd.to_datetime(meteo_df['Datetimes'])   
    else:
        
        meteo_df = pd.read_csv(meteo_file)
        
        assert 'Datetimes' in meteo_df.columns
        assert 'tp__m' in meteo_df.columns
        assert 'skt__K' in meteo_df.columns

        meteo_df.index = pd.to_datetime(meteo_df['Datetimes']) 
        meteo_df = meteo_df.resample('D').sum()

    return meteo_df