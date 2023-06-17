#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import pandas as pd
import numpy as np

def find_dry_SARs(meteo_df:pd.DataFrame,
                  slc_datetimes:list,
                  days_back:int = 24,
                  orbit_time:str ='02:00:00',
                  lowest_temp_K:float = 273.15,
                  highest_acc_rain_m:float = 0.001)->tuple[str,list,pd.DataFrame]:
    """Finds SAR images that are related to low precipitation activity (low SSM conditions) and not low temperature (snow-free).

    TODO:
        Make sure that daily and hourly meteo_df works well with this functionality.

    Args:
        meteo_df (pd.DataFrame): Contains meteorological data at daily basis. We have three columns (tp__m, skt__K, Datetimes)
        slc_datetimes (list): list of (datetime.datetime objects) SAR acquisition datetimes
        days_back (int, optional): number of days to go back in order to accumulate precipitation. Defaults to 24.
        orbit_time (str, optional): Time of the SAR pass. Defaults to '02:00:00'.

    Returns:
        tuple[str,list,pd.DataFrame]: selected dry SAR acquisition, datetimes of other potential dry SAR acquisitions, meteorological data for SAR acuqisition datetimes
    """
    # calculate accumulated precipitation
    
    df2 = (meteo_df['tp__m'].shift().rolling(window=days_back, min_periods=1).sum().reset_index())
    df2.index=pd.to_datetime(meteo_df.index)
    
    meteo_df['tp__m'] = df2['tp__m']
    meteo_df = meteo_df.at_time(orbit_time)
    
    # change datetime to date format
    meteo_df.index = pd.to_datetime(meteo_df['Datetimes']).dt.date

    # select only the datetimes that we have a Sentinel-1 acquisition
    test = pd.DataFrame(np.zeros(len(slc_datetimes)), columns=['test'])
    test.index = slc_datetimes
    meteo_sel_df = meteo_df.join(test, how='outer')
    meteo_sel_df = meteo_sel_df.dropna()

    # We dont want accumulated precipiration (24 days) bigger that 1mm
    precipitation_mask = meteo_sel_df['tp__m'] < highest_acc_rain_m
    
    # We dont want temperature below 273.15 Kelvin
    temp_mask = meteo_sel_df['skt__K'] > lowest_temp_K
    
    Final_mask = precipitation_mask & temp_mask
    
    # selecting the first image as driest one.
    dry_datetime = Final_mask.where(Final_mask == True).first_valid_index()
    
    dry_date_str = dry_datetime.strftime('%Y%m%d')
    
    return dry_date_str, slc_datetimes[Final_mask], meteo_sel_df