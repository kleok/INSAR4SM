#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import pandas as pd
import numpy as np

def find_dry_SARs(meteo_df:pd.DataFrame,
                  slc_datetimes:list,
                  days_back:int,
                  orbit_time:str,
                  lowest_temp_K:float,
                  tp_quantile:float)->tuple[str,list,pd.DataFrame]:
    
    """Finds SAR images that are related to low precipitation activity (low SSM conditions) and not low temperature (snow-free).

    TODO:
        Make sure that daily and hourly meteo_df works well with this functionality.

    Args:
        meteo_df (pd.DataFrame): Contains meteorological data at daily basis. We have 4 columns (tp, skt__K, time, swvl1)
        slc_datetimes (list): list of (datetime.datetime objects) SAR acquisition datetimes
        days_back (int, optional): number of days to go back in order to accumulate precipitation.
        orbit_time (str, optional): Time of the SAR pass.

    Returns:
        tuple[str,list,pd.DataFrame]: selected dry SAR acquisition, datetimes of other potential dry SAR acquisitions, meteorological data for SAR acuqisition datetimes
    """
    # calculate accumulated precipitation
    
    df2 = (meteo_df['tp'].shift().rolling(window=days_back, min_periods=1).sum().reset_index())
    df2.index=pd.to_datetime(meteo_df.index)
    
    meteo_df['tp'] = df2['tp']
    meteo_df.index = pd.to_datetime(meteo_df.index)
    meteo_df = meteo_df.at_time(orbit_time)

    # change datetime to date format
    meteo_df.index = meteo_df.index.date

    # select only the datetimes that we have a Sentinel-1 acquisition
    test = pd.DataFrame(np.zeros(len(slc_datetimes)), columns=['test'])
    test.index = slc_datetimes
    meteo_sel_df = meteo_df.join(test, how='outer').dropna()

    # We dont want accumulated precipiration (24 days) bigger that 1mm
    precipitation_thres = np.quantile(meteo_sel_df['tp'].values, tp_quantile)
    precipitation_mask = meteo_sel_df['tp'] < precipitation_thres
    
    # We dont want temperature below 273.15 Kelvin
    temp_mask = meteo_sel_df['skt'] > lowest_temp_K
    
    Final_mask = precipitation_mask & temp_mask

    if not np.any(Final_mask): 
        print('meteo data :\n')
        print(meteo_sel_df)
        print("precipitation_threshold")
        print(precipitation_thres)
        print('Precipitation mask :\n')
        print(precipitation_mask)
        print('Temperature mask :\n')
        print(temp_mask)

    assert(np.any(Final_mask))
    # selecting the first image as driest one.
    dry_datetime = Final_mask.where(Final_mask == True).first_valid_index()
    
    dry_date_str = dry_datetime.strftime('%Y%m%d')
    
    return dry_date_str, slc_datetimes[Final_mask], meteo_sel_df