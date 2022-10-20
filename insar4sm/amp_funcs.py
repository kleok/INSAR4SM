#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import os
from osgeo import gdal
import numpy as np
import pandas as pd

def runCalamp(inputDS:str, output_dir:str)-> tuple[pd.DataFrame, np.array]:
    """ Amplitude calibration factors calculation.
        - Read each slc image
        - Calculate the number of valid pixels (exclude mask, zero values)
        - Calculate sum of amplitudes of valid pixels
        - Calculate mean value of valid pixels which is the amplitude calibration factor
        - Save Amplitude calibration factor to pandas dataframe with datetime and two columns (Amplitude_factors, Datetime)
        - Saves amplitude calibration factors in csv in disk
        - Saves SLC SAR stack to npy file in disk

    Args:
        inputDS (str): The path of vrt file of SLC SAR stack
        output_dir (str): the output directory that Amplitude calibration factors and SLC SAR stack in npy format will be saved

    Returns:
        tuple[pd.DataFrame, np.array]: Amplitude calibration factors, SLC SAR stack
    """
    stack_vrt = gdal.Open( inputDS )
    Amp_cal_factors={}
    Dates=[]
    for band in range( stack_vrt.RasterCount ):
        band += 1
        slc_date=stack_vrt.GetRasterBand(band)
        srcband_cpx = stack_vrt.GetRasterBand(band).ReadAsArray()
        
        srcband_amp=np.abs(srcband_cpx)
        srcband_amp[srcband_amp==0]=np.nan
        Amp_cal_factor=np.nanmean(srcband_amp)
        Amp_cal_factors[slc_date.GetMetadata('slc')['Date']]=Amp_cal_factor
        Dates.append(slc_date.GetMetadata('slc')['Date'])
    
    # saves Calibation factors   
    Amp_cal_factors_df= pd.DataFrame.from_dict(Amp_cal_factors, orient='index', columns=['Amplitude_factors'])
    Amp_cal_factors_df.index = pd.to_datetime(Amp_cal_factors_df.index)
    Amp_cal_factors_df['Datetime']=Amp_cal_factors_df.index
    Amp_cal_factors_df.to_csv(os.path.join(output_dir, 'AmpCalFactors.csv'))
    
    # get slc data and save to npy file
        
    slc_stack=stack_vrt.ReadAsArray()
        
    with open('{}/slc_stack.npy'.format(output_dir), 'wb') as f:
        np.save(f, slc_stack)

    return Amp_cal_factors_df, slc_stack