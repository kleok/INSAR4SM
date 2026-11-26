#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, glob
import numpy as np
from datetime import datetime
import shutil
from insar4sm.amp_funcs import runCalamp

def calc_int_amp(input_dir:str, output_dir:str, primary_date:str)-> int:
    """Performs interferometric processing
    
        - Restructures the Topstack datasets
        - Finds the index of primary SLC SAR 
        - Finds the Amplitude calibration factors
        - Computes and saves to disk the Interferometric Calibrated SLC stack
        - Computes and save to disk the normalize Amplitude information  
        - Updates baseline information, given the primary SLC SAR

    Args:
        input_dir (str): Input directory that we have SLCs, geometry, baseline and datetime information is stored
        output_dir (str): Output directory that SLCs, geometry, baseline and datetime information will be stored
        primary_date (str): the date of primary image

    Returns:
        int: the index of the primary SLC SAR
    """
    
    ########################################################################
    #### Create the folder structure (interferon) for the output directory
    ########################################################################
    # copy slc directory
    ########################################################################
    InterferonslcDir=os.path.join(input_dir,'slcs')
    slcDir=os.path.join(output_dir,'slcs')
    if not os.path.exists(slcDir):
        destination = shutil.copytree(InterferonslcDir, slcDir) 
    
    # copy geometrical information
    ########################################################################
    InterferongeometryDir=os.path.join(input_dir,'geometry')
    geometryDir=os.path.join(output_dir,'geometry')
    if not os.path.exists(geometryDir): os.makedirs(geometryDir)
    
    hgt_file=os.path.join(InterferongeometryDir,'hgt.npy')
    los_file=os.path.join(InterferongeometryDir,'los.npy')
    lon_file=os.path.join(InterferongeometryDir,'lon.npy')
    lat_file=os.path.join(InterferongeometryDir,'lat.npy')
    shadowMask_file=os.path.join(InterferongeometryDir,'shadowMask.npy')
    
    # save the incidence angle
    Inc_angle_raw=np.load(os.path.join(input_dir,'geometry/incLocal.npy'))
    Inc_angle_file=os.path.join(output_dir,'geometry/incLocal.npy')
    with open(Inc_angle_file, 'wb') as f:
        np.save(f, Inc_angle_raw) 
    
    Baseline_file=os.path.join(input_dir,'geometry/perp_baseline_stack.npy')

    for src_file in [hgt_file, los_file, lon_file, lat_file, shadowMask_file]:
        dest_file=os.path.join(geometryDir,os.path.basename(src_file))
        shutil.copy(src_file,dest_file)
    
    # copy coreg_stack directory
    ########################################################################
    coreg_stackDir=os.path.join(output_dir,'coreg_stack')
    if not os.path.exists(coreg_stackDir): os.makedirs(coreg_stackDir)
    
    # calculate sorted slc datetime objects and find the primary_index
    slc_files=glob.glob(slcDir+'/*.vrt')
    slc_dates=[os.path.basename(slc_file).split('.')[0] for slc_file in slc_files]
    slc_datetimes = np.array([datetime.strptime(slc_date,'%Y%m%d') for slc_date in sorted(slc_dates)])
    slc_datetimes_file = os.path.join(coreg_stackDir,'slc_datetimes.npy')
    
    with open(slc_datetimes_file, 'wb') as f:
        np.save(f, slc_datetimes) 
    
    primary_index=np.squeeze(np.where(slc_datetimes == datetime.strptime(primary_date,'%Y%m%d'))[0])
    
    with open('{}/primary_index.npy'.format(coreg_stackDir), 'wb') as f:
        np.save(f, np.array([primary_index]))
       
    # get slc data  and amp calibration factors   
    AmpCalFactors, slc_stack =runCalamp(inputDS = os.path.join(input_dir,'coreg_stack/slcs_base.vrt'),
                                        output_dir = coreg_stackDir)
    
    # sorted from earliest to latest slc dates
    AmpCalFactors_numpy = np.array(AmpCalFactors['Amplitude_factors'].sort_index())
    
    complex_interferometric_calibrated_data=np.zeros_like(slc_stack)
    complex_interferometric_calibrated_data=slc_stack[primary_index,:,:]*np.conjugate(slc_stack)
    complex_interferometric_calibrated_data=complex_interferometric_calibrated_data/AmpCalFactors_numpy[:,np.newaxis,np.newaxis]
    complex_interferometric_calibrated_data[primary_index,:,:]=1+0j
    
    with open('{}/complex_int_cal_stack.npy'.format(coreg_stackDir), 'wb') as f:
        np.save(f, complex_interferometric_calibrated_data.astype(np.complex128))
    
    
    # get amplitudes
    amp_nparray_raw=np.abs(slc_stack)
        
    # free up memory
    del slc_stack, complex_interferometric_calibrated_data
       
    # calculate calibrated(normalized) amplitude
    amp_nparray_raw[amp_nparray_raw<0.0005]=0.0005
    amp_nparray=np.zeros_like(amp_nparray_raw)
    for slave_index in range(amp_nparray.shape[0]):
        amp_nparray[slave_index,:,:]=amp_nparray_raw[slave_index,:,:]/(AmpCalFactors_numpy[slave_index]*amp_nparray_raw[primary_index,:,:])
      
    amp_nparray[amp_nparray<0.0005]=0.0005  
  
    # save calibrated(normalized) amplitude information    
    with open('{}/amp_stack.npy'.format(coreg_stackDir), 'wb') as f:
        np.save(f, amp_nparray)       
   
    # load baseline dataset
    Baseline_file=os.path.join(input_dir,'geometry/perp_baseline_stack.npy')
    Baseline_stack_data=np.load(Baseline_file)
    Baseline_stack_data=Baseline_stack_data-Baseline_stack_data[primary_index,:,:]

    with open('{}/perp_baseline_stack.npy'.format(geometryDir), 'wb') as f:
        np.save(f, Baseline_stack_data) 
         
          
    return primary_index
      


  



