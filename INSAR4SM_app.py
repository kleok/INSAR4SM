#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#---------------------Import libraries --------------
import os
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import geopandas as gpd
from tqdm import tqdm
from scipy.stats import pearsonr
import datetime
import matplotlib.pyplot as plt

#--------------------- INSAR4SM functionalities --------------
from insar4sm.download_ERA5_land import Get_ERA5_data
from insar4sm.classes import INSAR4SM_stack, SM_point
from insar4sm.joblib_progress_bar import tqdm_joblib

def sm_estimation(stack:INSAR4SM_stack, sm_ind:int, DS_flag: bool = True)->np.array:
    """Estimates soil moisture using insar4sm functionalities

    Args:
        stack (INSAR4SM_stack): the object of the INSAR4SM_stack class
        sm_ind (int): index of soil moisture estimation point
        DS_flag (bool): flag that determines if distributed scatterers will be computed.

    Returns:
        np.array: soil moisture estimations from inversion
    """
    sm_point_ts = SM_point(stack, sm_ind)
    sm_point_ts.amp_sel = DS_flag
    sm_point_ts.get_DS_info(stack)
    sm_point_ts.calc_covar_matrix()
    if sm_point_ts.non_coverage or np.all(sm_point_ts.amp_DS==0):
        return np.full(sm_point_ts.n_ifg, np.nan)
    else:
        sm_point_ts.get_DS_geometry(stack)
        sm_point_ts.calc_driest_date()
        if dry_date_manual_flag: sm_point_ts.driest_date = pd.to_datetime(dry_date)
        sm_point_ts.calc_sm_sorting()
        sm_point_ts.calc_sm_coherence()
        sm_point_ts.calc_sm_index()
        sm_point_ts.inversion()
        return sm_point_ts.sm_inverted
    
#%%##########################################################################
#-----------             Input arguments            ------------------------#
#############################################################################

# the name of your project
projectname = 'My_first_INSAR4SM_app'

# number of CPUs to be used
n_CPUs = 6

# the output directory 
export_dir = '/RSL02/SM_Arabia/{}'.format(projectname)

# the directory of the topstack processing stack
topstackDir = '/RSL02/SM_Arabia/Topstack_processing_2023/'

# soil information datasets (https://soilgrids.org/)
sand_soilgrids = '/RSL02/SM_Arabia/soilgrids/clay.tif'
clay_soilgrids = '/RSL02/SM_Arabia/soilgrids/sand.tif'

# time of Sentinel-1 pass.
orbit_time = '15:00:00'

# the AOI geojson file, ensure that AOI is inside your topstack stack
AOI = '/RSL02/SM_Arabia/aoi/aoi_test.geojson'

# half spatial resolution of soil moisture grid in meters
grid_size = 125

# You can set manually a dry date (one of your SAR acquisition dates) or set to None
dry_date = '20180401' 
# set to True in case you provide manually an dry_date
dry_date_manual_flag = True

#%%##########################################################################
#-----------        Provide meteorological data     ------------------------#
#############################################################################
# You can either 
# 1. provide an ERA5-land file with ['total_precipitation','skin_temperature','volumetric_soil_water_layer_1']
# 2. a csv file with 3 columns (Datetimes, tp__m, skt__K) 

#meteo_file = '/RSL02/SM_Arabia/My_first_INSAR4SM_app/ERA5/ERA5_20180401T000000_20181127T230000_17.5_53.6_17.2_53.9.nc'
meteo_file = None

# 3. set start and end date and automatically download the ERA5-land data
if meteo_file == None:
    start_date = '20180401' # format is YYYYMMDD
    end_date = '20181127' # format is YYYYMMDD

    ERA5_dir = os.path.join(export_dir, 'ERA5')
    if not os.path.exists(ERA5_dir): os.makedirs(ERA5_dir)

    meteo_file = Get_ERA5_data(ERA5_variables = ['total_precipitation','skin_temperature','volumetric_soil_water_layer_1'],
                            start_datetime = datetime.datetime.strptime('{}T000000'.format(start_date), '%Y%m%dT%H%M%S'),
                            end_datetime =  datetime.datetime.strptime('{}T230000'.format(end_date), '%Y%m%dT%H%M%S'),
                            AOI_file = AOI,
                            ERA5_dir = ERA5_dir)
    
# set to True in case you provide or downloaded an ERA5-Land file
ERA5_flag = True

# In case you downloaded surface soil moisture from ERA5-land, set to True for comparison purposes
ERA5_sm_flag = True

#%%##########################################################################
#-----------               InSAR4SM pipeline        ------------------------#
#############################################################################
                                                                                                                                                                                                                                                                                                                                                                                                                                                    
stack = INSAR4SM_stack(topstackDir = topstackDir,
                       projectname = projectname,
                       n_CPUs = n_CPUs,
                       AOI = AOI,
                       meteo_file = meteo_file,
                       ERA5_flag = ERA5_flag,
                       sand = sand_soilgrids,
                       clay = clay_soilgrids,
                       orbit_time = orbit_time,
                       export_dir = export_dir)

stack.prepare_datasets()
stack.plot()
stack.get_dry_SARs()
stack.calc_insar_stack()
stack.calc_grid(grid_size = grid_size)

with tqdm_joblib(tqdm(desc="SM Invertions", total=stack.n_sm_points)) as progress_bar:
    sm_estimations_list = Parallel(n_jobs=stack.CPUs, backend="threading")(delayed(sm_estimation)(stack, sm_ind) for sm_ind in range(stack.n_sm_points))


column_dates = [slc_date.strftime("D%Y%m%d") for slc_date in stack.slc_datetimes]
sm_estimations_df = pd.DataFrame(sm_estimations_list, index = range(stack.n_sm_points), columns = column_dates)

sm_estimations_df['geometry'] = stack.sm_points.values
sm_estimations_gdg = gpd.GeoDataFrame(sm_estimations_df, geometry='geometry')
sm_estimations_gdg.to_file(os.path.join(stack.export_dir,'sm_inversions_{}_{}.shp'.format(projectname,grid_size)))

#%%###########################################################################
#-------     Comparison  with ERA5 surface soil moisture if given     -------#
##############################################################################
if ERA5_sm_flag:
    comparison_df = (stack.meteo_sel_df['swvl1__m**3 m**-3']*100).copy().to_frame()
    
    sm_estimations_df.dropna(inplace=True)
    if 'geometry' in sm_estimations_df.columns:
        sm_estimations_df.drop(columns='geometry', inplace=True)
    comparison_df['sm_inverted'] = sm_estimations_df.mean(axis=0).values
    comparison_df.to_csv('{}/comparison_{}.csv'.format(stack.export_dir, grid_size), index=False)
    
    predictions = comparison_df['swvl1__m**3 m**-3'].values
    targets = comparison_df['sm_inverted'].values
    n = predictions.shape[0]
    rmse = np.linalg.norm(predictions - targets) / np.sqrt(n)
    r, p_value = pearsonr(predictions, targets)

    comparison_df.plot(figsize=(13,13), style='.-')
    plt.title('RMSE: {:.2f} m3/m3 \n R: {:.2f}'.format(rmse,r))
    plt.savefig('{}/ERA5_comparison_{}.png'.format(stack.export_dir, grid_size), dpi=200)
    plt.close()