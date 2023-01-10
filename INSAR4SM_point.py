#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SM estimation at a single point

Author: Kleanthis Karamvasis
Organization: National Technical University of Athens
Creation Date: 04/09/2021
Project: INSAR4SM
"""

#---------------------Import libraries --------------
from joblib import Parallel, delayed
import pandas as pd
import geopandas as gpd
from tqdm import tqdm
from random import sample
import matplotlib.pyplot as plt
import numpy as np

#--------------------- INSAR4SM functionalities --------------
from insar4sm.classes import INSAR4SM_stack, SM_point

#%%##########################################################################
#-----------      Input arguments   ------------------------#
#############################################################################

# The ISMN station
# station_name = 'FordDryLake'
station_name = 'DesertCenter'
orbit_time = '02:00:00'

#orbit_nums = ['100','173']
#orbit_nums = ['100']
#sq_sizes = [50,100,200,250,300,400,500]

# orbit_num = '100'
#orbit_num = '173'
orbit_num = '166'
sq_size = 250

###############################################################################
# the name of your experiment
projectname = 'INSAR4SM_ISMN_newtest2_{}_sq{}_{}'.format(orbit_num, sq_size, station_name)

# the directory of the topstack processing
topstackDir = '/RSL02/SM_NA/Topstack_processing_orbit_{}'.format(orbit_num)

# the AOI geojson file for your project
# ensure that AOI is inside your topstack stack
#AOI = '/RSL02/SM_NA/Plotting/bbox_aoi.geojson'
AOI = '/RSL02/SM_NA/ISMN/{}/{}_AOI.geojson'.format(station_name,station_name)

# the meteorological file. You can either provide an ERA5-land file or a csv file with 3 columns (Datetimes, tp__m, skt__K).
meteo_file = '/RSL02/SM_NA/era5/era5_land_na_orbit_{}.nc'.format(orbit_num)
# set to True in case you provide an ERA5-Land file
ERA5_flag = True
# In case you downloaded surface soil moisture from ERA5-land, set to True for comparison purposes
ERA5_sm_flag = True

# the output directory 
export_dir = '/RSL02/SM_NA/{}'.format(projectname)

# soil information datasets (https://soilgrids.org/)
sand_soilgrids = 87
clay_soilgrids = 13

# the insitu measurements in csv format
ISMN_csv = '/RSL02/SM_NA/ISMN/{}/ismn_station_{}.csv'.format(station_name, station_name)

# geometrical infromation regarding ISMN station

#IMSN_polygon = gpd.read_file('/RSL02/SM_NA/ISMN/{}/{}_neighborhood.geojson'.format(station_name, station_name))['geometry']
IMSN_polygon = gpd.read_file('/RSL02/SM_NA/ISMN/{}/{}_neighborhood_r{}.geojson'.format(station_name,
                                                                                        station_name,
                                                                                        sq_size))['geometry']
ISMN_point = IMSN_polygon.centroid

#%%##########################################################################
#-----------      Step A: Preparation of slc stack   ------------------------#
#############################################################################

stack = INSAR4SM_stack(topstackDir = topstackDir,
                       projectname = projectname,
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

#%%###########################################################################
#-------     Step B: SM Estimation        -------#
##############################################################################
stack.sm_points = ISMN_point
stack.sm_polygons = IMSN_polygon
stack.n_sm_points = len(stack.sm_points)

sm_point_ts = SM_point(stack, sm_ind=0)
sm_point_ts.get_DS_info(stack)
sm_point_ts.calc_covar_matrix()
sm_point_ts.get_DS_geometry(stack)

sm_point_ts.calc_driest_date()
#sm_point_ts.driest_date = pd.to_datetime('20180704')
sm_point_ts.driest_date = pd.to_datetime('20180710')
sm_point_ts.calc_sm_sorting()

# in_situ_data = pd.read_csv('/RSL02/SM_NA/comparison_FordDryLake.csv')
# print(in_situ_data['sm_plot'].loc[sm_point_ts.best_sorting])
sm_point_ts.calc_sm_coherence()
sm_point_ts.calc_sm_index()
sm_point_ts.inversion()



#%%###########################################################################
#-------     Step C: Plotting        -------#
##############################################################################

#-- Plotting raw coherence

fig, ax = plt.subplots(1,1, figsize=(15, 15))
img = ax.imshow(sm_point_ts.coh_full_DS[0,:,:])
y_label_list = [sar_datetime.strftime('%d-%m-%Y') for sar_datetime in sm_point_ts.slc_dates.date]
ax.set_yticks(np.arange(len(y_label_list)))
ax.set_yticklabels(y_label_list)
fig.colorbar(img)
plt.title('Driest Date: {}'.format(sm_point_ts.driest_date))
plt.savefig('{}/Raw_coh_{}_sq{}.png'.format(stack.export_dir, station_name, sq_size), dpi=200)
plt.close()

#-- Plotting sm coherence

fig, ax = plt.subplots(1,1, figsize=(15, 15))
img = ax.imshow(sm_point_ts.coh_sm[0,:,:])
y_label_list = [sar_datetime.strftime('%d-%m-%Y') for sar_datetime in sm_point_ts.slc_dates.date]
ax.set_yticks(np.arange(len(y_label_list)))
ax.set_yticklabels(y_label_list)
fig.colorbar(img)
plt.savefig('{}/SM_coh_{}_sq{}.png'.format(stack.export_dir, station_name, sq_size), dpi=200)
plt.close()
    
#%%###########################################################################
#-------     Step D: Comparison        -------#
##############################################################################

IMSN_df = pd.read_csv(ISMN_csv)
IMSN_df.index = pd.to_datetime(IMSN_df['Datetime'])
IMSN_df = IMSN_df['sm_plot']
# select only particular hour

IMSN_df = IMSN_df.at_time(orbit_time).to_frame()

#insar4sm_df = pd.DataFrame(np.ones_like(sm_point_ts.best_sorting), columns=['insar4sm'])
sm_estimations = {'SM0':sm_point_ts.SM0,
                  'SM_index':sm_point_ts.SM_index,
                  'insar4sm':sm_point_ts.sm_inverted
                  }

insar4sm_df = pd.DataFrame(sm_estimations)
#insar4sm_df = pd.DataFrame(sm_point_ts.SM_index, columns=['insar4sm'])
insar4sm_df.index = pd.to_datetime(stack.slc_datetimes)
insar4sm_df.index = insar4sm_df.index + pd.Timedelta('{} hour'.format(pd.to_datetime(orbit_time).hour))

comparison_df = IMSN_df.join(insar4sm_df, how='outer').dropna()
comparison_df['Datetime'] = comparison_df.index
comparison_df.to_csv('{}/comparison_{}.csv'.format(stack.export_dir,station_name), index=False)

predictions = comparison_df['insar4sm'].values
targets = comparison_df['sm_plot'].values
n = predictions.shape[0]
rmse = np.linalg.norm(predictions - targets) / np.sqrt(n)

comparison_df[['sm_plot', 'SM0', 'SM_index', 'insar4sm']].plot(figsize=(13,13), style='.-')
plt.title('RMSE: {} m3/m3'.format(round(rmse,2)))

plt.savefig('{}/SM_estimations_{}_sq{}.png'.format(stack.export_dir, station_name, sq_size), dpi=200)
plt.savefig('{}/SM_estimations_{}_sq{}.svg'.format(stack.export_dir, station_name, sq_size), format="svg")
plt.close()

plot_df = IMSN_df.join(insar4sm_df, how='outer').dropna(subset=['insar4sm'])
df = plot_df.iloc[sm_point_ts.best_sorting].round(decimals = 2).copy()
df['Datetime'] = df.index
cell_text = []
for row in range(len(df)):
    cell_text.append(df.iloc[row])
plt.table(cellText=cell_text, colLabels=df.columns,fontsize=14, loc='center')
plt.axis('off')

plt.savefig('{}/Ordering_{}_sq{}.png'.format(stack.export_dir, station_name, sq_size), dpi=200)
plt.close()