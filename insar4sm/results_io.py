#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Saving and reusing the soil moisture estimations of each track.

Every track gets one folder per set of inversion parameters, named after the grid size, the
dry soil moisture state and a hash of all parameters (saved next to the results in
params.json). Rerunning with the same parameters loads the saved estimations instead of
inverting again, while changed parameters write a new folder and keep the previous ones.
"""
import os
import json
import hashlib
import numpy as np
import pandas as pd
import geopandas as gpd

PARAMS_FILE = 'params.json'
CSV_FILE = 'sm_inversions.csv'


def params_hash(params:dict)->str:
    """Short hash of a parameter dictionary (order of keys does not matter).
    """
    return hashlib.sha1(json.dumps(params, sort_keys=True, default=str).encode()).hexdigest()[:8]


def get_run_dir(export_dir:str, stack_file:str, params:dict)->tuple[str, dict]:
    """Results folder of a track for a set of inversion parameters.

    Args:
        export_dir (str): output directory of all tracks
        stack_file (str): path of stack file (see geocoded_stack.save_stack)
        params (dict): parameters shared by all tracks (must contain grid_size and sm_dry_state)

    Returns:
        tuple[str, dict]: results folder, parameters of the track (params plus the stack name and its acquisitions)
    """
    with np.load(stack_file) as data:
        acquisitions = [str(d) for d in data['datetimes']]
    track_name = os.path.basename(stack_file).split('.')[0]
    track_params = dict(params, stack=track_name, acquisitions=acquisitions)
    run_name = 'grid{}m_dry{}_{}'.format(params['grid_size'], params['sm_dry_state'], params_hash(track_params))
    return os.path.join(export_dir, track_name, run_name), track_params


def load_sm_estimations(run_dir:str, params:dict)->pd.DataFrame|None:
    """Loads the saved estimations of a track, if they were produced with the same parameters.

    Args:
        run_dir (str): results folder (see get_run_dir)
        params (dict): parameters of the track (see get_run_dir)

    Returns:
        pd.DataFrame|None: estimations (one row per grid cell, one column per acquisition), or None if there are no complete results
    """
    params_file = os.path.join(run_dir, PARAMS_FILE)
    if not os.path.exists(params_file):
        return None
    with open(params_file) as f:
        if json.load(f) != json.loads(json.dumps(params, default=str)):
            return None
    return pd.read_csv(os.path.join(run_dir, CSV_FILE), index_col=0)


def save_sm_estimations(run_dir:str, params:dict, sm_df:pd.DataFrame, sm_points:gpd.GeoSeries, name:str)->None:
    """Saves the estimations of a track as csv and shapefile, then its parameters.

    params.json is written last, so an interrupted run never leaves results that look complete.

    Args:
        run_dir (str): results folder (see get_run_dir)
        params (dict): parameters of the track (see get_run_dir)
        sm_df (pd.DataFrame): estimations (one row per grid cell, one column per acquisition)
        sm_points (gpd.GeoSeries): centroids of the grid cells in WGS84
        name (str): base name of the shapefile
    """
    os.makedirs(run_dir, exist_ok=True)
    sm_df.to_csv(os.path.join(run_dir, CSV_FILE))
    gpd.GeoDataFrame(sm_df, geometry=sm_points.values, crs='EPSG:4326').to_file(os.path.join(run_dir, '{}.shp'.format(name)))
    with open(os.path.join(run_dir, PARAMS_FILE), 'w') as f:
        json.dump(params, f, indent=1, default=str)
