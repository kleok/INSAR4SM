#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""In situ soil moisture of the International Soil Moisture Network (ISMN, https://ismn.earth) for validation.
"""
import os
import glob
import zipfile
import numpy as np
import pandas as pd


def read_ismn_stm(stm_file:str)->tuple[str, float, float, float, pd.Series]:
    """Reads an ISMN soil moisture file (format "separate files with header", .stm).

    Only the records flagged as good (G) are kept. Zero values are dropped too, since a failed
    sensor may report zeros that still pass the quality control.

    Args:
        stm_file (str): path of .stm file

    Returns:
        tuple[str, float, float, float, pd.Series]: station name, latitude, longitude, depth from (m), hourly soil moisture (%) in UTC
    """
    with open(stm_file) as f:
        header = f.readline().split()
    df = pd.read_csv(stm_file, sep=r'\s+', skiprows=1, header=None, names=['date', 'time', 'sm', 'flag', 'orig_flag'])
    df.index = pd.to_datetime(df['date'] + ' ' + df['time'])
    sm = df['sm'].where((df['flag'] == 'G') & (df['sm'] > 0)) * 100
    return header[2], float(header[3]), float(header[4]), float(header[6]), sm


def get_ismn_stations(ismn_file:str, out_dir:str, max_depth_m:float = 0.05, min_valid_fraction:float = 0.5)->pd.DataFrame:
    """Extracts an ISMN download and reads the soil moisture of every station.

    Args:
        ismn_file (str): path of the ISMN zip file (or of an already extracted folder)
        out_dir (str): folder where the zip file is extracted
        max_depth_m (float, optional): only sensors with upper depth at or above this value are kept. Defaults to 0.05.
        min_valid_fraction (float, optional): sensors with a smaller fraction of valid hourly records are dropped. Defaults to 0.5.

    Returns:
        pd.DataFrame: one row per sensor with station, lat, lon, depth and its hourly soil moisture series (sm)
    """
    if os.path.isdir(ismn_file):
        out_dir = ismn_file
    else:
        with zipfile.ZipFile(ismn_file) as z:
            z.extractall(out_dir)
    stations = [read_ismn_stm(f) for f in sorted(glob.glob(os.path.join(out_dir, '**', '*_sm_*.stm'), recursive=True))]
    stations_df = pd.DataFrame(stations, columns=['station', 'lat', 'lon', 'depth', 'sm'])
    # some networks install sensors in inches (e.g. SCAN: 2 in = 0.0508 m), just below max_depth_m = 0.05
    if not (stations_df['depth'] <= max_depth_m).any():
        raise ValueError('No sensor at or above max_depth_m = {} m. Upper depths in {}: {}'.format(
            max_depth_m, out_dir, sorted(stations_df['depth'].unique())))
    stations_df = stations_df[stations_df['depth'] <= max_depth_m]
    valid_fraction = stations_df['sm'].apply(lambda sm: sm.notna().mean())
    for station, fraction in zip(stations_df['station'], valid_fraction):
        if fraction < min_valid_fraction:
            print('Dropping station {}: only {:.0%} of its records are valid'.format(station, fraction))
    return stations_df[valid_fraction >= min_valid_fraction].reset_index(drop=True)


def validation_scores(insitu:np.array, insar:np.array)->dict:
    """Agreement of InSAR4SM estimations with in situ soil moisture (both in %).

    Args:
        insitu (np.array): in situ soil moisture
        insar (np.array): InSAR4SM soil moisture at the same times

    Returns:
        dict: number of pairs, RMSE, unbiased RMSE, bias (InSAR4SM - in situ), Pearson R
    """
    ok = np.isfinite(insitu) & np.isfinite(insar)
    x, y = insitu[ok], insar[ok]
    if ok.sum() < 3:
        return {'n': int(ok.sum()), 'RMSE': np.nan, 'ubRMSE': np.nan, 'bias': np.nan, 'R': np.nan}
    return {'n': int(ok.sum()),
            'RMSE': np.sqrt(np.mean((y - x)**2)),
            'ubRMSE': np.sqrt(np.mean(((y - y.mean()) - (x - x.mean()))**2)),
            'bias': np.mean(y - x),
            'R': np.corrcoef(x, y)[0, 1]}
