#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reads in-situ soil moisture that the user downloaded from the International Soil Moisture Network.

ISMN data cannot be redistributed (https://ismn.earth/en/terms-and-conditions/), so InSAR4SM ships
none: every user downloads the station data with their own ISMN account.
"""
from __future__ import annotations
import pandas as pd


def read_ismn(filename:str)->pd.Series:
    """Reads the hourly soil moisture of one ISMN sensor.

    Args:
        filename (str): ISMN .stm file ("Header+values" format, only values flagged G are kept), or a CSV with columns date_time, soil_moisture

    Returns:
        pd.Series: volumetric soil moisture (m3/m3) indexed by UTC time
    """
    if str(filename).lower().endswith('.stm'):
        with open(filename) as f:
            next(f) # station header
            rows = [line.split() for line in f if line.strip()]
        rows = [r for r in rows if r[3] == 'G']
        index = pd.to_datetime([r[0] + ' ' + r[1] for r in rows], format='%Y/%m/%d %H:%M')
        sm = pd.Series([float(r[2]) for r in rows], index=index, name='soil_moisture')
    else:
        sm = pd.read_csv(filename, parse_dates=['date_time']).set_index('date_time')['soil_moisture']
    sm.index.name = 'date_time'
    return sm.sort_index().dropna()
