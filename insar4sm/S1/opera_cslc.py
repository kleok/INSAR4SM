#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sentinel-1 OPERA L2 CSLC-S1 access: only the AOI window of each burst is read remotely.

OPERA CSLC-S1 products are Sentinel-1 bursts geocoded by JPL with COMPASS (5 m x 10 m posting,
topographic and ellipsoidal phase flattened). They cover North America only. Elsewhere, use
insar4sm.S1.s1_cslc, which geocodes the bursts locally with COMPASS/isce3.

Needs NASA Earthdata credentials (earthaccess reads them from ~/.netrc, or asks once). No isce3.
"""
from __future__ import annotations
import os
import re
import time
import numpy as np
import pandas as pd
import geopandas as gpd
import h5py
import earthaccess

from insar4sm.geocoded_stack import (umm_footprint, umm_start_time, search_summary, aoi_grid, select_complete_dates,
                                     mosaic_sources, dem_on_grid, local_incidence_angle, save_stack)

CSLC_SHORT_NAME = 'OPERA_L2_CSLC-S1_V1'
# e.g. OPERA_L2_CSLC-S1_T100-213521-IW3_20251201T133628Z_20260217T201504Z_S1A_VV_v1.1
NATIVE_ID_RE = re.compile(r'^OPERA_L2_CSLC-S1_T(?P<track>\d{3})-(?P<burst>\d{6})-(?P<swath>IW\d)_'
                          r'(?P<start>\d{8}T\d{6})Z_(?P<production>\d{8}T\d{6})Z_(?P<platform>S1[A-Z])_'
                          r'(?P<pol>[HV]{2})_v(?P<version>[\d.]+)$')
FSSPEC_CONFIG = {'cache_type': 'background', 'block_size': 4 * 1024 * 1024}
FLATTENING_FLAGS = ('metadata/processing_information/parameters/topographic_flattening_applied',
                    'metadata/processing_information/parameters/ellipsoidal_flattening_applied')


def search_opera_cslc(AOI_File:str, time_start:str, time_end:str, pol:str = 'VV')->dict:
    """Finds the OPERA CSLC-S1 bursts over the AOI, keeping the latest version of each acquisition.

    Args:
        AOI_File (str): path of vector AOI file
        time_start (str): start datetime (YYYYMMDDTHHMMSS, UTC)
        time_end (str): end datetime (YYYYMMDDTHHMMSS, UTC)
        pol (str, optional): polarization. Defaults to 'VV'.

    Returns:
        dict: {track number (e.g. '100'): {date (YYYYMMDD): {burst_id (e.g. 't100_213521_iw3'): earthaccess result}}}
    """
    bounds = tuple(gpd.read_file(AOI_File).to_crs(epsg=4326).total_bounds)
    results = earthaccess.search_data(short_name = CSLC_SHORT_NAME,
                                      bounding_box = bounds,
                                      temporal = (pd.to_datetime(time_start).isoformat(), pd.to_datetime(time_end).isoformat()))

    # reprocessed acquisitions come back once per version / production time: keep the latest
    latest = {}
    for result in results:
        m = NATIVE_ID_RE.match(result['meta']['native-id'])
        if m is None:
            print('Skipping granule with unexpected name: {}'.format(result['meta']['native-id']))
            continue
        if m['pol'] != pol:
            continue
        burst_id = 't{}_{}_{}'.format(m['track'], m['burst'], m['swath'].lower())
        key = (m['track'], burst_id, m['start'])
        version = (tuple(int(v) for v in m['version'].split('.')), m['production'])
        if key not in latest or version > latest[key][0]:
            latest[key] = (version, result)

    tracks = {}
    for (track, burst_id, start), (_, result) in latest.items():
        tracks.setdefault(track, {}).setdefault(start[:8], {})[burst_id] = result
    return tracks


def track_label(track:str, dates:dict)->str:
    """Track number plus pass direction (e.g. '100D'), as in the stack names, from the catalogue record.

    Args:
        track (str): track number (e.g. '100')
        dates (dict): {date: {burst_id: earthaccess result}} of the track (see search_opera_cslc)

    Returns:
        str: track label
    """
    result = next(iter(next(iter(dates.values())).values()))
    direction = [a['Values'][0] for a in result['umm']['AdditionalAttributes'] if a['Name'] == 'ASCENDING_DESCENDING'][0]
    return track + direction[0]


def search_opera_tracks(AOI_File:str, time_start:str, time_end:str, pol:str = 'VV')->list:
    """Summary of the Sentinel-1 OPERA CSLC-S1 tracks over the AOI, from the catalogue only (nothing is read).

    Args:
        AOI_File (str): path of vector AOI file
        time_start (str): start datetime (YYYYMMDDTHHMMSS, UTC)
        time_end (str): end datetime (YYYYMMDDTHHMMSS, UTC)
        pol (str, optional): polarization. Defaults to 'VV'.

    Returns:
        list: one dict per track (see geocoded_stack.search_summary)
    """
    return [search_summary('S1', track_label(track, dates), dates, AOI_File,
                           footprint = umm_footprint,
                           start_time = umm_start_time)
            for track, dates in sorted(search_opera_cslc(AOI_File, time_start, time_end, pol).items())]


def read_opera_window(result, x:np.array, y:np.array, epsg:int|None, pol:str = 'VV', n_tries:int = 3)->dict:
    """Reads the part of an OPERA CSLC-S1 burst that falls on a map grid, without downloading the burst.

    Args:
        result: one earthaccess search result
        x (np.array): x pixel centres of the grid, or None to only read the burst lattice
        y (np.array): y pixel centres of the grid, or None to only read the burst lattice
        epsg (int | None): EPSG code of the grid
        pol (str, optional): polarization. Defaults to 'VV'.
        n_tries (int, optional): attempts before giving up on network errors. Defaults to 3.

    Returns:
        dict: chip (complex, NaN outside the burst), acquisition datetime, frequency (GHz), pass direction, orbit state vectors and the burst lattice (epsg, x0, dx, y0, dy)
    """
    url = [link for link in result.data_links(access='external') if link.endswith('.h5')][0]
    for attempt in range(n_tries):
        try:
            fs = earthaccess.get_fsspec_https_session()
            with fs.open(url, **FSSPEC_CONFIG) as fh, h5py.File(fh, 'r') as f:
                orbit = f['metadata/orbit']
                out = {'epsg': int(f['data/projection'][()]),
                       'x0': float(f['data/x_coordinates'][0]), 'dx': float(f['data/x_spacing'][()]),
                       'y0': float(f['data/y_coordinates'][0]), 'dy': float(f['data/y_spacing'][()]),
                       'datetime': str(pd.to_datetime(f['identification/zero_doppler_start_time'][()].decode())),
                       'freq_GHz': float(f['metadata/processing_information/input_burst_metadata/radar_center_frequency'][()]) / 1e9,
                       'pass_direction': f['identification/orbit_pass_direction'][()].decode(),
                       'flattened': all(bool(f[flag][()]) for flag in FLATTENING_FLAGS if flag in f),
                       'orbit_time': orbit['time'][()],
                       'orbit_pos': np.stack([orbit['position_' + c][()] for c in 'xyz'], axis=-1),
                       'orbit_vel': np.stack([orbit['velocity_' + c][()] for c in 'xyz'], axis=-1)}
                if x is None:
                    return out
                if out['epsg'] != epsg:
                    raise ValueError('Burst {} is in EPSG:{}, the track grid in EPSG:{}'.format(
                        result['meta']['native-id'], out['epsg'], epsg))
                if not out['flattened']:
                    raise ValueError('Burst {} is not flattened, InSAR4SM assumes zero perpendicular baselines'.format(
                        result['meta']['native-id']))

                # grid pixel -> burst pixel; all bursts share the 5 m x 10 m posting lattice
                cols = (x - out['x0']) / out['dx']
                rows = (y - out['y0']) / out['dy']
                if np.max(np.abs(cols - np.round(cols))) > 0.01 or np.max(np.abs(rows - np.round(rows))) > 0.01:
                    raise ValueError('Burst {} is not on the lattice of the track grid'.format(result['meta']['native-id']))
                cols, rows = np.round(cols).astype(int), np.round(rows).astype(int)
                in_x = (cols >= 0) & (cols < f['data/x_coordinates'].shape[0])
                in_y = (rows >= 0) & (rows < f['data/y_coordinates'].shape[0])

                chip = np.full((len(y), len(x)), np.nan + 1j * np.nan, dtype=np.complex64)
                if in_x.any() and in_y.any():
                    r0, r1 = rows[in_y].min(), rows[in_y].max() + 1
                    c0, c1 = cols[in_x].min(), cols[in_x].max() + 1
                    data = f['data/' + pol][r0:r1, c0:c1].astype(np.complex64)
                    data[data == 0] = np.nan
                    chip[np.ix_(in_y, in_x)] = data
                out['chip'] = chip
                return out
        except (OSError, RuntimeError) as err:
            print('  attempt {} failed for {}: {}'.format(attempt + 1, result['meta']['native-id'], err))
            time.sleep(10 * (attempt + 1))
    raise RuntimeError('Could not read {}'.format(result['meta']['native-id']))


def build_opera_stacks(AOI_File:str, time_start:str, time_end:str, out_dir:str, dem_file:str, margin_m:float, pol:str = 'VV', min_acquisitions:int = 5, tracks:list|None = None)->list:
    """Builds one geocoded SLC stack file per Sentinel-1 track from OPERA CSLC-S1 bursts (North America).

    Bursts of the same pass that cover the AOI are mosaicked. Read windows are cached in
    out_dir/chips, so an interrupted run continues where it stopped.

    Args:
        AOI_File (str): path of vector AOI file
        time_start (str): start datetime (YYYYMMDDTHHMMSS, UTC)
        time_end (str): end datetime (YYYYMMDDTHHMMSS, UTC)
        out_dir (str): output directory
        dem_file (str): DEM with ellipsoidal heights (see geocoded_stack.get_dem)
        margin_m (float): margin around the AOI in meters (at least the soil moisture grid size)
        pol (str, optional): polarization. Defaults to 'VV'.
        min_acquisitions (int, optional): tracks with fewer complete acquisitions are skipped. Defaults to 5.
        tracks (list | None, optional): tracks to process (e.g. ['100D']), see search_opera_tracks. Defaults to None (all tracks).

    Returns:
        list: paths of the saved stack files
    """
    chips_dir = os.path.join(out_dir, 'chips')
    os.makedirs(chips_dir, exist_ok=True)
    stack_files = []

    found_tracks = search_opera_cslc(AOI_File, time_start, time_end, pol)
    if not found_tracks:
        print('No OPERA CSLC-S1 bursts found (OPERA covers North America only; use insar4sm.S1.s1_cslc elsewhere)')
    if tracks is not None:
        found_tracks = {track: dates for track, dates in found_tracks.items() if track_label(track, dates) in tracks}

    for track, dates in sorted(found_tracks.items()):
        kept_dates, bursts = select_complete_dates({date: set(by_burst) for date, by_burst in dates.items()})
        print('S1 (OPERA) track {}: {} acquisitions, {} with all bursts {}'.format(track, len(dates), len(kept_dates), bursts))
        if len(kept_dates) < min_acquisitions:
            print('  skipped, fewer than {} acquisitions'.format(min_acquisitions))
            continue

        # grid on the posting lattice of the track, taken from its first burst
        grid_file = os.path.join(chips_dir, 'grid_T{}.npz'.format(track))
        if not os.path.exists(grid_file):
            lattice = read_opera_window(dates[kept_dates[0]][bursts[0]], None, None, None, pol)
            x, y = aoi_grid(AOI_File, lattice['epsg'], lattice['x0'], lattice['dx'], lattice['y0'], lattice['dy'], margin_m)
            np.savez(grid_file, x=x, y=y, epsg=lattice['epsg'], pass_direction=lattice['pass_direction'])
        with np.load(grid_file) as grid:
            x, y, epsg = grid['x'], grid['y'], int(grid['epsg'])
            track_label = track + str(grid['pass_direction'])[0]

        chips = {}
        for date in kept_dates:
            for burst_id in bursts:
                result = dates[date][burst_id]
                chip_file = os.path.join(chips_dir, result['meta']['native-id'] + '.npz')
                if not os.path.exists(chip_file):
                    t0 = time.time()
                    window = read_opera_window(result, x, y, epsg, pol)
                    np.savez(chip_file, **window)
                    print('  {} {} read in {:.0f} s'.format(date, burst_id, time.time() - t0))
                with np.load(chip_file) as data:
                    chips[(date, burst_id)] = {key: data[key] for key in data.files}

        source_stacks = [np.stack([chips[(date, burst_id)]['chip'] for date in kept_dates]) for burst_id in bursts]
        slc, source_index = mosaic_sources(source_stacks)
        print('  {:.1f}% of the AOI grid has data on every date'.format(100 * np.mean(source_index >= 0)))

        first = chips[(kept_dates[0], bursts[0])]
        hgt = dem_on_grid(dem_file, x, y, epsg)
        inc = local_incidence_angle(x, y, epsg, hgt, first['orbit_time'], first['orbit_pos'], first['orbit_vel'])
        stack_file = os.path.join(out_dir, 'S1_{}.npz'.format(track_label))
        save_stack(stack_file, slc, [pd.Timestamp(str(chips[(date, bursts[0])]['datetime'])) for date in kept_dates],
                   x, y, epsg, inc, hgt, float(first['freq_GHz']), 'S1', track_label)
        stack_files.append(stack_file)

    return stack_files
