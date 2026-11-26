#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""NISAR L2 GSLC access: only the AOI window of each granule is read remotely (no full download).

NASA Earthdata credentials are needed (earthaccess reads them from ~/.netrc, or asks once).
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
from shapely.geometry import Polygon

from insar4sm.geocoded_stack import (search_summary, aoi_grid, select_complete_dates, mosaic_sources,
                                     dem_on_grid, local_incidence_angle, save_stack)

GSLC_SHORT_NAME = 'NISAR_L2_GSLC_PROVISIONAL_V1'
FREQ_A_GROUP = '/science/LSAR/GSLC/grids/frequencyA'
ORBIT_GROUP = '/science/LSAR/GSLC/metadata/orbit'
IDENTIFICATION_GROUP = '/science/LSAR/identification'
MASK_INVALID_VALUES = (0, 255)
# e.g. NISAR_L2_PR_GSLC_023_066_D_067_2005_DHDH_A_20260618T185214_20260618T185249_P05023_N_F_J_001
NATIVE_ID_RE = re.compile(r'^NISAR_L2_PR_GSLC_\d{3}_(?P<track>\d{3})_(?P<dir>[AD])_(?P<frame>\d{3})_\w{4}_\w{4}_[A-Z]_'
                          r'(?P<start>\d{8}T\d{6})_\d{8}T\d{6}_(?P<crid>[A-Z]\d+)_[A-Z]_[A-Z]_[A-Z]_(?P<counter>\d{3})$')
FSSPEC_CONFIG = {'cache_type': 'background', 'block_size': 8 * 1024 * 1024}


def search_gslc(AOI_File:str, time_start:str, time_end:str)->dict:
    """Finds the NISAR GSLC granules over the AOI, keeping the latest version of each acquisition.

    Args:
        AOI_File (str): path of vector AOI file
        time_start (str): start datetime (YYYYMMDDTHHMMSS, UTC)
        time_end (str): end datetime (YYYYMMDDTHHMMSS, UTC)

    Returns:
        dict: {track (e.g. '066D'): {date (YYYYMMDD): {frame: earthaccess result}}}
    """
    bounds = tuple(gpd.read_file(AOI_File).to_crs(epsg=4326).total_bounds)
    results = earthaccess.search_data(short_name = GSLC_SHORT_NAME,
                                      bounding_box = bounds,
                                      temporal = (pd.to_datetime(time_start).isoformat(), pd.to_datetime(time_end).isoformat()))

    # reprocessed acquisitions come back once per version (CRID): keep the highest one
    latest = {}
    for result in results:
        m = NATIVE_ID_RE.match(result['meta']['native-id'])
        if m is None:
            print('Skipping granule with unexpected name: {}'.format(result['meta']['native-id']))
            continue
        key = (m['track'] + m['dir'], m['frame'], m['start'])
        version = (m['crid'][0], int(m['crid'][1:]), int(m['counter']))
        if key not in latest or version > latest[key][0]:
            latest[key] = (version, result)

    tracks = {}
    for (track, frame, start), (_, result) in latest.items():
        tracks.setdefault(track, {}).setdefault(start[:8], {})[frame] = result
    return tracks


def gslc_footprint(result)->Polygon:
    """Footprint of a GSLC granule from its catalogue record.
    """
    points = result['umm']['SpatialExtent']['HorizontalSpatialDomain']['Geometry']['GPolygons'][0]['Boundary']['Points']
    return Polygon([(point['Longitude'], point['Latitude']) for point in points])


def search_nisar_tracks(AOI_File:str, time_start:str, time_end:str)->list:
    """Summary of the NISAR tracks over the AOI, from the catalogue only (nothing is read).

    Args:
        AOI_File (str): path of vector AOI file
        time_start (str): start datetime (YYYYMMDDTHHMMSS, UTC)
        time_end (str): end datetime (YYYYMMDDTHHMMSS, UTC)

    Returns:
        list: one dict per track (see geocoded_stack.search_summary)
    """
    return [search_summary('NISAR', track, dates, AOI_File,
                           footprint = gslc_footprint,
                           start_time = lambda result: pd.to_datetime(result['umm']['TemporalExtent']['RangeDateTime']['BeginningDateTime']))
            for track, dates in sorted(search_gslc(AOI_File, time_start, time_end).items())]


def read_gslc_window(result, x:np.array, y:np.array, epsg:int|None, pol:str = 'HH', n_tries:int = 3)->dict:
    """Reads the part of a GSLC granule that falls on a map grid, without downloading the granule.

    Args:
        result: one earthaccess search result
        x (np.array): x pixel centres of the grid, or None to only read the granule lattice
        y (np.array): y pixel centres of the grid, or None to only read the granule lattice
        epsg (int | None): EPSG code of the grid
        pol (str, optional): polarization. Defaults to 'HH'.
        n_tries (int, optional): attempts before giving up on network errors. Defaults to 3.

    Returns:
        dict: chip (complex, NaN outside the granule or masked), acquisition datetime, frequency (GHz), orbit state vectors and the granule lattice (epsg, x0, dx, y0, dy)
    """
    url = result.data_links(access='external')[0]
    for attempt in range(n_tries):
        try:
            fs = earthaccess.get_fsspec_https_session()
            with fs.open(url, **FSSPEC_CONFIG) as fh, h5py.File(fh, 'r') as f:
                grid = f[FREQ_A_GROUP]
                out = {'epsg': int(grid['projection'][()]),
                       'x0': float(grid['xCoordinates'][0]), 'dx': float(grid['xCoordinateSpacing'][()]),
                       'y0': float(grid['yCoordinates'][0]), 'dy': float(grid['yCoordinateSpacing'][()]),
                       'datetime': str(pd.to_datetime(f[IDENTIFICATION_GROUP]['zeroDopplerStartTime'][()].decode())),
                       'freq_GHz': float(grid['centerFrequency'][()]) / 1e9,
                       'orbit_time': f[ORBIT_GROUP]['time'][()],
                       'orbit_pos': f[ORBIT_GROUP]['position'][()],
                       'orbit_vel': f[ORBIT_GROUP]['velocity'][()]}
                if x is None:
                    return out
                if out['epsg'] != epsg:
                    raise ValueError('Granule {} is in EPSG:{}, the track grid in EPSG:{}'.format(
                        result['meta']['native-id'], out['epsg'], epsg))

                # grid pixel -> granule pixel; frames of a track share one lattice
                cols = (x - out['x0']) / out['dx']
                rows = (y - out['y0']) / out['dy']
                if np.max(np.abs(cols - np.round(cols))) > 0.01 or np.max(np.abs(rows - np.round(rows))) > 0.01:
                    raise ValueError('Granule {} is not on the lattice of the track grid'.format(result['meta']['native-id']))
                cols, rows = np.round(cols).astype(int), np.round(rows).astype(int)
                in_x = (cols >= 0) & (cols < grid['xCoordinates'].shape[0])
                in_y = (rows >= 0) & (rows < grid['yCoordinates'].shape[0])

                chip = np.full((len(y), len(x)), np.nan + 1j * np.nan, dtype=np.complex64)
                if in_x.any() and in_y.any():
                    r0, r1 = rows[in_y].min(), rows[in_y].max() + 1
                    c0, c1 = cols[in_x].min(), cols[in_x].max() + 1
                    data = grid[pol][r0:r1, c0:c1].astype(np.complex64)
                    if 'mask' in grid:
                        data[np.isin(grid['mask'][r0:r1, c0:c1], MASK_INVALID_VALUES)] = np.nan
                    data[data == 0] = np.nan
                    chip[np.ix_(in_y, in_x)] = data
                out['chip'] = chip
                return out
        except (OSError, RuntimeError) as err:
            print('  attempt {} failed for {}: {}'.format(attempt + 1, result['meta']['native-id'], err))
            time.sleep(10 * (attempt + 1))
    raise RuntimeError('Could not read {}'.format(result['meta']['native-id']))


def build_nisar_stacks(AOI_File:str, time_start:str, time_end:str, out_dir:str, dem_file:str, margin_m:float, pol:str = 'HH', min_acquisitions:int = 5, tracks:list|None = None)->list:
    """Builds one geocoded SLC stack file per NISAR track that covers the AOI.

    Frames of the same pass that both cover the AOI are mosaicked. Read windows are cached
    in out_dir/chips, so an interrupted run continues where it stopped.

    Args:
        AOI_File (str): path of vector AOI file
        time_start (str): start datetime (YYYYMMDDTHHMMSS, UTC)
        time_end (str): end datetime (YYYYMMDDTHHMMSS, UTC)
        out_dir (str): output directory
        dem_file (str): DEM with ellipsoidal heights (see geocoded_stack.get_dem)
        margin_m (float): margin around the AOI in meters (at least the soil moisture grid size)
        pol (str, optional): polarization. Defaults to 'HH'.
        min_acquisitions (int, optional): tracks with fewer complete acquisitions are skipped. Defaults to 5.
        tracks (list | None, optional): tracks to process (e.g. ['066D']), see search_nisar_tracks. Defaults to None (all tracks).

    Returns:
        list: paths of the saved stack files
    """
    chips_dir = os.path.join(out_dir, 'chips')
    os.makedirs(chips_dir, exist_ok=True)
    stack_files = []

    for track, dates in sorted(search_gslc(AOI_File, time_start, time_end).items()):
        if tracks is not None and track not in tracks:
            continue
        kept_dates, frames = select_complete_dates({date: set(by_frame) for date, by_frame in dates.items()})
        print('NISAR track {}: {} acquisitions, {} with all frames {}'.format(track, len(dates), len(kept_dates), frames))
        if len(kept_dates) < min_acquisitions:
            print('  skipped, fewer than {} acquisitions'.format(min_acquisitions))
            continue

        # grid on the native lattice of the track, taken from its first granule
        grid_file = os.path.join(chips_dir, 'grid_{}.npz'.format(track))
        if not os.path.exists(grid_file):
            lattice = read_gslc_window(dates[kept_dates[0]][frames[0]], None, None, None)
            x, y = aoi_grid(AOI_File, lattice['epsg'], lattice['x0'], lattice['dx'], lattice['y0'], lattice['dy'], margin_m)
            np.savez(grid_file, x=x, y=y, epsg=lattice['epsg'])
        with np.load(grid_file) as grid:
            x, y, epsg = grid['x'], grid['y'], int(grid['epsg'])

        chips = {}
        for date in kept_dates:
            for frame in frames:
                result = dates[date][frame]
                chip_file = os.path.join(chips_dir, result['meta']['native-id'] + '.npz')
                if not os.path.exists(chip_file):
                    t0 = time.time()
                    window = read_gslc_window(result, x, y, epsg, pol)
                    np.savez(chip_file, **window)
                    print('  {} frame {} read in {:.0f} s'.format(date, frame, time.time() - t0))
                with np.load(chip_file) as data:
                    chips[(date, frame)] = {key: data[key] for key in data.files}

        source_stacks = [np.stack([chips[(date, frame)]['chip'] for date in kept_dates]) for frame in frames]
        slc, source_index = mosaic_sources(source_stacks)
        print('  {:.1f}% of the AOI grid has data on every date'.format(100 * np.mean(source_index >= 0)))

        first = chips[(kept_dates[0], frames[0])]
        hgt = dem_on_grid(dem_file, x, y, epsg)
        inc = local_incidence_angle(x, y, epsg, hgt, first['orbit_time'], first['orbit_pos'], first['orbit_vel'])
        stack_file = os.path.join(out_dir, 'NISAR_{}.npz'.format(track))
        save_stack(stack_file, slc, [pd.Timestamp(str(chips[(date, frames[0])]['datetime'])) for date in kept_dates],
                   x, y, epsg, inc, hgt, float(first['freq_GHz']), 'NISAR', track)
        stack_files.append(stack_file)

    return stack_files
