#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sentinel-1 geocoded CSLCs over the AOI, from single bursts instead of whole SLC scenes.

Chain per burst and date: ASF burst extractor + burst2safe (one-burst SAFE, ~150 MB),
s1reader orbit download, then COMPASS (the OPERA CSLC-S1 processor) geocoding only the
AOI. COMPASS reads the output grid of each burst from a burst database; here that database
gives every burst the same AOI box, so all bursts and dates land on one grid and an AOI
that crosses bursts is just a mosaic. Needs Linux (isce3) and NASA Earthdata credentials
in ~/.netrc.
"""
from __future__ import annotations
import os
import sys
import glob
import ctypes
import shutil
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import h5py
import asf_search as asf

from insar4sm.gridding import WGS84_to_UTM
from insar4sm.geocoded_stack import (select_complete_dates, mosaic_sources, dem_on_grid,
                                     local_incidence_angle, save_stack)

X_POSTING, Y_POSTING = 5, 10  # OPERA CSLC-S1 posting in meters

RUNCONFIG_TEMPLATE = '''runconfig:
  name: cslc_s1_workflow_default
  groups:
      pge_name_group:
          pge_name: CSLC_S1_PGE
      input_file_group:
          safe_file_path: [{safe}]
          orbit_file_path: [{orbits}]
          burst_id: [{burst_id}]
      dynamic_ancillary_file_group:
          dem_file: {dem}
      static_ancillary_file_group:
          burst_database_file: {burst_db}
      product_path_group:
          product_path: {product_dir}
          scratch_path: {scratch_dir}
          sas_output_file: {product_dir}
      primary_executable:
          product_type: CSLC_S1
      processing:
          polarization: co-pol
          geocoding:
              flatten: True
              x_posting: {dx}
              y_posting: {dy}
              x_snap: {dx}
              y_snap: {dy}
      output:
          chunk_size: [16, 16]
'''


def search_bursts(AOI_File:str, time_start:str, time_end:str, pol:str = 'VV')->dict:
    """Finds the Sentinel-1 bursts over the AOI (ASF burst catalogue, global coverage).

    Args:
        AOI_File (str): path of vector AOI file
        time_start (str): start datetime (YYYYMMDDTHHMMSS, UTC)
        time_end (str): end datetime (YYYYMMDDTHHMMSS, UTC)
        pol (str, optional): polarization. Defaults to 'VV'.

    Returns:
        dict: {track (e.g. '030A'): {date (YYYYMMDD): {burst_id (e.g. 't030_062533_iw3'): burst granule name}}}
    """
    aoi = gpd.read_file(AOI_File).to_crs(epsg=4326).geometry.union_all()
    results = asf.search(dataset = 'SLC-BURST',
                         intersectsWith = aoi.wkt,
                         start = pd.to_datetime(time_start).isoformat(),
                         end = pd.to_datetime(time_end).isoformat(),
                         polarization = pol)
    tracks = {}
    for result in results:
        p = result.properties
        track = '{:03d}{}'.format(int(p['pathNumber']), p['flightDirection'][0])
        burst_id = 't' + p['burst']['fullBurstID'].lower()
        date = pd.to_datetime(p['startTime']).strftime('%Y%m%d')
        tracks.setdefault(track, {}).setdefault(date, {})[burst_id] = p['fileID']
    return tracks


def write_burst_db(AOI_File:str, burst_ids:list, burst_db_file:str, margin_m:float)->int:
    """Writes a COMPASS burst database that maps every burst to the same AOI box.

    Args:
        AOI_File (str): path of vector AOI file
        burst_ids (list): burst ids (e.g. 't030_062533_iw3')
        burst_db_file (str): output sqlite file
        margin_m (float): margin around the AOI in meters

    Returns:
        int: EPSG code of the output grid (UTM zone of the AOI)
    """
    aoi = gpd.read_file(AOI_File).to_crs(epsg=4326)
    centroid = aoi.geometry.union_all().centroid
    epsg = WGS84_to_UTM(centroid.x, centroid.y)
    xmin, ymin, xmax, ymax = aoi.to_crs(epsg=epsg).total_bounds
    xmin, xmax = np.floor((xmin - margin_m) / X_POSTING) * X_POSTING, np.ceil((xmax + margin_m) / X_POSTING) * X_POSTING
    ymin, ymax = np.floor((ymin - margin_m) / Y_POSTING) * Y_POSTING, np.ceil((ymax + margin_m) / Y_POSTING) * Y_POSTING

    con = sqlite3.connect(burst_db_file)
    con.execute('DROP TABLE IF EXISTS burst_id_map')
    con.execute('CREATE TABLE burst_id_map (burst_id_jpl TEXT, epsg INTEGER, xmin REAL, ymin REAL, xmax REAL, ymax REAL)')
    con.executemany('INSERT INTO burst_id_map VALUES (?,?,?,?,?,?)',
                    [(burst_id, epsg, xmin, ymin, xmax, ymax) for burst_id in burst_ids])
    con.commit()
    con.close()
    return epsg


@contextmanager
def redirect_output(log_file:str):
    """Sends Python and C level (isce3 journal, SOLID) stdout and stderr to log_file."""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    libc = ctypes.CDLL(None)
    sys.stdout.flush(); sys.stderr.flush()
    saved_fds = os.dup(1), os.dup(2)
    with open(log_file, 'w') as log, redirect_stdout(log), redirect_stderr(log):
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
        try:
            yield
        finally:
            libc.fflush(None) # C stdio buffers, otherwise they are written after the fds are restored
            log.flush()
            os.dup2(saved_fds[0], 1)
            os.dup2(saved_fds[1], 2)
            for fd in saved_fds: os.close(fd)


def process_burst(granule:str, burst_id:str, date:str, work_dir:str, dem_file:str, burst_db_file:str, keep_safe:bool = False)->str:
    """Burst granule -> SAFE -> orbit -> COMPASS geocoded CSLC on the AOI grid.

    Args:
        granule (str): ASF burst granule name (e.g. S1_062533_IW3_20261017T175423_VV_F578-BURST)
        burst_id (str): burst id (e.g. 't030_062533_iw3')
        date (str): acquisition date (YYYYMMDD)
        work_dir (str): S1 working directory
        dem_file (str): DEM with ellipsoidal heights
        burst_db_file (str): burst database from write_burst_db
        keep_safe (bool, optional): keep the downloaded SAFE. Defaults to False.

    Returns:
        str: path of the CSLC HDF5 file
    """
    from burst2safe.burst2safe import burst2safe
    from s1reader.s1_orbit import retrieve_orbit_file
    from compass import s1_cslc

    cslc_dir = os.path.join(work_dir, 'cslc')
    cslc_file = os.path.join(cslc_dir, burst_id, date, '{}_{}.h5'.format(burst_id, date))
    if os.path.exists(cslc_file):
        return cslc_file

    safe_dir = os.path.join(work_dir, 'safe', granule)
    for attempt in range(3):
        # burst2safe skips files that exist, so clear partial downloads of interrupted runs or failed attempts
        shutil.rmtree(safe_dir, ignore_errors=True)
        os.makedirs(safe_dir)
        try:
            # burst2safe downloads with asyncio.run(), which fails inside a running event loop
            # (Jupyter), so run it in a worker thread that has none
            with ThreadPoolExecutor(max_workers=1) as executor:
                safe = str(executor.submit(burst2safe, granules=[granule], all_anns=True,
                                           work_dir=Path(safe_dir)).result())
            break
        except Exception as err:
            print('  attempt {} failed for {}: {}'.format(attempt + 1, granule, err))
            time.sleep(30 * (attempt + 1))
    else:
        raise RuntimeError('Could not download {}'.format(granule))

    orbits = retrieve_orbit_file(safe, os.path.join(work_dir, 'orbits', os.path.basename(safe)), concatenate=True)
    orbits = [orbits] if isinstance(orbits, str) else list(orbits)

    runconfig_file = os.path.join(work_dir, 'runconfigs', '{}_{}.yaml'.format(burst_id, date))
    os.makedirs(os.path.dirname(runconfig_file), exist_ok=True)
    with open(runconfig_file, 'w') as f:
        f.write(RUNCONFIG_TEMPLATE.format(safe=safe, orbits=', '.join(orbits), burst_id=burst_id,
                                          dem=dem_file, burst_db=burst_db_file,
                                          product_dir=cslc_dir, scratch_dir=os.path.join(work_dir, 'scratch'),
                                          dx=X_POSTING, dy=Y_POSTING))
    # COMPASS output goes to a log per burst and date, shown only if it fails
    log_file = os.path.join(work_dir, 'logs', '{}_{}.log'.format(burst_id, date))
    try:
        with redirect_output(log_file):
            s1_cslc.run(runconfig_file, 'geo')
    except Exception:
        with open(log_file) as f:
            print(''.join(f.readlines()[-30:]))
        print('  COMPASS failed for {} {}, full log in {}'.format(burst_id, date, log_file))
        raise

    # COMPASS writes its intermediate files to scratch/burst_id/date
    shutil.rmtree(os.path.join(work_dir, 'scratch', burst_id, date), ignore_errors=True)
    if not keep_safe:
        shutil.rmtree(safe_dir)
    return cslc_file


def read_cslc(cslc_file:str, pol:str = 'VV')->dict:
    """Reads a COMPASS (or OPERA) CSLC-S1 HDF5 file.

    Args:
        cslc_file (str): path of CSLC HDF5 file
        pol (str, optional): polarization. Defaults to 'VV'.

    Returns:
        dict: slc (NaN where invalid), x, y, epsg, acquisition datetime, frequency (GHz) and orbit state vectors
    """
    with h5py.File(cslc_file, 'r') as f:
        slc = f['data/' + pol][()].astype(np.complex64)
        slc[slc == 0] = np.nan
        orbit = f['metadata/orbit']
        return {'slc': slc,
                'x': f['data/x_coordinates'][()],
                'y': f['data/y_coordinates'][()],
                'epsg': int(f['data/projection'][()]),
                'datetime': pd.to_datetime(f['identification/zero_doppler_start_time'][()].decode()),
                'freq_GHz': float(f['metadata/processing_information/input_burst_metadata/radar_center_frequency'][()]) / 1e9,
                'orbit_time': orbit['time'][()],
                'orbit_pos': np.stack([orbit['position_' + c][()] for c in 'xyz'], axis=-1),
                'orbit_vel': np.stack([orbit['velocity_' + c][()] for c in 'xyz'], axis=-1)}


def build_s1_stacks(AOI_File:str, time_start:str, time_end:str, out_dir:str, dem_file:str, margin_m:float, pol:str = 'VV', min_acquisitions:int = 5, keep_safe:bool = False)->list:
    """Builds one geocoded SLC stack file per Sentinel-1 track that covers the AOI.

    Bursts of the same pass that cover the AOI are mosaicked. CSLCs already produced are
    reused, so an interrupted run continues where it stopped.

    Args:
        AOI_File (str): path of vector AOI file
        time_start (str): start datetime (YYYYMMDDTHHMMSS, UTC)
        time_end (str): end datetime (YYYYMMDDTHHMMSS, UTC)
        out_dir (str): output directory
        dem_file (str): DEM with ellipsoidal heights (see geocoded_stack.get_dem)
        margin_m (float): margin around the AOI in meters (at least the soil moisture grid size)
        pol (str, optional): polarization. Defaults to 'VV'.
        min_acquisitions (int, optional): tracks with fewer complete acquisitions are skipped. Defaults to 5.
        keep_safe (bool, optional): keep the downloaded bursts. Defaults to False.

    Returns:
        list: paths of the saved stack files
    """
    work_dir = os.path.join(out_dir, 'S1_work')
    os.makedirs(work_dir, exist_ok=True)
    tracks = search_bursts(AOI_File, time_start, time_end, pol)
    burst_db_file = os.path.join(work_dir, 'aoi_burst_db.sqlite3')
    epsg = write_burst_db(AOI_File, sorted({b for dates in tracks.values() for bursts in dates.values() for b in bursts}),
                          burst_db_file, margin_m)
    stack_files = []

    for track, dates in sorted(tracks.items()):
        kept_dates, bursts = select_complete_dates({date: set(by_burst) for date, by_burst in dates.items()})
        print('S1 track {}: {} acquisitions, {} with all bursts {}'.format(track, len(dates), len(kept_dates), bursts))
        if len(kept_dates) < min_acquisitions:
            print('  skipped, fewer than {} acquisitions'.format(min_acquisitions))
            continue

        cslcs = {}
        for date in kept_dates:
            for burst_id in bursts:
                t0 = time.time()
                cslc_file = process_burst(dates[date][burst_id], burst_id, date, work_dir, dem_file, burst_db_file, keep_safe)
                cslcs[(date, burst_id)] = read_cslc(cslc_file, pol)
                print('  {} {} done in {:.0f} s'.format(date, burst_id, time.time() - t0))

        first = cslcs[(kept_dates[0], bursts[0])]
        x, y = first['x'], first['y']
        for cslc in cslcs.values():
            assert cslc['epsg'] == epsg and np.array_equal(cslc['x'], x) and np.array_equal(cslc['y'], y), 'CSLCs are not on one grid'

        source_stacks = [np.stack([cslcs[(date, burst_id)]['slc'] for date in kept_dates]) for burst_id in bursts]
        slc, source_index = mosaic_sources(source_stacks)
        print('  {:.1f}% of the AOI grid has data on every date'.format(100 * np.mean(source_index >= 0)))

        hgt = dem_on_grid(dem_file, x, y, epsg)
        inc = local_incidence_angle(x, y, epsg, hgt, first['orbit_time'], first['orbit_pos'], first['orbit_vel'])
        stack_file = os.path.join(out_dir, 'S1_{}.npz'.format(track))
        save_stack(stack_file, slc, [cslcs[(date, bursts[0])]['datetime'] for date in kept_dates],
                   x, y, epsg, inc, hgt, first['freq_GHz'], 'S1', track)
        stack_files.append(stack_file)

    return stack_files
