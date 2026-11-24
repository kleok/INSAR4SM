#!/usr/bin/env python3
"""Collects the outputs of the Ford Dry Lake 12-month run into small CSV files in results/.

They hold InSAR4SM and ERA5-Land results only, no in-situ data: ISMN data cannot be redistributed,
so the notebook and the report read the ISMN file that each user downloads. Run this once after
the processing run:

    python tests/smoke/run_ford_dry_lake.py --project-dir <RUN_DIR> --ismn-csv <ISMN_CSV> \
        --time-start 20251122T000000 --time-end 20261111T230000 --ds --min-acquisitions 10
    python case_studies/FordDryLake/prepare_results.py --run-dir <RUN_DIR>
"""
import os
import sys
import glob
import json
import argparse

import numpy as np
import pandas as pd
import geopandas as gpd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, REPO_DIR)

from insar4sm.geocoded_stack import track_summary
from insar4sm.prep_meteo import convert_to_df

AOI_FILE = os.path.join(REPO_DIR, 'tests', 'data', 'FordDryLake_aoi.geojson')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run-dir', required=True, help='--project-dir of run_ford_dry_lake.py')
    parser.add_argument('--out-dir', default=os.path.join(HERE, 'results'))
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    ssm_dir = os.path.join(args.run_dir, 'SSM')
    with open(os.path.join(ssm_dir, 'run_info.json')) as f:
        run_info = json.load(f)

    # 1. track summaries (acquisitions, revisit, coherence, incidence, AOI coverage)
    tracks = pd.DataFrame([track_summary(f) for f in run_info['stack_files']])
    orbit_hour = {}
    for stack_file in run_info['stack_files']: # hour used for ERA5-Land and the in-situ comparison
        with np.load(stack_file) as data:
            t = pd.to_datetime(data['datetimes'])
            orbit_hour[(str(data['sensor']), str(data['track']))] = int(np.round(np.mean(t.hour + t.minute / 60))) % 24
    tracks.to_csv(os.path.join(args.out_dir, 'tracks.csv'), index=False)

    # 2. InSAR4SM (station cell and AOI mean) and ERA5-Land at every acquisition
    comparisons, estimates = [], []
    for track_dir in sorted(glob.glob(os.path.join(ssm_dir, '*_*'))):
        if not os.path.isdir(track_dir):
            continue
        sensor, track = os.path.basename(track_dir).split('_')
        comp = pd.read_csv(os.path.join(track_dir, 'comparison_ismn_125.csv'), parse_dates=['date'])
        comp = comp.drop(columns='insitu_5cm') # ISMN values are not shared
        comp['date'] = comp['date'] + pd.Timedelta(hours=orbit_hour[(sensor, track)])
        comp = comp.rename(columns={'date': 'time_utc'})
        comp.insert(0, 'track', track)
        comp.insert(0, 'sensor', sensor)
        comparisons.append(comp)

        # 3. per grid cell estimates (long format)
        gdf = gpd.read_file(os.path.join(track_dir, 'sm_inversions_125.gpkg'))
        date_cols = [c for c in gdf.columns if c.startswith('D')]
        wide = pd.DataFrame({'cell': gdf.index, 'lon': gdf.geometry.x.round(6), 'lat': gdf.geometry.y.round(6)})
        wide = pd.concat([wide, gdf[date_cols].reset_index(drop=True)], axis=1)
        long = wide.melt(id_vars=['cell', 'lon', 'lat'], var_name='date', value_name='sm_pct')
        long['date'] = pd.to_datetime(long['date'].str[1:])
        long.insert(0, 'track', track)
        long.insert(0, 'sensor', sensor)
        estimates.append(long.dropna(subset=['sm_pct']).round({'sm_pct': 2}))

    pd.concat(comparisons).round(3).to_csv(os.path.join(args.out_dir, 'insar4sm_era5.csv'), index=False)
    pd.concat(estimates).to_csv(os.path.join(args.out_dir, 'estimates.csv.gz'), index=False, compression='gzip')

    # 4. daily context: ERA5-Land AOI daily mean soil moisture and precipitation
    meteo = convert_to_df(os.path.join(args.run_dir, 'ERA5', 'merged_ERA5.nc'), AOI_FILE)
    meteo.index = pd.to_datetime(meteo.index)
    start, end = pd.Timestamp(run_info['period'][0]).normalize(), pd.Timestamp(run_info['period'][1]).normalize()
    daily = pd.DataFrame({'era5_swvl1_pct': meteo['swvl1'].resample('D').mean() * 100,
                          'era5_precip_mm': meteo['tp'].resample('D').sum(min_count=1)}).loc[start:end]
    daily.index.name = 'date'
    daily.round(3).to_csv(os.path.join(args.out_dir, 'era5_daily.csv'))

    # run settings without local paths: AOI relative to the repository, stack files by name only
    shared_info = dict(run_info, aoi=os.path.relpath(run_info['aoi'], REPO_DIR),
                       stack_files=[os.path.basename(f) for f in run_info['stack_files']])
    with open(os.path.join(args.out_dir, 'run_info.json'), 'w') as f:
        json.dump(shared_info, f, indent=2)
    print('results written to', args.out_dir)
    for f in sorted(os.listdir(args.out_dir)):
        print('  {:<24} {:>8.1f} kB'.format(f, os.path.getsize(os.path.join(args.out_dir, f)) / 1e3))


if __name__ == '__main__':
    main()
