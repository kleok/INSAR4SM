#!/usr/bin/env python3
"""End-to-end smoke test of InSAR4SM on the Ford Dry Lake SCAN station (2 x 2 km AOI, ~2 months).

Downloads ERA5-Land (CDS, ~/.cdsapirc), NISAR GSLC windows and Sentinel-1 bursts (NASA Earthdata,
~/.netrc), builds one stack per track, inverts soil moisture and compares it with the ISMN
5 cm in-situ soil moisture of the station.

The run is resumable: downloaded data are reused on a rerun.

Usage:
    python tests/smoke/run_ford_dry_lake.py --project-dir /media/kle/cropg_ssd1/INSAR4SM_tests/FordDryLake \
        --ismn-csv /path/to/sm_FordDryLake_..._0.0508_0.0508.csv [--sensors S1 NISAR]
"""
import os
import sys
import json
import argparse
import datetime

import numpy as np
import pandas as pd
import geopandas as gpd
from joblib import Parallel, delayed
from tqdm import tqdm
from scipy.stats import pearsonr

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO_DIR)

from insar4sm.ERA5.ERA5_class import ERA5
from insar4sm.geocoded_stack import GEOCODED_stack, get_dem, track_summary
from insar4sm.joblib_progress_bar import tqdm_joblib
from insar4sm.sm_estimation_funcs import sm_estimation

# ----------------------------- test case ------------------------------
AOI_FILE = os.path.join(REPO_DIR, 'tests', 'data', 'FordDryLake_aoi.geojson')
STATION = {'name': 'FordDryLake', 'lon': -115.09763, 'lat': 33.65470}
# end on a full hour: ERA5-Land is hourly, so a 23:59:59 end looks like a missing hour on every rerun
TIME_START, TIME_END = '20251201T000000', '20260131T230000'
# ERA5-Land starts days_back earlier, so the first acquisitions get a full precipitation window
DAYS_BACK = 24
# ISMN station soil texture (sand 79 %, clay 11 %), as np.int64 like the notebook
SAND, CLAY = np.int64(79), np.int64(11)
GRID_SIZE = 125          # half resolution of the soil moisture grid (m)
DS_FLAG = False          # short stacks: amplitude similarity test has too few samples
MIN_ACQUISITIONS = 4
DRY_DATE, DRY_DATE_MANUAL_FLAG = None, False
TP_QUANTILE = 0.2


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--project-dir', required=True, help='where downloads and outputs are stored')
    parser.add_argument('--ismn-csv', required=True, help='ISMN 5 cm soil moisture CSV (date_time, soil_moisture)')
    parser.add_argument('--sensors', nargs='+', default=['S1', 'NISAR'], choices=['S1', 'NISAR'])
    parser.add_argument('--s1-source', default='OPERA', choices=['OPERA', 'COMPASS'],
                        help='OPERA: JPL geocoded bursts, North America, no isce3. COMPASS: local geocoding, global, needs isce3')
    parser.add_argument('--n-cpus', type=int, default=max(1, (os.cpu_count() or 2) - 2))
    return parser.parse_args()


def download_era5(project_dir):
    era5_start = (pd.to_datetime(TIME_START) - pd.Timedelta(days=DAYS_BACK)).strftime('%Y%m%dT%H%M%S')
    era5 = ERA5({'projectfolder': project_dir,
                 'time_start': era5_start,
                 'time_end': TIME_END,
                 'AOI_File': AOI_FILE,
                 'ERA5_variables': ['volumetric_soil_water_layer_1', 'total_precipitation', 'skin_temperature']})
    era5.get_download_dates()
    era5.download_ERA5_data()
    era5.create_ERA5_polygons()
    return era5.era5_merged_filename


def build_stacks(project_dir, sensors, s1_source):
    import earthaccess
    earthaccess.login(strategy='netrc')
    stack_dir = os.path.join(project_dir, 'SAR_stacks')
    dem_file = get_dem(AOI_FILE, os.path.join(project_dir, 'DEM', 'dem.tif'))
    stack_files = []
    if 'NISAR' in sensors:
        from insar4sm.NISAR.nisar_gslc import build_nisar_stacks
        stack_files += build_nisar_stacks(AOI_FILE, TIME_START, TIME_END, os.path.join(stack_dir, 'NISAR'),
                                          dem_file, margin_m=2 * GRID_SIZE, min_acquisitions=MIN_ACQUISITIONS)
    if 'S1' in sensors and s1_source == 'OPERA':
        from insar4sm.S1.opera_cslc import build_opera_stacks
        stack_files += build_opera_stacks(AOI_FILE, TIME_START, TIME_END, os.path.join(stack_dir, 'S1_OPERA'),
                                          dem_file, margin_m=2 * GRID_SIZE, min_acquisitions=MIN_ACQUISITIONS)
    if 'S1' in sensors and s1_source == 'COMPASS':
        from insar4sm.S1.s1_cslc import build_s1_stacks  # needs the INSAR4SM_env_compass.yml environment
        stack_files += build_s1_stacks(AOI_FILE, TIME_START, TIME_END, os.path.join(stack_dir, 'S1'),
                                       dem_file, margin_m=2 * GRID_SIZE, min_acquisitions=MIN_ACQUISITIONS)
    return stack_files


def invert_stack(stack_file, meteo_file, export_dir, n_cpus):
    stack = GEOCODED_stack(stack_file=stack_file, projectname='FordDryLake_smoke', n_CPUs=n_cpus,
                           AOI=AOI_FILE, meteo_file=meteo_file, sand=SAND, clay=CLAY, days_back=DAYS_BACK,
                           export_dir=os.path.join(export_dir, os.path.basename(stack_file).split('.')[0]))
    print('------ {} track {}'.format(stack.sensor, stack.track))
    stack.prepare_datasets(lowest_temp_K=273.15)
    stack.plot()
    stack.get_dry_SARs(lowest_temp_K=273.15, tp_quantile=TP_QUANTILE)
    stack.calc_insar_stack()
    stack.calc_grid(grid_size=GRID_SIZE)

    with tqdm_joblib(tqdm(desc='SM inversions', total=stack.n_sm_points)):
        estimations = Parallel(n_jobs=stack.CPUs)(delayed(sm_estimation)(stack, sm_ind, DRY_DATE_MANUAL_FLAG, DRY_DATE, DS_FLAG)
                                                  for sm_ind in range(stack.n_sm_points))
    columns = [d.strftime('D%Y%m%d') for d in stack.slc_datetimes]
    sm_df = pd.DataFrame(estimations, index=range(stack.n_sm_points), columns=columns)
    gdf = gpd.GeoDataFrame(sm_df.copy(), geometry=stack.sm_points.values, crs='EPSG:4326')
    gdf.to_file(os.path.join(stack.export_dir, 'sm_inversions_{}.gpkg'.format(GRID_SIZE)))
    return stack, gdf


def compare_with_ismn(stack, gdf, ismn):
    """Station cell and AOI mean vs ISMN 5 cm soil moisture at the acquisition hour (all in %)."""
    date_cols = [c for c in gdf.columns if c.startswith('D')]
    utm = gdf.estimate_utm_crs()
    station = gpd.GeoSeries(gpd.points_from_xy([STATION['lon']], [STATION['lat']]), crs='EPSG:4326').to_crs(utm).iloc[0]
    station_cell = gdf.to_crs(utm).distance(station).idxmin()
    orbit_hour = int(stack.orbit_time[:2])

    rows = []
    for col in date_cols:
        t = pd.Timestamp(col[1:]) + pd.Timedelta(hours=orbit_hour)
        insitu = ismn.asof(t) * 100 if t >= ismn.index[0] else np.nan
        rows.append({'date': t.date(), 'insitu_5cm': insitu,
                     'insar4sm_station_cell': gdf.loc[station_cell, col],
                     'insar4sm_aoi_mean': gdf[col].mean(skipna=True),
                     'era5_swvl1': stack.meteo_sel_df['swvl1'].get(t.date(), np.nan) * 100})
    df = pd.DataFrame(rows)

    scores = {'sensor': stack.sensor, 'track': stack.track, 'n_dates': len(df),
              'valid_cells_fraction': float(gdf[date_cols].notna().all(axis=1).mean())}
    for est in ['insar4sm_station_cell', 'insar4sm_aoi_mean', 'era5_swvl1']:
        ok = df[[est, 'insitu_5cm']].dropna()
        scores['RMSE_' + est] = float(np.sqrt(np.mean((ok[est] - ok['insitu_5cm'])**2))) if len(ok) else np.nan
        scores['R_' + est] = float(pearsonr(ok[est], ok['insitu_5cm'])[0]) if len(ok) > 2 and ok[est].std() > 0 else np.nan
    return df, scores


def main():
    args = parse_args()
    os.makedirs(args.project_dir, exist_ok=True)
    export_dir = os.path.join(args.project_dir, 'SSM')
    started = datetime.datetime.now()

    ismn = pd.read_csv(args.ismn_csv, parse_dates=['date_time']).set_index('date_time')['soil_moisture'].sort_index()

    meteo_file = download_era5(args.project_dir)
    stack_files = build_stacks(args.project_dir, args.sensors, args.s1_source)
    assert stack_files, 'no track with at least {} acquisitions'.format(MIN_ACQUISITIONS)
    print(pd.DataFrame([track_summary(f) for f in stack_files]).to_string(index=False))

    all_scores = []
    for stack_file in stack_files:
        stack, gdf = invert_stack(stack_file, meteo_file, export_dir, args.n_cpus)
        df, scores = compare_with_ismn(stack, gdf, ismn)
        df.to_csv(os.path.join(stack.export_dir, 'comparison_ismn_{}.csv'.format(GRID_SIZE)), index=False)
        print(df.round(2).to_string(index=False))
        all_scores.append(scores)

    scores_df = pd.DataFrame(all_scores)
    scores_df.to_csv(os.path.join(export_dir, 'scores_ford_dry_lake.csv'), index=False)
    print(scores_df.round(3).to_string(index=False))
    with open(os.path.join(export_dir, 'run_info.json'), 'w') as f:
        json.dump({'aoi': AOI_FILE, 'period': [TIME_START, TIME_END], 'sensors': args.sensors, 's1_source': args.s1_source,
                   'stack_files': stack_files, 'runtime_min': (datetime.datetime.now() - started).seconds / 60}, f, indent=2)


if __name__ == '__main__':
    main()
