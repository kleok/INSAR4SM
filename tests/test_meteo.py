"""Offline tests of the ERA5-Land precipitation handling and the dry-date selection."""
import numpy as np
import pandas as pd
import xarray as xr

from insar4sm.ERA5.Preprocess_ERA5_data import merge_ERA5_land_datasets
from insar4sm.meteo_funcs import find_dry_SARs


def _era5_file(tmp_path, times, tp_acc_m):
    """Writes a small ERA5-Land-like NetCDF (2x2 cells) with accumulated tp in meters."""
    shape = (len(times), 2, 2)
    ds = xr.Dataset({'tp': (('time', 'latitude', 'longitude'), np.broadcast_to(np.asarray(tp_acc_m)[:, None, None], shape).copy()),
                     'skt': (('time', 'latitude', 'longitude'), np.full(shape, 290.0)),
                     'swvl1': (('time', 'latitude', 'longitude'), np.full(shape, 0.1))},
                    coords={'time': times, 'latitude': [33.7, 33.6], 'longitude': [-115.2, -115.1]})
    filename = tmp_path / 'era5.nc'
    ds.to_netcdf(filename)
    return str(filename)


def test_tp_is_deaccumulated_to_hourly(tmp_path):
    # 23 UTC to 03 UTC; ERA5-Land tp accumulates from 00 UTC, the 00 UTC value holds the whole previous day
    times = pd.date_range('2026-01-01 21:00', '2026-01-02 03:00', freq='h')
    hourly_mm = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 0.5, 0.0])
    acc_mm = np.empty_like(hourly_mm)
    acc_mm[:3] = 5.0 + np.cumsum(hourly_mm[:3])    # earlier in day 1 (5 mm before 21 UTC)
    acc_mm[3] = acc_mm[2] + hourly_mm[3]           # 00 UTC: day total
    acc_mm[4:] = np.cumsum(hourly_mm[4:])          # day 2 restarts at 01 UTC
    merged = merge_ERA5_land_datasets([_era5_file(tmp_path, times, acc_mm / 1000)], None)

    tp = merged['tp'].isel(x=0, y=0).values
    assert np.isnan(tp[0])                         # no previous hour available
    assert np.allclose(tp[1:], hourly_mm[1:])
    assert merged['tp'].attrs['units'] == 'mm/hour'


def _hourly_meteo(start, end, rain):
    index = pd.date_range(start, end, freq='h')
    df = pd.DataFrame({'tp': 0.0, 'skt': 290.0, 'swvl1': 0.1}, index=index)
    for when, mm in rain.items():
        df.loc[pd.Timestamp(when), 'tp'] = mm
    return df


def test_days_back_is_days_not_rows():
    # one rain event 10 days before the second acquisition: with a 24-day window it counts,
    # with a 24-row (24 hour) window it would not. It is 26 days before the third acquisition.
    meteo = _hourly_meteo('2025-12-01', '2026-02-01', {'2026-01-05 06:00': 20.0})
    sar_dates = pd.to_datetime(['2025-12-20', '2026-01-15', '2026-01-31'])
    dry_date, dry_dates, meteo_sel = find_dry_SARs(meteo.copy(), sar_dates, days_back=24, orbit_time='13:00:00',
                                                   lowest_temp_K=273.15, tp_quantile=0.5)
    assert meteo_sel['tp'].tolist() == [0.0, 20.0, 0.0]
    assert dry_date == '20251220'
    assert list(dry_dates) == [pd.Timestamp('2025-12-20'), pd.Timestamp('2026-01-31')]


def test_arid_site_without_rain_keeps_dry_dates():
    # no rain at all: the quantile is 0, every date must still be a dry candidate
    meteo = _hourly_meteo('2025-12-01', '2026-02-01', {})
    sar_dates = pd.to_datetime(['2025-12-20', '2026-01-01', '2026-01-13'])
    dry_date, dry_dates, _ = find_dry_SARs(meteo.copy(), sar_dates, days_back=24, orbit_time='13:00:00',
                                           lowest_temp_K=273.15, tp_quantile=0.2)
    assert dry_date == '20251220'
    assert len(dry_dates) == 3


def test_frozen_dates_are_not_dry():
    meteo = _hourly_meteo('2025-12-01', '2026-02-01', {})
    meteo.loc['2025-12-20', 'skt'] = 260.0
    sar_dates = pd.to_datetime(['2025-12-20', '2026-01-01'])
    dry_date, dry_dates, _ = find_dry_SARs(meteo.copy(), sar_dates, days_back=24, orbit_time='13:00:00',
                                           lowest_temp_K=273.15, tp_quantile=0.2)
    assert dry_date == '20260101'
