"""Offline tests of the geocoded-stack helpers (no downloads)."""
import numpy as np
import geopandas as gpd
import pytest
from pyproj import Transformer
from shapely.geometry import box

from insar4sm.geocoded_stack import (_lattice, aoi_grid, select_complete_dates,
                                     mosaic_sources, local_incidence_angle)


def test_lattice_positive_and_negative_step():
    x = _lattice(100.0, 130.0, ref=2.5, step=5.0)
    assert np.allclose(x, [102.5, 107.5, 112.5, 117.5, 122.5, 127.5])
    # north-up y axis: negative step, values decrease
    y = _lattice(100.0, 130.0, ref=2.5, step=-5.0)
    assert np.allclose(y, [127.5, 122.5, 117.5, 112.5, 107.5, 102.5])


def test_aoi_grid_covers_aoi_with_margin(tmp_path):
    aoi_file = tmp_path / 'aoi.geojson'
    gpd.GeoDataFrame(geometry=[box(-115.11, 33.645, -115.085, 33.665)], crs='EPSG:4326').to_file(aoi_file)
    epsg = 32611
    x, y = aoi_grid(str(aoi_file), epsg, x_ref=0.0, dx=5.0, y_ref=0.0, dy=-10.0, margin_m=250)
    xmin, ymin, xmax, ymax = gpd.read_file(aoi_file).to_crs(epsg=epsg).total_bounds
    assert x[0] <= xmin - 245 and x[-1] >= xmax + 245
    assert y[0] >= ymax + 240 and y[-1] <= ymin - 240
    assert np.allclose(np.diff(x), 5.0) and np.allclose(np.diff(y), -10.0)
    # pixel centres stay on the native lattice
    assert np.allclose(np.round(x / 5.0) * 5.0, x) and np.allclose(np.round(y / 10.0) * 10.0, y)


def test_select_complete_dates_drops_dates_missing_a_frequent_source():
    sources = {'20260101': {'a', 'b'},
               '20260113': {'a', 'b'},
               '20260125': {'a'},          # misses frequent source b -> dropped
               '20260206': {'a', 'b', 'c'},  # rare source c is not required
               '20260218': {'a', 'b'}}
    dates, required = select_complete_dates(sources, min_presence=0.8)
    assert required == ['a', 'b']
    assert dates == ['20260101', '20260113', '20260206', '20260218']


def test_mosaic_keeps_one_source_per_pixel():
    nan = np.nan + 1j * np.nan
    s0 = np.ones((3, 2, 2), dtype=np.complex64)
    s0[1, 0, 0] = nan          # pixel (0,0) of source 0 invalid on one date
    s0[:, 1, 1] = 0            # pixel (1,1) of source 0 has no data
    s1 = 2 * np.ones((3, 2, 2), dtype=np.complex64)
    s1[:, 1, 1] = nan          # pixel (1,1) has no valid source at all
    mosaic, index = mosaic_sources([s0, s1])
    assert index.tolist() == [[1, 0], [0, -1]]
    # pixel (0,0) takes the whole time series from source 1, never mixes sources
    assert np.all(mosaic[:, 0, 0] == 2)
    assert np.all(mosaic[:, 0, 1] == 1)
    assert np.all(np.isnan(mosaic[:, 1, 1]))


def test_local_incidence_angle_flat_terrain():
    """Straight orbit at height h, ground offset d to the east: incidence = atan(d/h) on flat ground."""
    lon0, lat0, epsg = -115.1, 33.655, 32611
    x0, y0 = Transformer.from_crs(4326, epsg, always_xy=True).transform(lon0, lat0)
    x = x0 + np.arange(-200, 201, 50.0)
    y = y0 + np.arange(200, -201, -50.0)
    hgt = np.zeros((len(y), len(x)))

    lo, la = np.radians(lon0), np.radians(lat0)
    east = np.array([-np.sin(lo), np.cos(lo), 0])
    north = np.array([-np.sin(la) * np.cos(lo), -np.sin(la) * np.sin(lo), np.cos(la)])
    up = np.array([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])
    target = np.array(Transformer.from_crs(4979, 4978, always_xy=True).transform(lon0, lat0, 0.0))

    h, d, v = 700e3, 400e3, 7000.0
    t = np.arange(-60, 61, 10.0)
    pos = target + up * h + east * d + np.outer(t, north) * v
    vel = np.tile(north * v, (len(t), 1))

    inc = local_incidence_angle(x, y, epsg, hgt, t, pos, vel)
    centre = inc[inc.shape[0] // 2, inc.shape[1] // 2]
    assert centre == pytest.approx(np.degrees(np.arctan(d / h)), abs=0.1)
    # a slope facing the satellite (rising to the west) lowers the local incidence angle
    X, _ = np.meshgrid(x - x0, y)
    inc_slope = local_incidence_angle(x, y, epsg, -0.1 * X, t, pos, vel)
    assert inc_slope[inc.shape[0] // 2, inc.shape[1] // 2] < centre - 3
