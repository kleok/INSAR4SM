#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Geocoded SLC stacks (NISAR GSLC, Sentinel-1 CSLC from COMPASS) for InSAR4SM.

Geocoded SLCs come coregistered on a map grid, so GEOCODED_stack builds the SLC cube,
lat/lon, incidence and height grids and the primary index straight from a saved stack
file, and every downstream InSAR4SM step (DS selection, sorting, coherence model,
inversion) runs on them.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import rasterio
from rasterio.warp import reproject, Resampling
from pyproj import Transformer
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq

from insar4sm.classes import INSAR4SM_stack
from insar4sm.prep_meteo import convert_to_df


def _lattice(vmin:float, vmax:float, ref:float, step:float)->np.array:
    """Pixel centres ref + k*step that fall inside [vmin, vmax], in the order of the step sign.
    """
    k = np.sort([(vmin - ref) / step, (vmax - ref) / step])
    return ref + step * np.arange(np.ceil(k[0]), np.floor(k[1]) + 1)


def aoi_grid(AOI_File:str, epsg:int, x_ref:float, dx:float, y_ref:float, dy:float, margin_m:float)->tuple[np.array, np.array]:
    """Map grid that covers the AOI (plus a margin) on the native pixel lattice of a sensor.

    Args:
        AOI_File (str): path of vector AOI file
        epsg (int): EPSG code of the grid
        x_ref (float): x of any pixel centre of the native lattice
        dx (float): pixel spacing in x (meters)
        y_ref (float): y of any pixel centre of the native lattice
        dy (float): pixel spacing in y (meters, negative for north-up)
        margin_m (float): margin around the AOI bounds in meters

    Returns:
        tuple[np.array, np.array]: x and y pixel centres of the grid
    """
    xmin, ymin, xmax, ymax = gpd.read_file(AOI_File).to_crs(epsg=epsg).total_bounds
    x = _lattice(xmin - margin_m, xmax + margin_m, x_ref, dx)
    y = _lattice(ymin - margin_m, ymax + margin_m, y_ref, dy)
    return x, y


def select_complete_dates(sources_per_date:dict, min_presence:float = 0.8)->tuple[list, list]:
    """Selects the acquisition dates that contain every frequently present source (burst or frame).

    A pixel keeps one source for all dates (see mosaic_sources), so a date that misses a
    source would leave a hole for the whole stack. Sources present in at least min_presence
    of the dates are required, rarer ones are ignored.

    Args:
        sources_per_date (dict): {date: set of source ids}
        min_presence (float, optional): fraction of dates a source must appear in to be required. Defaults to 0.8.

    Returns:
        tuple[list, list]: sorted kept dates, sorted required source ids
    """
    n_dates = len(sources_per_date)
    counts = pd.Series([s for sources in sources_per_date.values() for s in sources]).value_counts()
    required = sorted(counts[counts >= min_presence * n_dates].index)
    dates = sorted(d for d, sources in sources_per_date.items() if set(required) <= set(sources))
    return dates, required


def mosaic_sources(source_stacks:list)->tuple[np.array, np.array]:
    """Mosaics stacks of several sources (bursts or frames) that share the same map grid.

    Each pixel takes its values from the first source that is valid at that pixel on every
    date, so the time series of a pixel never switches source between dates.

    Args:
        source_stacks (list): 3D arrays (SARs, Ys, Xs) of complex values, NaN where invalid

    Returns:
        tuple[np.array, np.array]: mosaicked 3D array (NaN where no source is valid on all dates), 2D array of the index of the source used per pixel (-1 where none)
    """
    mosaic = np.full(source_stacks[0].shape, np.nan + 1j * np.nan, dtype=np.complex64)
    source_index = np.full(source_stacks[0].shape[1:], -1, dtype=np.int16)
    for ind, stack in enumerate(source_stacks):
        valid = np.all(np.isfinite(stack) & (stack != 0), axis=0) & (source_index < 0)
        mosaic[:, valid] = stack[:, valid]
        source_index[valid] = ind
    return mosaic, source_index


def get_dem(AOI_File:str, dem_file:str, margin_deg:float = 0.1)->str:
    """Downloads the Copernicus GLO-30 DEM (ellipsoidal heights) around the AOI, if not already on disk.

    Args:
        AOI_File (str): path of vector AOI file
        dem_file (str): output GeoTIFF path
        margin_deg (float, optional): margin around the AOI bounds in degrees. Defaults to 0.1.

    Returns:
        str: path of the DEM GeoTIFF
    """
    if not os.path.exists(dem_file):
        from dem_stitcher import stitch_dem
        lon_min, lat_min, lon_max, lat_max = gpd.read_file(AOI_File).to_crs(epsg=4326).total_bounds
        bounds = [lon_min - margin_deg, lat_min - margin_deg, lon_max + margin_deg, lat_max + margin_deg]
        dem, profile = stitch_dem(bounds, dem_name='glo_30', dst_ellipsoidal_height=True, dst_area_or_point='Point')
        os.makedirs(os.path.dirname(dem_file), exist_ok=True)
        with rasterio.open(dem_file, 'w', **profile) as dst:
            dst.write(dem, 1)
    return dem_file


def dem_on_grid(dem_file:str, x:np.array, y:np.array, epsg:int)->np.array:
    """Resamples a DEM (ellipsoidal heights) to a map grid.

    Args:
        dem_file (str): path of DEM GeoTIFF
        x (np.array): x pixel centres of the grid
        y (np.array): y pixel centres of the grid
        epsg (int): EPSG code of the grid

    Returns:
        np.array: 2D array (Ys, Xs) of heights in meters
    """
    dx, dy = x[1] - x[0], y[1] - y[0]
    transform = rasterio.transform.from_origin(x[0] - dx / 2, y[0] - dy / 2, dx, -dy)
    hgt = np.zeros((len(y), len(x)), dtype=np.float64)
    with rasterio.open(dem_file) as src:
        reproject(rasterio.band(src, 1), hgt, dst_transform=transform,
                  dst_crs='EPSG:{}'.format(epsg), resampling=Resampling.bilinear)
    return hgt


def local_incidence_angle(x:np.array, y:np.array, epsg:int, hgt:np.array, orbit_time:np.array, orbit_pos:np.array, orbit_vel:np.array)->np.array:
    """Local (terrain) incidence angle on a map grid from the orbit state vectors and a DEM.

    The satellite position is taken at the zero-Doppler time of the grid centre; over an AOI
    of a few km the squint this ignores is far below 1 degree.

    Args:
        x (np.array): x pixel centres of the grid
        y (np.array): y pixel centres of the grid
        epsg (int): EPSG code of the grid
        hgt (np.array): 2D array (Ys, Xs) of ellipsoidal heights
        orbit_time (np.array): times of the state vectors (seconds, any epoch)
        orbit_pos (np.array): (n, 3) ECEF positions of the state vectors
        orbit_vel (np.array): (n, 3) ECEF velocities of the state vectors

    Returns:
        np.array: 2D array (Ys, Xs) of local incidence angles in degrees
    """
    X, Y = np.meshgrid(x, y)
    lon, lat = Transformer.from_crs(epsg, 4326, always_xy=True).transform(X, Y)
    to_ecef = Transformer.from_crs(4979, 4978, always_xy=True)
    tx, ty, tz = to_ecef.transform(lon, lat, hgt)
    target = np.stack([tx, ty, tz], axis=-1)

    # zero-Doppler time of the grid centre: (P(t) - T).V(t) = 0
    pos, vel = CubicSpline(orbit_time, orbit_pos), CubicSpline(orbit_time, orbit_vel)
    centre = target[target.shape[0] // 2, target.shape[1] // 2]
    t0 = brentq(lambda t: np.dot(pos(t) - centre, vel(t)), orbit_time[0], orbit_time[-1])
    los = pos(t0) - target
    los /= np.linalg.norm(los, axis=-1, keepdims=True)

    # terrain normal in local east-north-up, rotated to ECEF
    dz_dy, dz_dx = np.gradient(hgt, y[1] - y[0], x[1] - x[0])
    n_enu = np.stack([-dz_dx, -dz_dy, np.ones_like(hgt)], axis=-1)
    n_enu /= np.linalg.norm(n_enu, axis=-1, keepdims=True)
    lo, la = np.radians(lon), np.radians(lat)
    east = np.stack([-np.sin(lo), np.cos(lo), np.zeros_like(lo)], axis=-1)
    north = np.stack([-np.sin(la) * np.cos(lo), -np.sin(la) * np.sin(lo), np.cos(la)], axis=-1)
    up = np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], axis=-1)
    normal = n_enu[..., :1] * east + n_enu[..., 1:2] * north + n_enu[..., 2:] * up

    return np.degrees(np.arccos(np.clip(np.sum(los * normal, axis=-1), -1, 1)))


def track_summary(stack_file:str, looks:tuple = (9, 9))->dict:
    """Quality indicators of a stack, to compare tracks before running the inversion.

    Args:
        stack_file (str): path of stack file (see save_stack)
        looks (tuple, optional): boxcar window (Ys, Xs) for the coherence estimate. Defaults to (9, 9).

    Returns:
        dict: sensor, track, number of acquisitions, period, median revisit (days), median coherence of consecutive acquisitions, median local incidence angle, fraction of the AOI grid with data
    """
    from scipy.ndimage import uniform_filter
    with np.load(stack_file) as data:
        slc, datetimes = np.nan_to_num(data['slc']), pd.to_datetime(data['datetimes'])
        sensor, track, inc = str(data['sensor']), str(data['track']), data['incLocal']

    def box(a):
        return uniform_filter(a.real, looks) + 1j * uniform_filter(a.imag, looks)

    valid = np.all(slc != 0, axis=0)
    cohs = []
    for s1, s2 in zip(slc[:-1], slc[1:]):
        coh = np.abs(box(s1 * np.conj(s2))) / np.sqrt(uniform_filter(np.abs(s1)**2, looks) * uniform_filter(np.abs(s2)**2, looks) + 1e-12)
        cohs.append(np.median(coh[valid]))

    return {'sensor': sensor,
            'track': track,
            'n_acquisitions': len(datetimes),
            'first': datetimes[0].strftime('%Y%m%d'),
            'last': datetimes[-1].strftime('%Y%m%d'),
            'revisit_days': float(np.median(np.diff(datetimes.normalize()).astype('timedelta64[D]').astype(int))),
            'median_consecutive_coherence': float(np.median(cohs)),
            'median_incLocal': float(np.nanmedian(inc[valid])),
            'valid_fraction': float(np.mean(valid))}


def save_stack(filename:str, slc:np.array, datetimes:list, x:np.array, y:np.array, epsg:int, incLocal:np.array, hgt:np.array, freq_GHz:float, sensor:str, track:str)->None:
    """Saves a geocoded SLC stack to a compressed npz file (input of GEOCODED_stack).
    """
    np.savez_compressed(filename, slc=slc, datetimes=np.array([str(d) for d in datetimes]),
                        x=x, y=y, epsg=epsg, incLocal=incLocal, hgt=hgt, freq_GHz=freq_GHz,
                        sensor=sensor, track=track)


class GEOCODED_stack(INSAR4SM_stack):
    """InSAR4SM stack built from a geocoded SLC stack file (see save_stack) instead of an ISCE2 Topstack.
    """
    def __init__(self, stack_file:str, projectname:str, n_CPUs:int, AOI:str, meteo_file:str, sand, clay, days_back:int, export_dir:str):

        with np.load(stack_file) as data:
            self.stack_data = {key: data[key] for key in data.files}
        self.stack_file = stack_file
        self.sensor = str(self.stack_data['sensor'])
        self.track = str(self.stack_data['track'])
        acq_datetimes = pd.to_datetime(self.stack_data['datetimes'])

        # ERA5-Land is hourly: use the hour closest to the mean pass time of the track
        mean_hours = np.mean(acq_datetimes.hour + acq_datetimes.minute / 60)
        orbit_time = '{:02d}:00:00'.format(int(np.round(mean_hours)) % 24)

        super().__init__(projectname = projectname,
                         n_CPUs = n_CPUs,
                         AOI = AOI,
                         meteo_file = meteo_file,
                         sand = sand,
                         clay = clay,
                         orbit_time = orbit_time,
                         days_back = days_back,
                         export_dir = export_dir)
        self.freq_GHz = float(self.stack_data['freq_GHz'])

    def prepare_datasets(self, lowest_temp_K:float = 273.15):
        """Selects the acquisitions of the stack and drops those with frozen soil.

        Args:
            lowest_temp_K (float, optional): acquisitions with ERA5-Land skin temperature at or below this value are dropped, since the dielectric model does not hold for frozen soil. Defaults to 273.15.
        """
        acq_dates = pd.to_datetime(self.stack_data['datetimes']).normalize()

        meteo_df = convert_to_df(self.meteo_file, self.AOI)
        meteo_df.index = pd.to_datetime(meteo_df.index)
        skt = meteo_df['skt'].at_time(self.orbit_time)
        skt.index = skt.index.normalize()
        keep = np.array([skt.get(acq_date, np.nan) > lowest_temp_K for acq_date in acq_dates])
        if not np.all(keep):
            print('Dropping {} acquisitions with frozen soil or no ERA5 data: {}'.format(
                np.sum(~keep), [d.strftime('%Y%m%d') for d in acq_dates[~keep]]))
        self.keep_bands = np.where(keep)[0]

        self.slc_dates = [acq_dates[ind].strftime('%Y%m%d') for ind in self.keep_bands]
        self.slc_datetimes = pd.to_datetime(self.slc_dates)
        self.Start_time = self.slc_dates[0]
        self.End_time = self.slc_dates[-1]
        self.Start_time_str = self.Start_time
        self.End_time_str = self.End_time
        self.start_index = 0
        self.end_index = len(self.slc_dates)
        self.nbands = self.end_index - self.start_index
        self.n_dry_bands = int(self.nbands*self.dry_dates_ratio)

    def plot(self):
        """Plots mean amplitude and local incidence angle of the stack.
        """
        layers = {'mean_amplitude_dB': 20*np.log10(np.nanmean(np.abs(self.stack_data['slc'][self.keep_bands]), axis=0)),
                  'incLocal': self.stack_data['incLocal']}
        for name, layer in layers.items():
            fig = plt.figure(figsize=(6, 5))
            plt.imshow(layer)
            plt.colorbar(orientation="horizontal")
            plt.title('{} {} {}'.format(self.sensor, self.track, name))
            plt.savefig(os.path.join(self.plot_dir, name), dpi=100)
            plt.close()

    def calc_insar_stack(self):
        """Loads the geocoded SLC stack and its geometry.
        """
        self.primary_index = self.slc_dates.index(self.dry_date_sel)

        X, Y = np.meshgrid(self.stack_data['x'], self.stack_data['y'])
        self.lon_data, self.lat_data = Transformer.from_crs(int(self.stack_data['epsg']), 4326, always_xy=True).transform(X, Y)
        self.incLocal = self.stack_data['incLocal']
        self.hgt = self.stack_data['hgt']
        # pixels without data are zero in Topstack SLCs too; NaN would spoil the DS covariance
        self.slc_stack = np.nan_to_num(self.stack_data['slc'][self.keep_bands])
        self.amp_stack = np.abs(self.slc_stack)
        # geocoded SLCs are flattened with the DEM, so no baseline term is left to model
        self.perp_baseline_stack = np.zeros(self.slc_stack.shape, dtype=np.float32)
