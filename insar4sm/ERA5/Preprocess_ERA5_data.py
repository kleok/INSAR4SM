import numpy as np
import os
import xarray as xr
import zipfile
import cfgrib
import tempfile
import pandas as pd

def read_grib_from_zip(zip_path, grib_filename):
    """
    Extracts a GRIB file from a ZIP archive to a temporary location and reads it with cfgrib.

    Parameters:
        zip_path (str): Path to the ZIP file.
        grib_filename (str): Name of the GRIB file inside the ZIP archive.

    Returns:
        xarray.Dataset: Merged dataset from the GRIB file.
    """
    with zipfile.ZipFile(zip_path, "r") as z:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_grib_path = os.path.join(temp_dir, grib_filename)

            # Extract GRIB file to temporary directory
            z.extract(grib_filename, path=temp_dir)

            # Read all GRIB subsets on the valid time. Accumulated variables (tp) are otherwise
            # indexed by forecast reference time and step, instantaneous ones by time
            datasets = cfgrib.open_datasets(temp_grib_path, backend_kwargs={'time_dims': ('valid_time',)})

            for i, ds in enumerate(datasets):
                if 'valid_time' not in ds.dims: # single time request
                    ds = ds.expand_dims('valid_time')
                datasets[i] = ds.drop_vars(['time', 'step'], errors='ignore').rename({'valid_time': 'time'})

            merged_ds = xr.merge(datasets, compat="override", join="outer")
            return merged_ds.load()
        

def read_ERA5_land_datasets(ERA5_datasets):
    """This function reads NetCDF and zipped GRIB ERA5-Land files.

    Args:
        ERA5_datasets (str): filenames of the ERA5 datasets

    Returns:
        list: list of xarray datasets
    """
    ERA5_xarrays = []
    for ERA5_dataset in ERA5_datasets:
        try:
            # Force using netcdf4 engine for .nc files
            temp_ERA5_xr = xr.open_dataset(ERA5_dataset, engine='netcdf4')
            ERA5_xarrays.append(temp_ERA5_xr.load())
        except Exception as e:
            # Only if really not NetCDF, treat as zip containing GRIB
            temp_ERA5_xr = read_grib_from_zip(ERA5_dataset, "data.grib")
            ERA5_xarrays.append(temp_ERA5_xr.load())

    return ERA5_xarrays

def merge_ERA5_land_datasets(ERA5_datasets, ERA5_merged_filename):

    ERA5_xarrays = read_ERA5_land_datasets(ERA5_datasets)
    merged = xr.concat(ERA5_xarrays, dim='time')

    # --- Clean time dimension ---
    # Step 1: Remove entries with NaT in time
    time_values = merged['time'].values
    valid_time_mask = ~np.isnat(time_values)
    merged = merged.isel(time=valid_time_mask)

    # Step 2: Remove duplicate times
    _, unique_indices = np.unique(merged['time'].values, return_index=True)
    merged = merged.isel(time=unique_indices)

    # Step 3: Sort by time
    merged = merged.sortby('time')

    # Step 4: ERA5-Land tp is accumulated since 00 UTC (01 UTC holds the first hour, 00 UTC the whole
    # previous day), so subtract the previous hour. Hours without the previous hour become NaN
    # https://confluence.ecmwf.int/display/CKB/ERA5-Land%3A+data+documentation#heading-Accumulations
    times = merged['time'].values
    tp_acc = merged['tp'].values
    tp_prev = merged['tp'].reindex(time=times - np.timedelta64(1, 'h')).values
    first_hour = pd.DatetimeIndex(times).hour == 1
    tp_hourly = np.where(first_hour[:, None, None], tp_acc, tp_acc - tp_prev)
    merged['tp'] = (merged['tp'].dims, np.clip(tp_hourly, 0, None))

    if 'expver' in merged.dims: # it contains a mix of ERA5 and ERA5T data 
        # https://confluence.ecmwf.int/pages/viewpage.action?pageId=173385064
        merged = merged.reduce(np.nansum, 'expver')

    ERA5_xarray_save = xr.Dataset(
            {

            "skt": (["time", "y", "x"], merged['skt'].data),
            "swvl1": (["time", "y", "x"], merged['swvl1'].data),
            "tp": (["time", "y", "x"], merged['tp'].data*1000)

            },

            coords={
                    "x": (["x"], merged.longitude.data),
                    "y": (["y"], merged.latitude.data),
                    "time": merged.time.data
            },
    )

    ERA5_xarray_save.skt.attrs["long_name"] = 'Skin temperature'
    ERA5_xarray_save.skt.attrs["units"] = 'K'

    ERA5_xarray_save.swvl1.attrs["long_name"] = 'Volumetric soil water layer 1'
    ERA5_xarray_save.swvl1.attrs["units"] = 'm**3 m**-3'

    ERA5_xarray_save.tp.attrs["long_name"] = 'Hourly total precipitation'
    # checked by ERA5.get_download_dates, merged files of older versions hold daily accumulated tp
    ERA5_xarray_save.tp.attrs["units"] = 'mm/hour'

    ERA5_xarray_save.x.attrs['long_name'] = 'longitude'
    ERA5_xarray_save.x.attrs['units'] = 'degrees'
    ERA5_xarray_save.x.attrs['axis'] = 'X'

    ERA5_xarray_save.y.attrs['long_name'] = 'latitude'
    ERA5_xarray_save.y.attrs['units'] = 'degrees'
    ERA5_xarray_save.y.attrs['axis'] = 'Y'

    if ERA5_merged_filename is not None:
        ERA5_xarray_save.to_netcdf(ERA5_merged_filename)
        
    ERA5_xarray_save = ERA5_xarray_save.rio.write_crs("EPSG:4326")

    return ERA5_xarray_save