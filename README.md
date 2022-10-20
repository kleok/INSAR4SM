# <img src="https://github.com/kleok/INSAR4SM/blob/main/figures/insar4sm_logo.png" width="58"> InSAR4SM - Interferometric Synthetic Aperture Radar for Soil Moisture

## Introduction

InSAR4SM is a free and open-source software for estimating soil moisture using interferometric observables. It requires as inputs a) a Topstack ISCE SLC stack and b) a meteorological dataset (e.g. ERA5-Land data). The main output result is a point vector file that contains soil moisture information over time.

This is research code provided to you "as is" with NO WARRANTIES OF CORRECTNESS. Use at your own risk.

<img src="https://github.com/kleok/INSAR4SM/blob/main/figures/InSAR4SM_NA.png" width="900">

## 1. Installation
The installation notes below are tested only on Linux.

### 1.1 Download InSAR4SM
First you have to download InSAR4SM using the following command

```git clone https://github.com/kleok/InSAR4SM.git```

### 1.2 Create python environment for InSAR4SM

InSAR4SM is written in Python3 and relies on several Python modules. You can install them by using ```INSAR4SM_env.yml``` file.

### 1.3 Set environmental variables
on GNU/Linux, append to .bashrc file:
```
export InSAR4SM_HOME=~/InSAR4SM
export PYTHONPATH=${PYTHONPATH}:${InSAR4SM_HOME}
export PATH=${PATH}:${InSAR4SM_HOME}
```

## 2. Running InSAR4SM
[InSAR4SM_app.py](https://github.com/kleok/INSAR4SM/blob/main/InSAR4SM_app.py)

InSAR4SM provide soil moisture estimations using interferometric observables and meteorological data using a 5-step framework. 
- Identification of driest SAR image based on meteorological information.
- Calculation of interferometric observables (coherence and phase closure).
- Identification of SAR acquisitions related to dry soil moisture conditions using coherence and amplitude information.
- Calculation of coherence information due to soil moisture variations.
- Soil moisture inversion using De Zan`s model.

In order to run InSAR4SM please make sure to update/provide the following information located at "Input arguments" cell at [InSAR4SM_app.py](https://github.com/kleok/INSAR4SM/blob/main/InSAR4SM_app.py)

```
# the name of your project
projectname = 'INSAR4SM_estimations_test'

# the directory of the topstack processing stack
topstackDir = '/RSL02/SM_Arabia/Topstack_processing'

# time of Sentinel-1 pass.
orbit_time = '15:00:00'

# the AOI geojson file, ensure that AOI is inside your topstack stack
AOI = '/RSL02/SM_Arabia/aoi/aoi_test.geojson'

# spatial resolution of soil moisture grid in meters
grid_size = 250

# You can set manually a dry date (one of your SAR acquisition dates ) or set to None
dry_date = '20180401' 
# set to True in case you provide manually an dry_date
dry_date_manual_flag = True

# the meteorological file. You can either provide an ERA5-land file or a csv file with 3 columns (Datetimes, tp__m, skt__K).
meteo_file = '/RSL02/SM_Arabia/era5/adaptor.mars.internal-1665654570.8663068-23624-3-8bce5925-a7e7-4993-a701-0e05b4e9dabd.nc'
# set to True in case you provide an ERA5-Land file
ERA5_flag = True
# In case you downloaded surface soil moisture from ERA5-land, set to True for comparison purposes
ERA5_sm_flag = True

# soil information datasets (https://soilgrids.org/)
sand_soilgrids = '/RSL02/SM_Arabia/soilgrids/clay.tif'
clay_soilgrids = '/RSL02/SM_Arabia/soilgrids/sand.tif'

# the output directory 
export_dir = '/RSL02/SM_Arabia/{}'.format(projectname)
```
## 3. Documentation and citation
Algorithms implemented in the software are described in detail at our publication. If InSAR4SM was useful for you, we encourage you to cite the following work.

- Karamvasis K, Karathanassi V. Soil moisture estimation from Sentinel-1 interferometric observations over arid regions.(under review). Preprint available [here](https://arxiv.org/abs/2210.10665)

## 4. Contact us
Feel free to open an issue, comment or pull request. We would like to listen to your thoughts and your recommendations. Any help is very welcome!
