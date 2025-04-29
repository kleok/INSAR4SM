# <img src="https://github.com/kleok/INSAR4SM/blob/main/figures/insar4sm_logo.png" width="58"> InSAR4SM - Interferometric Synthetic Aperture Radar for Soil Moisture

## Introduction

InSAR4SM is a free and open-source software for estimating soil moisture using interferometric observables over arid regions. 
It requires as inputs the following data:

- a Topstack [ISCE](https://github.com/isce-framework/isce2) SLC stack
- a meteorological dataset (e.g. from [ERA5-Land](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-land?tab=overview) data)
- a soil texture dataset (sand and clay) for your region of interest (e.g. from [soilgrids](https://soilgrids.org/))

For each Sentinel-1 acqusitions a surface (top 5 cm) soil moisture map is provided.

This is research code provided to you "as is" with NO WARRANTIES OF CORRECTNESS. Use at your own risk.

<img src="https://github.com/kleok/INSAR4SM/blob/main/figures/InSAR4SM_NA.png" width="900">

## 1. Installation
The installation notes below are tested only on Linux.

### 1.1 Download InSAR4SM
First you have to download InSAR4SM using the following command

```git clone https://github.com/kleok/InSAR4SM.git```

### 1.2 Create python environment for InSAR4SM

InSAR4SM is written in Python3 and relies on several Python modules. You can install them by using ```INSAR4SM_env.yml``` file.

```conda env create -f INSAR4SM_env.yml```

### 1.3 Set environmental variables (optional)
on GNU/Linux, append to .bashrc file:
```
export InSAR4SM_HOME=~/InSAR4SM
export PYTHONPATH=${PYTHONPATH}:${InSAR4SM_HOME}
export PATH=${PATH}:${InSAR4SM_HOME}
```

## 2. Running InSAR4SM

InSAR4SM provide soil moisture estimations using interferometric observables, meteorological and soil texture data from the following pipeline. 
- Identification of driest SAR image based on meteorological information.
- Calculation of interferometric observables (coherence and phase closure).
- Identification of SAR acquisitions related to dry soil moisture conditions using coherence and amplitude information.
- Calculation of coherence information due to soil moisture variations.
- Soil moisture inversion using De Zan`s model.

Please start with the jupyter notebook example [here](https://github.com/kleok/INSAR4SM/blob/main/INSAR4SM_app.ipynb)

## 3. Documentation and citation
Algorithms implemented in the software are described in detail at our publication. If InSAR4SM was useful for you, we encourage you to cite the following work.

- Karamvasis, K., & Karathanassi, V. (2023). Soil moisture estimation from Sentinel-1 interferometric observations over arid regions. Computers & Geosciences, 178, 105410. [here](https://doi.org/10.1016/j.cageo.2023.105410)

## 4. Contact us
Feel free to open an issue, comment or pull request. We would like to listen to your thoughts and your recommendations. Any help is very welcome!
