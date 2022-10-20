Quickstart
==========

Use `InSAR4SM_app.py <https://github.com/kleok/INSAR4SM/blob/main/InSAR4SM_app.py>`_

InSAR4SM provide soil moisture estimations using interferometric observables and meteorological data using a 5-step framework. 
- Identification of driest SAR image based on meteorological information.
- Calculation of interferometric observables (coherence and phase closure).
- Identification of SAR acquisitions related to dry soil moisture conditions using coherence and amplitude information.
- Calculation of coherence information due to soil moisture variations.
- Soil moisture inversion using De Zan`s model.

In order to run InSAR4SM please make sure to update/provide the following information located at "Input arguments" cell at `InSAR4SM_app.py <https://github.com/kleok/INSAR4SM/blob/main/InSAR4SM_app.py>`_


.. code-block:: bash

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












