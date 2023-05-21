#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import numpy as np 
import os
import glob
from osgeo import gdal
import geopandas as gpd
import shapely

def latlon_2_SAR_coord(lat_data:np.array, lon_data:np.array, lat_p:float, lon_p:float, dist:float = 0.001)->tuple[int,int]:
    """finds the SAR coordinates given a lat,lon pair

    Args:
        lat_data (np.array): the latitude information of Topstack dataset
        lon_data (np.array): the longitude information of Topstack dataset
        lat_p (float): the latitude value of a single point
        lon_p (float): the longitude value of a single point
        dist (float, optional): spatial distance in degrees. Defaults to 0.001.

    Returns:
        tuple[int,int]: x_sar, y_sar the coordinates related to Topstack dataset
    """
    lat_mask = np.abs(lat_data-lat_p) < dist
    lon_mask = np.abs(lon_data-lon_p) < dist

    mask = lat_mask*lon_mask
    
    n_iter = 1
    
    while mask.nonzero()[0].shape[0] == 0 and  n_iter<10:
        print('It seems that you AOI is outside of Topstack region')
        print('We will modify your AOI...')
        dist += 0.001
        lat_mask = np.abs(lat_data-lat_p) < dist
        lon_mask = np.abs(lon_data-lon_p) < dist
        mask = lat_mask*lon_mask

    y_sar = int(np.mean(mask.nonzero()[0]))
    x_sar = int(np.mean(mask.nonzero()[1]))

    return x_sar, y_sar


def get_SAR_borders(merged_dir:str, AOI_WGS84:str)->tuple[int,int,int,int]:
    """Calculates the SAR Topstack coordinates given a WGS AOI polygon.

    Args:
        merged_dir (str): path of Topstack data (merged folder)
        AOI_WGS84 (str): path of vector AOI file

    Returns:
        tuple[int,int,int,int]: ymin, ymax, xmin, xmax coordinates related with Topstack datasets
    """
    latFile = os.path.join(merged_dir, "geom_master", "lat.rdr.full.vrt")
    lonFile = os.path.join(merged_dir, "geom_master", "lon.rdr.full.vrt")
    
    # latFile = os.path.join(merged_dir, "geom_reference", "lat.rdr.full.vrt")
    # lonFile = os.path.join(merged_dir, "geom_reference", "lon.rdr.full.vrt")
    
    lat_data = gdal.Open(latFile).ReadAsArray()
    lon_data = gdal.Open(lonFile).ReadAsArray()
    
    polygon_WGS84 = gpd.read_file(AOI_WGS84).geometry.iloc[0]
    
    if type(polygon_WGS84) == shapely.geometry.multipolygon.MultiPolygon:
        polygon_WGS84 = list(polygon_WGS84)[0]
  
    assert type(polygon_WGS84) == shapely.geometry.polygon.Polygon
    
    lons, lats = polygon_WGS84.exterior.coords.xy
    
    # we drop the last element because it is the same with the first one.
    lons = np.array(lons)[:-1]
    lats = np.array(lats)[:-1]
    assert lons.shape == lats.shape
    
    SAR_Xs = np.zeros_like(lons).astype(np.int32)
    SAR_Ys = np.zeros_like(lons).astype(np.int32)
    
    for ind_p in range(lons.shape[0]):
        lat_p = lats[ind_p]
        lon_p = lons[ind_p]
        SAR_Xs[ind_p], SAR_Ys[ind_p] = latlon_2_SAR_coord(lat_data, lon_data, lat_p, lon_p)
        
    # Find borders based on given AOI
    xmin = np.min(SAR_Xs)
    xmax = np.max(SAR_Xs)
    ymin = np.min(SAR_Ys)
    ymax = np.max(SAR_Ys)
    
    return ymin, ymax, xmin, xmax

def write_to_numpy(geometry_folder:str)->None:
    """Converts geometrical datasets (lat, lon, hgt, shadowMask, los, incLocal, baselines) from vrt to npy format.

    Args:
        geometry_folder (str): path of vrt files for geometric information
    """
    datasets=['shadowMask.vrt',
              'hgt.vrt',
              'incLocal.vrt',
              'lat.vrt',
              'lon.vrt',
              'los.vrt']
    
    for dataset in datasets:
        data=gdal.Open(os.path.join(geometry_folder,dataset)).ReadAsArray()
        
        with open(os.path.join(geometry_folder,dataset.split('.')[0]+'.npy'), 'wb') as f:
            np.save(f, data)
    
    # baseline files
    baseline_grid_files=glob.glob(geometry_folder+'/perp_baseline*.vrt')
    
    baseline_grid_files_sorted=sorted(baseline_grid_files)
    # earlier image is selected as master!!!
    master_baseline_grid_file = baseline_grid_files_sorted[0]
    
    master_baseline_grid_data=gdal.Open(master_baseline_grid_file).ReadAsArray()
    
    baseline_stack_numpy=np.zeros((len(baseline_grid_files_sorted),
                                   master_baseline_grid_data.shape[0],
                                   master_baseline_grid_data.shape[1]))
    
    for SAR_index, baseline_grid in enumerate(baseline_grid_files_sorted):
        slave_baseline_grid=gdal.Open(baseline_grid).ReadAsArray()
        
        if baseline_grid == master_baseline_grid_file:
            baseline_stack_numpy[SAR_index,:,:]= slave_baseline_grid-master_baseline_grid_data
        else:
            baseline_stack_numpy[SAR_index,:,:]= slave_baseline_grid-master_baseline_grid_data

        
    with open(os.path.join(geometry_folder,'perp_baseline_stack.npy'), 'wb') as f:
        np.save(f, baseline_stack_numpy)
        
        
def tops_2_vrt(indir:str, outdir:str, stackdir:str, AOI_WGS84:str, geomdir:str)->None:
    """Creates vrt files for subsetting Topstack data (slc data, lat, lon, hgt, shadowMask, los, incLocal, baselines). Saves to disk geometric information (npy format).

    Args:
        indir (str): Topstack path
        outdir (str): directory that vrt files and npy files will be saved
        stackdir (str): path for slc information of Topstack data
        AOI_WGS84 (str): path of vector AOI file
        geomdir (str): path for geometrical information of Topstack data
    """
    ###Get ann list and slc list
    slclist = glob.glob(os.path.join(indir,'SLC','*','*.slc.full'))
    num_slc = len(slclist)

    print('number of SLCs discovered: ', num_slc)
    #print('we assume that the SLCs and the vrt files are sorted in the same order')
    
    slclist.sort()


    ###Read the first ann file to get some basic things like dimensions
    ###Just walk through each of them and create a separate VRT first
    if not os.path.exists(outdir):
        print('creating directory: {0}'.format(outdir))
        os.makedirs(outdir)
    else:
        print('directory "{0}" already exists.'.format(outdir))

    data = []
    dates = []

    width = None
    height = None

    print('write vrt file for each SLC ...')
    for ind, slc in enumerate(slclist):

        ###Parse the vrt file information.
        metadata = {}
        width = None
        height = None
        path = None

        ds = gdal.Open(slc , gdal.GA_ReadOnly)
        width = ds.RasterXSize
        height = ds.RasterYSize
        ds = None

        metadata['WAVELENGTH'] = 0.05546576 
        metadata['ACQUISITION_TIME'] = os.path.basename(os.path.dirname(slc))
        
        path = os.path.abspath(slc)

        tag = metadata['ACQUISITION_TIME'] 

        vrttmpl='''<VRTDataset rasterXSize="{width}" rasterYSize="{height}">
    <VRTRasterBand dataType="CFloat32" band="1" subClass="VRTRawRasterBand">
        <sourceFilename>{PATH}</sourceFilename>
        <ImageOffset>0</ImageOffset>
        <PixelOffset>8</PixelOffset>
        <LineOffset>{linewidth}</LineOffset>
        <ByteOrder>LSB</ByteOrder>
    </VRTRasterBand>
</VRTDataset>'''
        

#        outname =  datetime.datetime.strptime(tag.upper(), '%d-%b-%Y %H:%M:%S UTC').strftime('%Y%m%d')

        outname = metadata['ACQUISITION_TIME']
        out_file = os.path.join(outdir, '{0}.vrt'.format(outname))
        with open(out_file, 'w') as fid:
            fid.write( vrttmpl.format(width=width,
                                     height=height,
                                     PATH=path,
                                     linewidth=8*width))

        data.append(metadata)
        dates.append(outname)


    ####Set up single stack file
    if os.path.exists( stackdir):
        print('stack directory: {0} already exists'.format(stackdir))
    else:
        print('creating stack directory: {0}'.format(stackdir))
        os.makedirs(stackdir)    


    # setting up a subset of the stack
    ymin, ymax, xmin, xmax = get_SAR_borders(merged_dir = indir,
                                             AOI_WGS84 = AOI_WGS84)
    
    xsize = xmax - xmin
    ysize = ymax - ymin

    slcs_base_file = os.path.join(stackdir, 'slcs_base.vrt')
    print('write vrt file for stack directory')
    with open(slcs_base_file, 'w') as fid:
        fid.write( '<VRTDataset rasterXSize="{xsize}" rasterYSize="{ysize}">\n'.format(xsize=xsize, ysize=ysize))

        for ind, (date, meta) in enumerate( zip(dates, data)):
            outstr = '''    <VRTRasterBand dataType="CFloat32" band="{index}">
        <SimpleSource>
            <SourceFilename>{path}</SourceFilename>
            <SourceBand>1</SourceBand>
            <SourceProperties RasterXSize="{width}" RasterYSize="{height}" DataType="CFloat32"/>
            <SrcRect xOff="{xmin}" yOff="{ymin}" xSize="{xsize}" ySize="{ysize}"/>
            <DstRect xOff="0" yOff="0" xSize="{xsize}" ySize="{ysize}"/> 
        </SimpleSource>
        <Metadata domain="slc">
            <MDI key="Date">{date}</MDI>
            <MDI key="Wavelength">{wvl}</MDI>
            <MDI key="AcquisitionTime">{acq}</MDI>
        </Metadata>
    </VRTRasterBand>\n'''.format(width=width, height=height,
                                xmin=xmin, ymin=ymin,
                                xsize=xsize, ysize=ysize,
                                date=date, acq=meta['ACQUISITION_TIME'],
                                wvl = meta['WAVELENGTH'], index=ind+1, 
                                path = os.path.abspath( os.path.join(outdir, date+'.vrt')))
            fid.write(outstr)

        fid.write('</VRTDataset>')

    ####Set up latitude, longitude and height files
    
    if os.path.exists(geomdir):
        print('directory {0} already exists.'.format(geomdir))
    else:
        print('creating geometry directory: {0}'.format(geomdir))
        os.makedirs( geomdir)


    vrttmpl='''<VRTDataset rasterXSize="{xsize}" rasterYSize="{ysize}">
    <VRTRasterBand dataType="Float64" band="1">
      <SimpleSource>
        <SourceFilename>{PATH}</SourceFilename>
        <SourceBand>1</SourceBand>
        <SourceProperties RasterXSize="{width}" RasterYSize="{height}" DataType="Float64"/>
        <SrcRect xOff="{xmin}" yOff="{ymin}" xSize="{xsize}" ySize="{ysize}"/>
        <DstRect xOff="0" yOff="0" xSize="{xsize}" ySize="{ysize}"/>
      </SimpleSource>
    </VRTRasterBand>
</VRTDataset>'''

    print('write vrt file for geometry dataset')
    layers = ['lat', 'lon', 'hgt','shadowMask', 'los', 'incLocal']
    for ind, val in enumerate(layers):
        with open( os.path.join(geomdir, val+'.vrt'), 'w') as fid:
            fid.write( vrttmpl.format( xsize = xsize, ysize = ysize,
                                       xmin = xmin, ymin = ymin,
                                       width = width,
                                       height = height,
                                       PATH = os.path.abspath( os.path.join(indir, 'geom_master', val+'.rdr.full.vrt')),
                                    #    PATH = os.path.abspath( os.path.join(indir, 'geom_reference', val+'.rdr.full.vrt')),
                                       linewidth = width * 8))

    baseline_grids = glob.glob(os.path.join(indir,"baselines","2*","2*[0-9].full.vrt"))
    baseline_grids_sorted = sorted(baseline_grids)
    
    for baseline_grid in baseline_grids_sorted:
        baseline_grid_outputname= 'perp_baseline_'+os.path.basename(baseline_grid).split('.')[0]
        with open( os.path.join(geomdir, baseline_grid_outputname+'.vrt'), 'w') as fid:
            fid.write( vrttmpl.format( xsize = xsize, ysize = ysize,
                                       xmin = xmin, ymin = ymin,
                                       width = width,
                                       height = height,
                                       PATH = os.path.abspath(baseline_grid),
                                       linewidth = width * 8))
       
    write_to_numpy(geomdir)
            

    
    
    
                    
               
                  
               












