#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import numpy as np
from numpy import linalg as la
from scipy import stats
from itertools import combinations
import pyproj
import warnings
from shapely.geometry import Polygon
warnings.filterwarnings("ignore")

def llh2xy(lat_data:np.array,lon_data:np.array)->tuple[np.array, np.array]:
    """Convert lat lon information to Xs,Ys of WGS84 UTM projection

    Args:
        lat_data (np.array): the latitude information
        lon_data (np.array): the longitude information

    Returns:
        tuple[np.array, np.array]: Xs, Ys of the best WGS84 UTM projection
    """
    # Find a UTM projection to be able to calculate meters
    representative_latitude = round(np.mean(lat_data), 10)
    representative_longitude = round(np.mean(lon_data), 10)
    utm_zone = int(np.floor((representative_longitude + 180) / 6) + 1)
    if representative_latitude>0:
        hemisphere='north'
    else:
        hemisphere='south'
    utm_crs_str = '+proj=utm +zone={} +{} +ellps=WGS84 +datum=WGS84 +units=m +no_defs'.format(utm_zone,hemisphere)
    utm_crs_epsg = pyproj.CRS(utm_crs_str).to_epsg()
    
    inProj = pyproj.CRS('EPSG:4326')
    outProj = pyproj.CRS('EPSG:{}'.format(utm_crs_epsg))

    transformer = pyproj.Transformer.from_crs(inProj, outProj)
    Xs,Ys = transformer.transform(lat_data,lon_data)  
    
    return Xs, Ys

def rewrap(data:np.array)-> np.array:
    """Wraps the data between -pi and pi.

    Args:
        data (np.array): the data need to be wrapped

    Returns:
        np.array: Wrapped data. Values are between -pi and pi
    """
    return data - np.round(data/2.0/np.pi)*2.0*np.pi

def isPD(B:np.array)->bool:
    """Checks if input array is positive-definite, via Cholesky

    Args:
        B (np.array): the input array

    Returns:
        bool: True if positive-definite
    """
    try:
        _ = la.cholesky(B)
        return True
    except la.LinAlgError:
        return False
    
def is_row_in_array(row:np.array, arr:np.array)->bool:
    """Checks if a single row (1D array) exists in an matrix (2D array), rowwise.

    Args:
        row (np.array): the 1D array
        arr (np.array): the 2D array 

    Returns:
        bool: True if given 1D array exists in 2D array, rowwise
    """
    return np.any(np.sum(np.abs(arr-row), axis=1) == 0)

def get_DS_pixels(slc_stack:np.array, amp_stack:np.array, lat:np.array, lon:np.array, aoi_geometry:Polygon)->tuple[np.array,np.array,np.array,np.array,np.array]:
    """Calculates the coordinates of SAR SLC pixels that corresponds to given AOI (polygon).

    Args:
        slc_stack (np.array): 3D array (SARs, Ys, Xs) of complex (SLC) values
        amp_stack (np.array): 3D array (SARs, Ys, Xs) of real (amplitude) values
        lat (np.array): 2D array (Ys, Xs) of real (latitude) values
        lon (np.array): 2D array (Ys, Xs) of real (longitude) values
        aoi_geometry (Polygon): Polygon of AOI

    Returns:
        tuple[np.array,np.array,np.array,np.array,np.array]: 1D array of first coordinates (ints) of DS_pixels, 1D array of second coordinates (ints) of DS_pixels, 2D array [SARS, DS_pixels] of selected (complex) SLC values, 2D array [SARS, selected_pixels] of selected (real) amplitude values, 2D array (mask) [Ys, Xs] of selected (bool) SLC pixels
    """
    lon_min, lat_min, lon_max, lat_max = aoi_geometry.bounds
    
    Point1 = [lon_min, lat_min]
    Point2 = [lon_min, lat_max]
    Point3 = [lon_max, lat_max]
    Point4 = [lon_max, lat_min]

    Point1_image_flat_coords = np.argmin(np.abs(lon-Point1[0])+np.abs(lat-Point1[1]))
    Point1_image_xy = np.unravel_index(Point1_image_flat_coords, lon.shape)
    
    Point2_image_flat_coords = np.argmin(np.abs(lon-Point2[0])+np.abs(lat-Point2[1]))
    Point2_image_xy = np.unravel_index(Point2_image_flat_coords, lon.shape)
    
    Point3_image_flat_coords = np.argmin(np.abs(lon-Point3[0])+np.abs(lat-Point3[1]))
    Point3_image_xy = np.unravel_index(Point3_image_flat_coords, lon.shape)
    
    Point4_image_flat_coords = np.argmin(np.abs(lon-Point4[0])+np.abs(lat-Point4[1]))
    Point4_image_xy = np.unravel_index(Point4_image_flat_coords, lon.shape)
    
    border_y1 = np.max([Point3_image_xy[0],Point4_image_xy[0]])
    border_y2 = np.min([Point1_image_xy[0],Point2_image_xy[0]])
    
    if border_y1>border_y2:
        max_y = border_y1
        min_y = border_y2
    else:
        max_y = border_y2
        min_y = border_y1
    
    border_x1 = np.max([Point1_image_xy[1],Point4_image_xy[1]])
    border_x2 = np.min([Point3_image_xy[1],Point2_image_xy[1]])  
    
    if border_x1>border_x2:
        max_x = border_x1
        min_x = border_x2
    else:
        max_x = border_x2
        min_x = border_x1
    
    geometry_mask = np.zeros(lon.shape, dtype=np.bool_)
    geometry_mask[min_y:max_y+1,min_x:max_x+1]= True

    field_pixels=np.nonzero(geometry_mask.astype(np.int64))
    
    DS_coords_1 = field_pixels[0]
    DS_coords_2 = field_pixels[1]
    DS_slc_values = slc_stack[:,DS_coords_1,DS_coords_2]
    DS_amp_values = amp_stack[:,DS_coords_1,DS_coords_2]
    
    return DS_coords_1, DS_coords_2, DS_slc_values, DS_amp_values, geometry_mask

def compute_Covar_Coh_weighted(slc_window_values:np.array, weights:np.array, nbands:int)-> tuple[np.array,np.array]:
    """Compute the covariance matrix of the distributed scatterer

    Args:
        slc_window_values (np.array): an 2D array with complex values (SARs, DS_pixels)
        weights (np.array): a 1D array with float values (0-1) (DS_pixels) that contains weights of DS pixels
        nbands (int): number of SLC SARs that we have

    Returns:
        tuple[np.array,np.array]: Covariance matrix of DS (complex values), coherence matrix of DS (real values)
    """
    # temp matrices for computing covariance matrix for each pixel
    Numer=0+0j
    Amp1=0.0
    Amp2=0.0
    Covar=np.zeros((nbands,nbands)).astype(np.complex64)
    
    for band1 in range(0, nbands):
        # computing only the upper right part of the matrix
        for band2 in range(band1+1, nbands): 
            Numer=np.sum((slc_window_values[band1,:]*np.conjugate(slc_window_values[band2,:]))*weights)
            Amp1=np.sum((np.power(np.abs(slc_window_values[band1,:]),2))*weights)
            Amp2=np.sum((np.power(np.abs(slc_window_values[band2,:]),2)) *weights)  
            res=Numer/np.sqrt(Amp1*Amp2)
            Covar[band1,band2]=res
            Covar[band2,band1]=np.conj(res)
        Covar[band1,band1]=1.0
               
    return Covar, np.abs(Covar) 

def find_best_pixels_amp_fast(DS_coords_1:np.array,
                              DS_coords_2:np.array,
                              DS_amp_values:np.array,
                              amp_stack:np.array,
                              p_value_thres:float = 0.05,
                              az_size:int = 0,
                              rg_size:int = 2)->np.array:
    """Finds a subset of DS_pixels that share the same amplitude temporal behaviour, via Kolmogorov-Smirnov test. Reference amplitude values are considered the ones close to center (at a certain window radius). 

    Args:
        DS_coords_1 (np.array): 1D array of first coordinates (ints) of DS_pixels
        DS_coords_2 (np.array): 1D array of second coordinates (ints) of DS_pixels
        DS_amp_values (np.array): 2D array [SARS, selected_pixels] of selected (real) amplitude values
        amp_stack (np.array): 3D array (SARs, Ys, Xs) of real (amplitude) values
        p_value_thres (float, optional): p-value threshold. Defaults to 0.05.
        win_size (int, optional): radius of window. Window is used to get the reference amplitudes. Defaults to 1.

    Returns:
        np.array: A boolean 1d array [DS_pixels,] that contains pixels with similar amplitude information
    """
    num_pixels = DS_coords_1.shape[0]
    DS_amp_similarity = np.zeros((num_pixels), dtype=np.bool_)
    
    # compare with mean intensity value of all region
    # mean_DS_amp = np.nanmean(DS_amp_values, axis=1)
    mean_DS_amp = np.nanmedian(DS_amp_values, axis=1)
    
    # compare with mean intensity value of center region
    center_coord1 = int(np.mean(DS_coords_1))
    center_coord2 = int(np.mean(DS_coords_2))
    center_amp_values = amp_stack[:,
                                  center_coord1-az_size:center_coord1+az_size+1, # azimuth dimension ~ 20 meters
                                  center_coord2-rg_size:center_coord2+rg_size+1] # range dimension ~ 4 meters
    center_amp_values_flatten = np.nanmean(np.nanmean(center_amp_values, axis=1), axis=1)
    mean_DS_amp = center_amp_values_flatten.copy()
    
    for DS_child_id in range(num_pixels):
        DS_child = DS_amp_values[:,DS_child_id]
        KS_statistic, p_value = stats.ks_2samp(mean_DS_amp,  DS_child)
        if p_value<p_value_thres:
            DS_amp_similarity[DS_child_id]=False
        else:
            DS_amp_similarity[DS_child_id]=True
            
    return DS_amp_similarity

def find_best_pixels_amp(DS_coords_1:np.array, DS_coords_2:np.array, DS_amp_values:np.array, p_value_thres:float = 0.05, similarity_perc:float = 0.4)->np.array:
    """Finds a subset of DS_pixels that share the same amplitude temporal behaviour, via Kolmogorov-Smirnov test. All possible pixels combinations are calculated. The pixels that found similar at a defined percentage were keeped.

    Args:
        DS_coords_1 (np.array): 1D array of first coordinates (ints) of DS_pixels
        DS_coords_2 (np.array): 1D array of second coordinates (ints) of DS_pixels
        DS_amp_values (np.array): 2D array [SARS, selected_pixels] of selected (real) amplitude values
        p_value_thres (float, optional): p-value threshold. Defaults to 0.05.
        similarity_perc (float, optional): the percentage of similarities among all pixels. Defaults to 0.4.

    Returns:
        np.array: A boolean 1d array [DS_pixels,] that contains pixels with similar amplitude information
    """
    assert DS_coords_1.shape[0] == DS_coords_2.shape[0]
    num_pixels = DS_coords_1.shape[0]
    DS_amp_similarity = np.zeros((num_pixels, num_pixels), dtype=np.bool_)
    
    for DS_parent_id in range(num_pixels):
        DS_parent = DS_amp_values[:,DS_parent_id]
 
        for DS_child_id in range(num_pixels):
            if DS_child_id == DS_parent_id:
                DS_amp_similarity[DS_parent_id, DS_child_id] = True
            else:
                DS_child = DS_amp_values[:,DS_child_id]
                KS_statistic, p_value = stats.ks_2samp(DS_parent,  DS_child)
                
                if p_value<p_value_thres:
                    DS_amp_similarity[DS_parent_id,DS_child_id]=False
                else:
                    DS_amp_similarity[DS_parent_id,DS_child_id]=True
                
    DS_amp_sim_percents =  np.mean(DS_amp_similarity, axis=0)
    Similar_pixels_amp_mask = DS_amp_sim_percents>similarity_perc
    print ("{} pixels from {} were found similar based on amplitude information".format(np.sum(Similar_pixels_amp_mask), num_pixels))
           
    return Similar_pixels_amp_mask

def compare_coherence_matrices(Coh1:np.array, pixel_comb_mask1:np.array, Coh2:np.array, pixel_comb_mask2:np.array)->tuple[np.array,np.array]:
    """Compares two coherence matrices and returns the one with largest frobenius norm.

    Args:
        Coh1 (np.array): 2D array with (real) coherence values 
        pixel_comb_mask1 (np.array): 1D array of (int) SLC pixel indices that form the first coherence matrix
        Coh2 (np.array): 2D array with (real) coherence values 
        pixel_comb_mask2 (np.array): 1D array of (int) SLC pixel indices that form the second coherence matrix

    Returns:
        tuple[np.array,np.array]: Selected coherence matrix, corresponding SLC pixel indices
    """
    Coh1_norm = la.norm(Coh1)
    Coh2_norm = la.norm(Coh2)
    if Coh1_norm>Coh2_norm:
        matrix = Coh1
        pixel_comb_mask = pixel_comb_mask1
    else:
        matrix = Coh2
        pixel_comb_mask = pixel_comb_mask2
        
    return matrix, pixel_comb_mask

def find_best_pixels_ph (DS_coords_1:np.array,
                         DS_coords_2:np.array,
                         DS_slc_values:np.array,
                         Best_pixels_amp:np.array,
                         keep_percent:float = 0.7)->np.array:
    """Calculates the subset of pixels (at a given percentage) that yields the highest overall coherence. 

    Args:
        DS_coords_1 (np.array): 1D array of first coordinates (ints) of DS_pixels
        DS_coords_2 (np.array): 1D array of second coordinates (ints) of DS_pixels
        DS_slc_values (np.array): 2D array [SARS, DS_pixels] of selected (complex) SLC values
        Best_pixels_amp (np.array): a boolean 1d array [DS_pixels,] that contains pixels with similar amplitude information
        keep_percent (float, optional): the percentage of pixels that we respect. Defaults to 0.7.

    Returns:
        np.array: A boolean 1d array [DS_pixels,] that contains pixels with similar amplitude information
    """
    all_pixels = DS_coords_1.shape[0]
    
    if Best_pixels_amp is None:
        initial_pixels = all_pixels
    else:
        DS_coords_1 = DS_coords_1[Best_pixels_amp]
        DS_coords_2 = DS_coords_2[Best_pixels_amp]
        DS_slc_values = DS_slc_values[:,Best_pixels_amp]
        initial_pixels = DS_coords_1.shape[0]

    min_num_pixels = int(initial_pixels*keep_percent)
    
    # First calculate the Coherence matrix of all pixels
    
    DS_Covar0, Best_Coh = compute_Covar_Coh_weighted(slc_window_values = DS_slc_values,
                                                    weights = np.ones(DS_slc_values.shape[1], dtype=np.float32),
                                                    nbands = DS_slc_values.shape[0])
    Best_pixel_comb = np.arange(initial_pixels)
    
    for num_pixels in range(initial_pixels, min_num_pixels, -1):
        pixels_combs_list = list(combinations(Best_pixel_comb, num_pixels-1))
        # mean_coh_matrix = np.zeros(len(pixels_combs_list))

        for comb_ind, pixel_comb in enumerate(pixels_combs_list):
            
            pixel_comb_temp = list(pixel_comb)
            
            DS_slc_values_temp = DS_slc_values[:,pixel_comb_temp]
            
            DS_Covar_temp, DS_Coh_temp = compute_Covar_Coh_weighted(slc_window_values = DS_slc_values_temp,
                                                          weights = np.ones(DS_slc_values_temp.shape[1], dtype=np.float32),
                                                          nbands = DS_slc_values_temp.shape[0])

            Best_Coh,  Best_pixel_comb = compare_coherence_matrices(Best_Coh, Best_pixel_comb, DS_Coh_temp, pixel_comb_temp)
    
    Similar_pixels_ph_mask = np.zeros(initial_pixels, dtype=np.bool_)
    Similar_pixels_ph_mask[Best_pixel_comb] = True
    
    return Similar_pixels_ph_mask
