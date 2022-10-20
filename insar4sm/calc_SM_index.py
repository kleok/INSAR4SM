#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from collections import Counter
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

def calc_coh_model(SM:np.array, coh_row_inds:np.array, coh_col_inds:np.array)->np.array:
    """Functionality that calculates the soil moisture index based on Eq. 4 of Burgi et al., 2021
        
        .. math:: log(\gamma_{ij}^{sm})= -abs(log(sm_i) - log(sm_j))
        
    Args:
        SM (np.array): Soil moisture
        coh_row_inds (np.array): Index related with rows of coherence matrix
        coh_col_inds (np.array): Index related with cols of coherence matrix

    Returns:
        np.array: Modelled coherence values

    ..References:
        Bürgi, P. M., & Lohman, R. B. (2021). High-resolution soil moisture evolution in hyper-arid regions: A comparison of InSAR, SAR, microwave, optical, and data assimilation systems in the southern Arabian Peninsula. Journal of Geophysical Research: Earth Surface, 126, e2021JF006158. https://doi.org/10.1029/2021JF006158 
    
    """
    
    n_cohs = coh_row_inds.shape[0]
    coh_model = np.zeros(n_cohs)
    
    for n_coh in range(n_cohs):
      coh_model[n_coh] = np.exp(-np.abs(np.log(SM[coh_row_inds[n_coh]])-np.log(SM[coh_col_inds[n_coh]])))
      
    return coh_model


def calc_burgi_sm_index(coh_sm:np.array, SM_sorting:np.array, n_dry_bands:int, sm_dry:float, ind_DS:int, band_start:int, band_end:int, temp_thres:int = 0)->tuple[np.array, np.array, np.array, np.array]:
    """Calculates the surface soil moisture (SSM) index based on interferometric coherence.
    
    Args:
        coh_sm (np.array): The coherence related to SSM variations for all SAR acquisitions
        SM_sorting (np.array): The ordering (ascending) of SAR acquisitions based on their SSM levels
        n_dry_bands (int): The number of SAR acquisitions that will be considered dry (30% of all SAR acquisitions)
        sm_dry (float): The SSM value that corresponds to driest conditions (default value 0.03 m3/m3)
        ind_DS (int): Number of distributed scatterer that will be analysed
        band_start (int): The index of first SAR acquisition to be analyzed
        band_end (int): The index of last SAR acquisition to be analyzed
        temp_thres (int, optional): The temporal distance (distance between SARs) that will be considered. If 1 means only the sequential SARs will be used. Defaults to 0.

    Returns:
        tuple[np.array, np.array, np.array, np.array]: proxy SSM index using all coherence information, proxy SSM index using coherence information over dry SARs, indices of SAR acquisitions considered dry, coherence related to SSM variations for selected SAR acquisitions
    """
    # initial values based on coherence index
    coh = coh_sm[ind_DS,band_start:band_end,band_start:band_end].copy()
    
    # we assign nan values over diagonal elements
    coh[np.diag_indices_from(coh)] = np.nan
    
    # creating nan mask
    coh_nan_mask = ~np.isnan(coh)

    if temp_thres==0:
        # create a coherence mask with valid elements
        coh_mask = coh_nan_mask 

    else:
        # find the coherence elements that have short temporal distance 
        bad_inds = np.triu_indices_from(coh, k=int(temp_thres))
        all_inds = np.triu_indices_from(coh, k=1)
        
        bad_ind_df = pd.DataFrame(bad_inds).T
        all_ind_df = pd.DataFrame(all_inds).T
        
        merged_df = pd.concat([bad_ind_df, all_ind_df])
        
        good_inds = merged_df.drop_duplicates(keep=False).values
        good_rows_inds = good_inds[:,0]
        good_cols_inds = good_inds[:,1]
        
        coh_dist_mask = np.zeros_like(coh).astype(np.bool_)
        coh_dist_mask[good_rows_inds, good_cols_inds] = True
        
        # create a coherence mask with valid elements
        coh_mask = coh_nan_mask * coh_dist_mask

    # get indices of independent coherence observations that are non nan
    coh_inds = np.triu_indices_from(coh_mask,1)
    coh_row_inds = coh_inds[0]
    coh_col_inds = coh_inds[1]
    
    coh_valid_values = coh_mask[coh_row_inds, coh_col_inds].copy()
    
    coh_row_inds = coh_inds[0][coh_valid_values]
    coh_col_inds = coh_inds[1][coh_valid_values]
    
    # In order to be sure about the selected dry bands we construct the 
    # cohence matrix of dry bands x dry bands. We expect to see really high 
    # values because of the small sm values of the dry bands.
    dry_bands_inds = SM_sorting[:n_dry_bands].copy()
    
    # select only the independent and non-nan values
    coh_obs = coh[coh_row_inds, coh_col_inds].copy()
    
    # clip the part of coherence matrix that correspond to dry conditions.
    dry_coh_matrix = coh[dry_bands_inds,:][:,dry_bands_inds].copy()
    
    # We drop the bands that have lower that mean coherence values in respect 
    # with other dry bands
    bad_dry_bands = (dry_coh_matrix<np.nanquantile(dry_coh_matrix.flatten(), 0.3)).nonzero()[0]
    drop_dry_bands = [item for item, count in Counter(bad_dry_bands).items() if count > 1]
    dry_bands_inds=np.delete(dry_bands_inds,drop_dry_bands)
    
    # replace large and nan values with mean dry coherence value
    # keep diagonal values nan
    dry_coh_matrix = coh[dry_bands_inds,:][:,dry_bands_inds].copy()

    coh_sm_filled = coh.copy()

    # we calculate a coherence vector using only coherence over (rows) bands 
    # that are dry. The resulting coherence vector represents the coherence
    # in respect to dry conditions. Values close to one represents dry 
    # conditions. Values close to zeros represents wet conditions.
    coh_vector = np.nanmean(coh_sm_filled[dry_bands_inds,:], axis=0)
    
    # in case we have all nan coherence values for some dry_bands columns, we 
    # fill in values over the dry bands by doing the following:
    # we find bands that have high coherence with the band with the missing 
    # values. 
    if np.any(np.isnan(coh_vector)):
        nan_bands = np.arange(coh.shape[0])[np.isnan(coh_vector)]
        for nan_band in nan_bands:
            coh_nan_vector = coh[nan_band,:]
            representative_bands = (coh_nan_vector>0.8).nonzero()
            fill_in_coh_value =  np.nanmean(coh_vector[representative_bands])
            coh_vector[nan_band] = fill_in_coh_value

    #-------------------------------------------------------------------------
    # # We calculate the mean coherence value of dry conditions.
    # dry_coh_value = np.mean(coh_vector[dry_bands_inds])
    # # We replace all the values above dry_coh_value with dry_coh_value.
    # coh_sm_clipped = np.clip(coh_vector, 0, dry_coh_value)
    # coh_sm_index = sm_dry/(100*coh_sm_clipped)
    #-------------------------------------------------------------------------
    coh_sm_index = sm_dry/(100*coh_vector)
    SM0 = coh_sm_index*100
    
    # bounds of inverted soil moistures
    bounds_SM0 = (0,100)
    bounds = []
    for element in range(coh.shape[0]):
        bounds.append(bounds_SM0)
      
    def objective_function(SM0):  
      coh_model = calc_coh_model(SM0, coh_row_inds, coh_col_inds)
    
      # coherence component cost
      Coh_cost = np.linalg.norm(coh_model-coh_obs)
      
      return Coh_cost
        
    results = minimize(objective_function,
                        SM0,
                        method='L-BFGS-B',
                        bounds = bounds)

    SM_index = results['x']
    
    return SM_index, SM0, dry_bands_inds, coh_sm_filled