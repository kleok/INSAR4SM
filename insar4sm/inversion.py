#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import datetime
from scipy.optimize import minimize
np.seterr(divide='ignore', invalid='ignore')

from insar4sm.forward_modelling_funcs import Covar_modelled_calc
from insar4sm.forward_modelling_funcs import phase_closure_modelled_calc
from insar4sm.phase_closure_funcs import Covar_2_phase_closures

def rewrap(data:np.array)-> np.array:
    """Wraps the data between -pi and pi.

    Args:
        data (np.array): the data need to be wrapped

    Returns:
        np.array: Wrapped data. Values are between -pi and pi
    """
    return data - np.round(data/2.0/np.pi)*2.0*np.pi

def inversion(SM0:np.array,
              DS_mean_inc_angle:float,
              opt_method:str,
              Phase_closures:np.array,
              Mask_ph_closure:np.array,
              SM_coh:np.array,
              dry_index:int,
              dry_bands_inds:np.array,
              sm_dry:float,
              freq_GHz:float,
              clay_pct:float,
              sand_pct:float)->np.array:
    """Soil moisture invertion functionality based on the De Zan`s model.

    Args:
        SM0 (np.array): Initial soil moisture values
        DS_mean_inc_angle (float): Incidence angle in degrees.
        opt_method (str): Minimization method
        Phase_closures (np.array): Phase closures
        Mask_ph_closure (np.array): boolean mask 
        SM_coh (np.array): coherence due to soil moisture variations
        dry_index (int): index of driest SAR acquisition
        dry_bands_inds (np.array): Indices of SAR acquisitions considered dry
        sm_dry (float): Driest soil moisture value
        freq_GHz (float): Operation frequency of SAR sensor in GHz.
        clay_pct (float): Clay mass fraction (0-100)%.
        sand_pct (float): Sand mass fraction (0-100)%.

    Returns:
        np.array: Soil moisture inverted values

    ..References:
        De Zan, F., Parizzi, A., Prats-Iraola, P., Lopez-Dekker, P., 2014. A SAR Interferometric Model for Soil Moisture. IEEE Trans. Geosci. Remote Sens. 52, 418-425. https://doi.org/10.1109/TGRS.2013.2241069   
    """
    
    # calculate number of band that we are going to work with
    n_bands_size = SM_coh.shape[0]

    # observable 1: Coherence
    coh_subset = SM_coh.copy()

    # # Keep all elements
    coh_dist_mask = np.ones_like(coh_subset).astype(np.bool_)
    
    # creating nan mask
    coh_nan_mask = ~np.isnan(coh_subset)
    
    # combining masks
    coh_final_mask = coh_dist_mask * coh_nan_mask
    # applying mask
    Coh_obs = coh_subset.copy()[coh_final_mask]

    # observable 2: Phase closure 
    Phase_closures_obs = Phase_closures[Mask_ph_closure].copy()

    def objective_function(SM0):  
        Covar_SM_model = Covar_modelled_calc( SM=SM0,
                                              theta_inc = DS_mean_inc_angle,
                                              freq_GHz = freq_GHz,
                                              clay_pct = clay_pct,
                                              sand_pct = sand_pct,
                                              ifg_pairs = None)
        # coherence component cost
        
        Coh_SM_model = np.abs(Covar_SM_model)[coh_final_mask]
        
        Coh_result = np.linalg.norm(Coh_SM_model-Coh_obs)
    
        # phase closure component cost
        
        Phase_closures_DeZan = Covar_2_phase_closures(Covar_SM_model)[Mask_ph_closure]
        Phase_closures_residuals = rewrap(Phase_closures_DeZan-Phase_closures_obs)
        Phase_closure_result = np.sum(np.power(Phase_closures_residuals, 2))
        
        # final cost
        result = Coh_result + Phase_closure_result
        #print(result)
        return result
    
    # bounds of inverted soil moistures
    #bounds_SM0 = (0,100)
    bounds_SM0 = (0,50)
    bounds = []
    for element in range(n_bands_size):
        bounds.append(bounds_SM0)
        
    # contraints
    
    cons_list = []
    
    # contraints related to dry SAR dates
    sm_dry_thres = sm_dry + 1
    for sar_dry_ind in dry_bands_inds:
        cons_list.append({'type': 'ineq', 'fun': lambda SM:  sm_dry_thres-SM[sar_dry_ind]})
    
    cons_list.append({'type': 'eq', 'fun': lambda SM:  SM[dry_index] - sm_dry})

    cons = tuple(cons_list)
    
    # minimization
    try:
        
        results = minimize(objective_function,
                           SM0,
                           options={'ftol':10e-1,'eps':0.05, 'maxiter':500},
                           #method='trust-constr',
                           method=opt_method,
                           bounds = bounds,
                           constraints = cons)
        
    except RuntimeError:
        
        results = {}
        results['x'] = np.full(SM0, np.nan, dtype = np.float64)
        
    return results['x']

def invert_sm(ph_DS:np.array,
              coh_sm:np.array,
              SM_index:np.array,
              dry_bands_inds:np.array,
              slc_dates:np.array,
              DS_ind:int,
              driest_date:datetime.datetime,
              inc_DS:float,
              band_start:int,
              band_end:int,
              nbands:int,
              opt_method:str,
              ph_closure_dist:int = 15,
              sm_dry_state:float = 3.0,
              freq_GHz:float = 5.405,
              clay_pct:float = 11,
              sand_pct:float = 79)->np.array:
    """Soil moisture invertion based on the De Zan`s model.

    Args:
        ph_DS (np.array): Interferometric (complex) values for the selected distributed scatterer (DS)
        coh_sm (np.array): Coherence due to soil moisture variations
        SM_index (np.array): Soil moisture proxy index information
        dry_bands_inds (np.array): Indices of SAR acquisitions considered dry
        slc_dates (np.array): Datetimes of SLC acquisitions
        DS_ind (int): Index of DS
        driest_date (datetime.datetime): Datetime of driest SLC acquisition
        inc_DS (float): Incidence angle in degrees.
        band_start (int): Index of the first SAR acqusition
        band_end (int): Index of the last SAR acqusition
        nbands (int): number of SAR acquisitions used
        opt_method (str): Minimization method
        ph_closure_dist (int, optional): the temporal difference (distance between SARs) that will be considered. 4 means only 3 sequential SARs will be used. Defaults to 15.
        sm_dry_state (float, optional): Driest soil moisture value. Defaults to 3.0.
        freq_GHz (float, optional): Operation frequency of SAR sensor in GHz. Defaults to 5.405.
        clay_pct (float, optional): Clay mass fraction (0-100)%. Defaults to 11.
        sand_pct (float, optional): Sand mass fraction (0-100)%. Defaults to 79.

    Returns:
        np.array: Soil moisture inverted values
        
    ..References:
        De Zan, F., Parizzi, A., Prats-Iraola, P., Lopez-Dekker, P., 2014. A SAR Interferometric Model for Soil Moisture. IEEE Trans. Geosci. Remote Sens. 52, 418-425. https://doi.org/10.1109/TGRS.2013.2241069   
    """
    #%% 1. Load data
    DS_mean_inc_angle = inc_DS[0]

    SM_covar = ph_DS[DS_ind, band_start:band_end, band_start:band_end]
    SM_coh = coh_sm[band_start:band_end,band_start:band_end]
    
    Phase_closures = Covar_2_phase_closures(SM_covar)
    dry_index = np.where(slc_dates == driest_date)[0][0]
    
    #%% 2. Inversion
    
    ## 1. Create the phase closure mask to select the short-time phase closures!
    # construct the triangle_idx_array and ifg_pairs matrices

    [Phase_closures_model,
      ifg_soil_moistures,
      G,
      triangle_idx_array,
      ifg_pairs]  = phase_closure_modelled_calc (np.random.rand(nbands),
                                                freq_GHz,
                                                clay_pct,
                                                sand_pct,
                                                DS_mean_inc_angle)
                                                 
    # create mask with short time spans of phase closures                                           
    Mask_ph_closure = np.zeros(triangle_idx_array.shape[0], np.bool)
    for ph_closure_ind, ph_closure_ints in enumerate(triangle_idx_array):
        int0 = ifg_pairs[ph_closure_ints[0]]
        int1 = ifg_pairs[ph_closure_ints[1]]
        int2 = ifg_pairs[ph_closure_ints[2]]
        
        min_int = np.min(np.array([int0, int1, int2]))
        max_int = np.max(np.array([int0, int1, int2]))
        if max_int-min_int < ph_closure_dist:
            Mask_ph_closure[ph_closure_ind] = True

    ## 2. Inversion
    
    SM_results = inversion(SM0 = SM_index,
                           DS_mean_inc_angle = DS_mean_inc_angle,
                           opt_method = opt_method,
                           Phase_closures = Phase_closures,
                           Mask_ph_closure = Mask_ph_closure,
                           SM_coh = SM_coh,
                           dry_index = dry_index,
                           dry_bands_inds = dry_bands_inds,
                           sm_dry = sm_dry_state,
                           freq_GHz = freq_GHz,
                           clay_pct = clay_pct,
                           sand_pct = sand_pct)
    
    return SM_results
