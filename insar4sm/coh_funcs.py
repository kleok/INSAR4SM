#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

def exp_fit(x:np.array, a:float, b:float) -> np.array:
    """Exponential model between time and coherence due to temporal decorrelation.

    .. math:: y = a e^{xb}

    Args:
        x (np.array): known quantity (time spans) 
        a (float): parameter to estimated
        b (float): parameter to estimated

    Returns:
        np.array: modelled data (coherence due to temporal decorrelation)
    """
    y = a* np.exp(-x*b)
    return y
        
def temp_coh__model(t:np.array, gamma_0:float, tau:float, gamma_k:float)->np.array:
    """Exponential model between time and coherence due to temporal decorrelation according to Parizzi et al., 2009

    .. math:: \gamma_t = (\gamma_0 - \gamma_k)e^{-t/\tau} + \gamma_k

    Args:
        t (np.array): known quantity (time spans) 
        gamma_0 (float): parameter to estimated
        tau (float): parameter to estimated
        gamma_k (float): parameter to estimated

    Returns:
        np.array: modelled data (coherence due to temporal decorrelation)

    ..References:
        Parizzi, A., Cong, X., & Eineder, M. (2009). First Results from Multifrequency Interferometry. A comparison of different decorrelation time constants at L, C, and X Band. ESA Scientific Publications, (SP-677), 1-5.
    """
 
    gamma_t = (gamma_0-gamma_k)* np.exp(-t/tau) + gamma_k
    
    return gamma_t  
        
def calc_SM_coherences(Datetimes:list,
                       SM_sorting:np.array,
                       n_bands:int,
                       n_ds:int,
                       ind_DS:int,
                       coh_full_DS:np.array,
                       band_start:int,
                       band_end:int,
                       n_dry_bands:int,
                       max_coh_flag:bool,
                       simple_exp_model:bool)->tuple[np.array, np.array, np.array]:
    """Calculates the coherence information due to soil moisture variations

    Args:
        Datetimes (list): list of SAR acquisition datetimes (datetime.datetime objects)
        SM_sorting (np.array): Ascending ordering of SAR acquisitions based on their surface soil moisture (SSM) level
        n_bands (int): number of SAR acquisitions used
        n_ds (int): number of distributed scatterer that will be computed
        ind_DS (int): index of distributed scatterer that will be computed
        coh_full_DS (np.array): 2D array of the raw coherence (real) values 
        band_start (int): index of first SAR acquisition
        band_end (int): index of last SAR acquisition
        n_dry_bands (int): number of SAR acquisitions that are considered "dry"
        max_coh_flag (bool): parameter related with the selection of coherence values for modelling temporal decorrelation. If True maximum coherence values for each time span are selected
        simple_exp_model (bool): parameter related with the selection of coherence values for modelling temporal decorrelation. If True the simple exponential model is selected.

    Returns:
        tuple[np.array, np.array, np.array]: coherence matrix related to soil moisture variations, coherence mean in respect with dry conditions, coherence std in respect with dry conditions
    """
    #%% 1. Load data
    
    Datetimes = Datetimes[band_start:band_end].copy()
    coh = coh_full_DS[ind_DS,band_start:band_end,band_start:band_end].copy()
    coh[np.diag_indices_from(coh)] = np.nan
    dry_bands_inds = SM_sorting[:n_dry_bands].copy()
    
    
    #%% 2. In order to be sure about the selected dry bands we construct the 
    # cohence matrix of dry bands x dry bands. We expect to see high 
    # values because of the small sm variations between dry values
    
    #coh = (coh - np.nanmin(coh)) / (np.nanmax(coh) - np.nanmin(coh))
    
    # Selection of coherence values related with dry acquisitions
    coh_dry = coh[dry_bands_inds,:][:, dry_bands_inds].copy()
    
    # identify "not-dry" bands and update the indices of dry bands
    bad_dry_bands = (coh_dry<np.nanquantile(coh_dry.flatten(), 0.1)).nonzero()[0]
    drop_dry_bands = [item for item, count in Counter(bad_dry_bands).items() if count > 1]
    drop_dry_bands_inds = SM_sorting[drop_dry_bands]
    drop_dry_bands_indices=np.argwhere(np.isin(dry_bands_inds , drop_dry_bands_inds))
    dry_bands_inds=np.delete(dry_bands_inds,drop_dry_bands_indices)
    
    # update the number of dry bands
    n_dry_bands = dry_bands_inds.shape[0]
    
    # update the dry coherence matrix
    coh_dry = coh[dry_bands_inds,:][:, dry_bands_inds].copy()

    # select only the upper triangular elements
    num_tri_elements = np.triu_indices_from(coh_dry,1)[0].shape[0]
    num_tri_indices = np.triu_indices_from(coh_dry,1)
    indices1 = num_tri_indices[0]
    indices2 = num_tri_indices[1]
    
    # Get the temporal infrormation of dry bands
    dry_datetimes = Datetimes[dry_bands_inds]
    

    #%% 3. calculate coherence related to temporal decorrelation
    # initialization
    coh_sm_mean = np.zeros((n_ds,n_bands), dtype=np.float32)
    coh_sm_std = np.zeros((n_ds,n_bands), dtype=np.float32)
    coh_sm = np.zeros((n_ds,n_bands,n_bands), dtype=np.float32)
    coh_decor = np.full((n_bands, n_bands), np.nan, dtype=np.float64)

    # get the coherence values for each time combination of dry dates
    Dt = np.array([np.abs((dry_datetimes[indices2[int_n]]-dry_datetimes[indices1[int_n]]).days) for int_n in range(num_tri_elements)])
    Permanent_Cohs = np.asarray(coh_dry[num_tri_indices])
    
    # select the maximum coherence values for each time combination
    coh_time_df = pd.DataFrame(np.array([Dt,Permanent_Cohs])).T
    max_coh_time_df = coh_time_df.groupby([0]).max()
    Dt_max = max_coh_time_df.index.values
    Permanent_max_Cohs = max_coh_time_df[1].values
    
    # Exponential models 
    if simple_exp_model: # simple exponential model
        if max_coh_flag:
            fit = curve_fit(exp_fit, Dt_max, Permanent_max_Cohs)
            a, b = fit[0]
            gamma_t_modelled= exp_fit(Dt, a, b)
        else:
            fit = curve_fit(exp_fit, Dt.astype(dtype=np.float128), Permanent_Cohs.astype(dtype=np.float128))
            a, b = fit[0]
            gamma_t_modelled= exp_fit(Dt, a, b)
    else: # Parizzi`s exponential model
        if max_coh_flag:
            try:
                fit = curve_fit(temp_coh__model, Dt_max, Permanent_max_Cohs, maxfev = 2000)
                gamma_0, tau, gamma_k = fit[0]
                gamma_t_modelled = temp_coh__model(Dt, gamma_0, tau, gamma_k)
            except RuntimeError:
                print("Coherence loss due to temporal decorrelation cannot be estimated!")
                coh_sm[ind_DS, :, :]  = coh
                coh_sm_dry = coh_sm[ind_DS, dry_bands_inds,:].copy()
                coh_sm_mean[ind_DS,:] = np.nanmean(coh_sm_dry, axis=0)
                coh_sm_std[ind_DS,:] = np.nanstd(coh_sm_dry, axis=0)
                return coh_sm, coh_sm_mean, coh_sm_std
        else:
            fit = curve_fit(temp_coh__model, Dt, Permanent_Cohs, maxfev = 2000)
            gamma_0, tau, gamma_k = fit[0]
            gamma_t_modelled = temp_coh__model(Dt, gamma_0, tau, gamma_k)

    # visualize reconstructed coherence deccorellation
    coh_decor_dry = np.full((n_dry_bands, n_dry_bands), np.nan, dtype=np.float64)
    for ind in range(num_tri_elements):
        ind1 = indices1[ind]
        ind2 = indices2[ind]
        coh_decor_dry[ind1, ind2] = gamma_t_modelled[ind]
        coh_decor_dry[ind2, ind1] = gamma_t_modelled[ind]

    # based on model calculate the temporal decorrelation of all combinations
    num_tri_elements = np.triu_indices_from(coh,1)[0].shape[0]
    num_tri_indices = np.triu_indices_from(coh,1)
    indices1 = num_tri_indices[0]
    indices2 = num_tri_indices[1]
    Dt = np.array([np.abs((Datetimes[indices1[int_n]]-Datetimes[indices2[int_n]]).days) for int_n in range(num_tri_elements)])
  
    for ind in range(num_tri_elements):
        ind1 = indices1[ind]
        ind2 = indices2[ind]
        if simple_exp_model:
            coh_decor[ind1, ind2] = exp_fit(Dt[ind], a, b)
            coh_decor[ind2, ind1] = exp_fit(Dt[ind], a, b)
        else:
            coh_decor[ind1, ind2] = temp_coh__model(Dt[ind], gamma_0, tau, gamma_k)
            coh_decor[ind2, ind1] = temp_coh__model(Dt[ind], gamma_0, tau, gamma_k)

    # clip the values of modelled temporal decorrelation 
    coh_decor = np.clip(coh_decor, 0.3, 1)

    # divide the modelled temporal decorrelation to find the coherence due to sm variations
    coh_values = coh/coh_decor

    # replace with nan the coherence values that are biggest than one.
    coh_values[coh_values>1] = np.nan

    # # rescale the coherence between 0 and 1
    # coh_range = np.nanquantile(coh_values,0.99) - np.nanquantile(coh_values,0.01)
    # coh_values = (coh_values - np.nanquantile(coh_values,0.01))/coh_range

    # coh_values[coh_values>1] = np.nan
    # coh_values[coh_values<0] = np.nan

    # print('quantile 01: {}'.format(np.nanquantile(coh_values,0.01)))
    # print('quantile 99: {}'.format(np.nanquantile(coh_values,0.99)))
    # print('range: {}'.format(coh_range))

    # coherence matrix related to soil moisture variations
    coh_sm[ind_DS, :, :]  = coh_values

    # calculates the coherence mean/std in respect with dry SAR acquisitions
    coh_sm_dry = coh_sm[ind_DS, dry_bands_inds,:].copy()
    coh_sm_mean[ind_DS,:] = np.nanmean(coh_sm_dry, axis=0)
    coh_sm_std[ind_DS,:] = np.nanstd(coh_sm_dry, axis=0)
        
    return coh_sm, coh_sm_mean, coh_sm_std
# %%
