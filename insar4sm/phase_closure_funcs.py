#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

def rewrap(data:np.array)-> np.array:
    """Wraps the data between -pi and pi.

    Args:
        data (np.array): the data need to be wrapped

    Returns:
        np.array: Wrapped data. Values are between -pi and pi
    """
    return data - np.round(data/2.0/np.pi)*2.0*np.pi

def Covar_2_phase_closures(DS_Covar:np.array)->np.array:
    """Calculates observed phase closures from observed covariance matrix

    Args:
        DS_Covar (np.array): covariance matrix that contains observed complex interferometric values

    Returns:
        np.array: Observed phase closures
    """
    
    assert len(DS_Covar.shape) == 2
    assert DS_Covar.shape[0] == DS_Covar.shape[1]
    
    n_sars = DS_Covar.shape[0]
    n_ifgs = int((n_sars)*(n_sars-1)/2)
    
    sar_primaries = np.arange(n_sars)

    ifg_pairs=[]
    for sar_primary in sar_primaries:
        sar_secondaries = sar_primaries[sar_primaries>sar_primary]
        for sar_secondary in sar_secondaries:
            ifg_pairs.append([sar_primary,sar_secondary])
        
    assert len(ifg_pairs) == n_ifgs
    
    ifg_phases = np.zeros(n_ifgs, dtype=np.float32)
    for ifg_index, sar_indices in enumerate(ifg_pairs):
        sar1, sar2 = sar_indices
        ifg_phases[ifg_index] = np.angle(DS_Covar[sar1,sar2])

    # construct the design matrix (G) for phase closures
    # the model is Gd = 0, 
    # where d is the interferometric phases of pairs of SAR acquisitions
    
    triangle_idx = []
    
    for ifg1 in ifg_pairs:
        sar1, sar2 = ifg1

        # ifg2 is required to have the same sar1 acquisition with the ifg1
        # find all the candidate sar3 acquisitions
        sar3_list =[]
        for ifg2 in ifg_pairs:
            if sar1 == ifg2[0] and ifg2 != ifg1:
                sar3_list.append(ifg2[1])
                
        # check if ifg with sar2 and sar3 exists        
        if len(sar3_list)>0:
            for sar3 in sar3_list:
                ifg3=[sar2,sar3]
                if ifg3 in ifg_pairs:
                    #
                    ifg1_checked = [sar1, sar2]
                    ifg2_checked = [sar1, sar3]
                    ifg3_checked = [sar2, sar3]
                    
                    # Append the indices of the interferograms to the index array
                    triangle_idx.append([ifg_pairs.index(ifg1_checked),
                                         ifg_pairs.index(ifg2_checked),
                                         ifg_pairs.index(ifg3_checked)])
                    
    triangle_idx_array = np.array(triangle_idx, dtype=np.int)
    triangle_idx_array = np.unique(triangle_idx_array, axis=0)
        
    # triangle_idx to G
    num_triangle = triangle_idx_array.shape[0]
    G = np.zeros((num_triangle, n_ifgs), np.float32)
        
    for i in range(num_triangle):
        G[i, triangle_idx_array[i, 0]] = 1
        G[i, triangle_idx_array[i, 1]] = -1
        G[i, triangle_idx_array[i, 2]] = 1     
    
    Phase_closures=G @ ifg_phases
    
    Phase_closures = rewrap(Phase_closures)
    
    
    return Phase_closures