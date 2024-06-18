#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from itertools import combinations

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
    sar_primaries = np.arange(n_sars)
    n_triples = int((n_sars)*(n_sars-1)*(n_sars-2)/6) # (n choose k) = n!/k!(n-k)! for n=n_sars, k=3
    ifg_triples = np.array([(i,j,k) for i,j,k in combinations(sar_primaries, 3)])
   
    triple_phases = np.zeros((n_triples, 3))
    triple_phases[:,0] = np.angle(DS_Covar[ifg_triples[:,0], ifg_triples[:,1]])
    triple_phases[:,1] = -np.angle(DS_Covar[ifg_triples[:,0], ifg_triples[:,2]])
    triple_phases[:,2] = np.angle(DS_Covar[ifg_triples[:,1], ifg_triples[:,2]])
    
    Phase_closures = np.sum(triple_phases, axis=1)       
    Phase_closures = rewrap(Phase_closures)
 
    
    return Phase_closures