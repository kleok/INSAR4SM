#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import numpy as np
from insar4sm.dielectic_models import hallikainen_1985_calc
from itertools import combinations

def vertical_wavenumber_calc(mv_pct:float,
                            theta_inc:float|None = None,
                            freq_GHz:float = 5.405,
                            clay_pct:float  = 13.0,
                            sand_pct:float = 87.0,
                            mu:float = 4*np.pi*10e-7 )-> np.complex:
    r""" A simplified version for the calculation of complex wavenumber was used based on (Eq 3) of De Zan et al., 2018

    .. math:: k = \sqrt{\omega^2\mu\epsilon}

    Args:
        mv_pct (float): Volumetric moisture content (0-100)%
        theta_inc (float, optional): Incidence angle in degrees. Defaults to None.
        freq_GHz (float, optional): Operation frequency of SAR sensor in GHz. The default value is 5.405 which refers to Sentinel-1. Defaults to 5.405.
        clay_pct (float, optional): Clay mass fraction (0-100)%. Defaults to 13.0.
        sand_pct (float, optional): Sand mass fraction (0-100)%. Defaults to 87.0.
        mu (float, optional): Dielectric permeability. Units are Newton/Ampere^2. Defaults to 4*np.pi*10e-7.

    Returns:
        np.complex: Vertical complex wavenumber

    ..References:
        De Zan, F., Gomba, G., 2018. Vegetation and soil moisture inversion from SAR closure phases: First experiments and results. Remote Sens. Environ. 217, 562-572. https://doi.org/10.1016/j.rse.2018.08.034
    
    """
    wmega = 2*np.pi*freq_GHz*10e9
    epsilon = hallikainen_1985_calc(clay_pct, sand_pct, mv_pct, freq_GHz)
    if theta_inc is None:
        kx = 0.0
    else:
        c = 299792458 # m/s
        freq_Hz = freq_GHz*10e9
        wavelength = c/freq_Hz
        kx = (2*np.pi*np.sin(np.radians(theta_inc)))/wavelength
    
    k = np.sqrt((wmega**2)*epsilon*mu-kx**2)
    
    # Since the medium is lossy, k`z and ε` are complex numbers. 
    # The aforementioned equation has two solutions because of the ambiguity 
    # of the square root, and we chose the “physical” one, i.e., the one with 
    # a negative imaginary part. 
    # This corresponds to a wave that attenuates going downward, so that 
    # |E`(x, y, z)| → 0 when z → ∞.    
    if type(k) == np.ndarray:
        k[np.imag(k) > 1] = np.conjugate(k[np.imag(k) > 1])
    else:
        if np.imag(k)>0:
            k = np.conjugate(k)
        
    return k

def InSAR_phase_model(mv_pct1:float, mv_pct2:float, theta_inc:float|None=None, freq_GHz:float= 5.405, clay_pct:float = 13.0, sand_pct:float = 87.0, beta:float=0.05)->np.complex:
    r"""Calculation of modelled interferometric value based on De Zan`s model. The implemented equation is Eq. 6 of Palmizano et al., 2022.

     .. math:: I_{12} = \frac{1}{2j(k_1-k_2^{*})+\beta}
    
    Args:
        mv_pct1 (float): Volumetric moisture content (0-100)% of SAR acquisition 1.
        mv_pct2 (float): Volumetric moisture content (0-100)% of SAR acquisition 2.
        theta_inc (float, optional): Incidence angle in degrees. Defaults to None.
        freq_GHz (float, optional): Operation frequency of SAR sensor in GHz. The default value is 5.405 which refers to Sentinel-1. Defaults to 5.405.
        clay_pct (float, optional): Clay mass fraction (0-100)%. Defaults to 13.0.
        sand_pct (float, optional): Sand mass fraction (0-100)%. Defaults to 87.0.
        beta (float, optional): beta corresponds to either increased attenuation of the electromagnetic wave in the soil or, equivalently, to a decrease of its penetration depth (Palmizano et al., 2022). Defaults to 0.05.

    Returns:
        np.complex: complex interferometric phase based on De Zan`s model.

    ..References:
        - De Zan, F., Gomba, G., 2018. Vegetation and soil moisture inversion from SAR closure phases: First experiments and results. Remote Sens. Environ. 217, 562-572. https://doi.org/10.1016/j.rse.2018.08.034
        - Palmisano, D., Satalino, G., Balenzano, A., Mattia, F., 2022. Coherent and Incoherent Change Detection for Soil Moisture Retrieval From Sentinel-1 Data. IEEE Geosci. Remote Sens. Lett. 19, 1-5. https://doi.org/10.1109/LGRS.2022.3154631

    """
    k1 = vertical_wavenumber_calc(mv_pct1, theta_inc, freq_GHz, clay_pct, sand_pct)
    k2 = vertical_wavenumber_calc(mv_pct2, theta_inc, freq_GHz, clay_pct, sand_pct)

    I12 = 1/(2*(1j*k1-1j*np.conjugate(k2))+beta)
    
    return I12

def InSAR_coherence_model(mv_pct1:float, mv_pct2:float, theta_inc:float|None=None, freq_GHz:float = 5.405, clay_pct:float = 13.0, sand_pct:float = 87.0, beta:float=0.05)->np.complex:
    """Calculated of modelled complex coherence using De Zan`s model using two soil moistures.

    Args:
        mv_pct1 (float): Volumetric moisture content (0-100)% of SAR acquisition 1.
        mv_pct2 (float): Volumetric moisture content (0-100)% of SAR acquisition 2.
        theta_inc (float, optional): Incidence angle in degrees. Defaults to None.
        freq_GHz (float, optional): Operation frequency of SAR sensor in GHz. The default value is 5.405 which refers to Sentinel-1. Defaults to 5.405.
        clay_pct (float, optional): Clay mass fraction (0-100)%. Defaults to 13.0.
        sand_pct (float, optional): Sand mass fraction (0-100)%. Defaults to 87.0.
        beta (float, optional): beta corresponds to either increased attenuation of the electromagnetic wave in the soil or, equivalently, to a decrease of its penetration depth (Palmizano et al., 2022). Defaults to 0.05.

    Returns:
        np.complex: complex coherence value based on De Zan`s model

    ..References:
        De Zan, F., Parizzi, A., Prats-Iraola, P., Lopez-Dekker, P., 2014. A SAR Interferometric Model for Soil Moisture. IEEE Trans. Geosci. Remote Sens. 52, 418-425. https://doi.org/10.1109/TGRS.2013.2241069
        De Zan, F., Gomba, G., 2018. Vegetation and soil moisture inversion from SAR closure phases: First experiments and results. Remote Sens. Environ. 217, 562-572. https://doi.org/10.1016/j.rse.2018.08.034
    
    """

    I11 = InSAR_phase_model(mv_pct1, mv_pct1, theta_inc, freq_GHz, clay_pct, sand_pct, beta)
    I22 = InSAR_phase_model(mv_pct2, mv_pct2, theta_inc, freq_GHz, clay_pct, sand_pct, beta)
    I12 = InSAR_phase_model(mv_pct1, mv_pct2, theta_inc, freq_GHz, clay_pct, sand_pct, beta)
    
    Coh12 = I12/np.sqrt(I11*I22)
    
    return Coh12

def Single_phase_closure_model(mv_pct1:float, mv_pct2:float, mv_pct3:float, theta_inc:float, freq_GHz:float, clay_pct:float, sand_pct:float)->float:
    """Calculate the modelled phase closure based on De Zan`s model for three soil moistures.

    Args:
        mv_pct1 (float): Volumetric moisture content (0-100)% of SAR acquisition 1
        mv_pct2 (float): Volumetric moisture content (0-100)% of SAR acquisition 2
        mv_pct3 (float): Volumetric moisture content (0-100)% of SAR acquisition 3
        theta_inc (float): Incidence angle in degrees.
        freq_GHz (float): Operation frequency of SAR sensor in GHz. The default value is 5.405 which refers to Sentinel-1.
        clay_pct (float):  Clay mass fraction (0-100)%.
        sand_pct (float): Sand mass fraction (0-100)%.

    Returns:
        float: modelled phase closure based on De Zan`s model

    ..References:
        De Zan, F., Parizzi, A., Prats-Iraola, P., Lopez-Dekker, P., 2014. A SAR Interferometric Model for Soil Moisture. IEEE Trans. Geosci. Remote Sens. 52, 418-425. https://doi.org/10.1109/TGRS.2013.2241069   
    
    """

    I12 = InSAR_phase_model(mv_pct1, mv_pct2, theta_inc, freq_GHz, clay_pct, sand_pct)
    I23 = InSAR_phase_model(mv_pct2, mv_pct3, theta_inc, freq_GHz, clay_pct, sand_pct)
    I31 = InSAR_phase_model(mv_pct3 ,mv_pct1, theta_inc, freq_GHz, clay_pct, sand_pct)
    
    Single_Ph_closure_model = np.angle(I12*I23*I31)

    return Single_Ph_closure_model

def Covar_modelled_calc(SM:np.array, theta_inc:float, freq_GHz:float, clay_pct:float, sand_pct:float, ifg_pairs:list|None=None)->np.array:
    """Calculates covariance matrix based on De Zan`s model

    Args:
        SM (np.array): Soil moisture
        theta_inc (float): Incidence angle in degrees.
        freq_GHz (float): Operation frequency of SAR sensor in GHz.
        clay_pct (float): Clay mass fraction (0-100)%.
        sand_pct (float): Sand mass fraction (0-100)%.
        ifg_pairs (list, optional): Interferometric pairs. Defaults to None.

    Returns:
        np.array: Modelled covariance matrix

    ..References:
        De Zan, F., Parizzi, A., Prats-Iraola, P., Lopez-Dekker, P., 2014. A SAR Interferometric Model for Soil Moisture. IEEE Trans. Geosci. Remote Sens. 52, 418-425. https://doi.org/10.1109/TGRS.2013.2241069   
    
    """
    n_sars = SM.shape[0]
    n_ifgs = int((n_sars)*(n_sars-1)/2) 
    
    if isinstance(ifg_pairs, type(None)):
        sar_primaries = np.arange(n_sars)
        ifg_pairs = np.array([(l,r) for l,r in combinations(sar_primaries, 2)])

            
        assert len(ifg_pairs) == n_ifgs
    
    Covar_model = np.ones((n_sars,n_sars), dtype=np.complex128)

    


    res = InSAR_coherence_model(mv_pct1 = SM[ifg_pairs[:,0]],
                                    mv_pct2 = SM[ifg_pairs[:,1]],
                                    theta_inc = theta_inc,
                                    freq_GHz= freq_GHz,
                                    clay_pct = clay_pct,
                                    sand_pct = sand_pct) 
    Covar_model[ifg_pairs[:,0], ifg_pairs[:,1]] = res
    Covar_model[ifg_pairs[:,1], ifg_pairs[:,0]] = np.conj(res)

    return Covar_model

def phase_closure_modelled_calc(SM:np.array, freq_GHz:float, clay_pct:float, sand_pct:float, theta_inc:float|None= None, ifg_pairs:list|None=None, triangle_idx_array:np.array=None)-> tuple[np.array, np.array, np.array, np.array, list]:
    """Calculates phase closures based on De Zan`s model

    Args:
        SM (np.array): Soil moisture
        freq_GHz (float): Operation frequency of SAR sensor in GHz.
        clay_pct (float): Clay mass fraction (0-100)%.
        sand_pct (float): Sand mass fraction (0-100)%.
        theta_inc (float, optional): Incidence angle in degrees. Defaults to None.
        ifg_pairs (list, optional): Interferometric pairs. Defaults to None.
        triangle_idx_array (np.array, optional): Indices of interferograms for each triplet. Defaults to None.

    Returns:
        tuple[np.array, np.array, np.array, np.array, list]: Phase closures, soil moisture change for each interferogram, Design matrix for phase closures, Indices of interferograms for each triplet, Interferometric pairs 
    
    ..References:
        De Zan, F., Parizzi, A., Prats-Iraola, P., Lopez-Dekker, P., 2014. A SAR Interferometric Model for Soil Moisture. IEEE Trans. Geosci. Remote Sens. 52, 418-425. https://doi.org/10.1109/TGRS.2013.2241069   
    
    """
    n_sars = SM.shape[0]
    # calculates the number of interferograms based on number of SAR images
    n_ifgs = int((n_sars)*(n_sars-1)/2) 
    
    if isinstance(ifg_pairs, type(None)):
        sar_primaries = np.arange(n_sars)
        ifg_pairs=[]
        for sar_primary in sar_primaries:
            sar_secondaries = sar_primaries[sar_primaries>sar_primary]
            for sar_secondary in sar_secondaries:
                ifg_pairs.append([sar_primary,sar_secondary])
            
        assert len(ifg_pairs) == n_ifgs
    
    ifg_soil_moistures = np.zeros(n_ifgs, dtype=np.float32)
    for ifg_index, sar_indices in enumerate(ifg_pairs):
        sar1, sar2 = sar_indices
        SM_sar1 = SM[sar1]
        SM_sar2 = SM[sar2]
        ifg_soil_moistures[ifg_index] = SM_sar1 - SM_sar2
        
    # construct the design matrix (G) for phase closures
    # the model is Gd = 0, 
    # where d is the soil moisture differences of pairs of SAR acquisitions
    
    if isinstance(triangle_idx_array, type(None)):      
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
 
    ifg_triples = np.array([(i,j,k) for i,j,k in combinations(sar_primaries, 3)])
    Phase_closures_model = Single_phase_closure_model(SM[ifg_triples[:,0]],
                                                            SM[ifg_triples[:,1]],
                                                            SM[ifg_triples[:,2]],
                                                            theta_inc,
                                                            freq_GHz,
                                                            clay_pct,
                                                            sand_pct)

    return Phase_closures_model, ifg_soil_moistures, G, triangle_idx_array, ifg_pairs   











