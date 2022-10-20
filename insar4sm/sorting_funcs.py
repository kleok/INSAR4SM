#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import warnings
import datetime
warnings.filterwarnings("ignore")

def coh_dist_calc(acq_sorted:np.array, Coh_array:np.array)->pd.DataFrame:
    """ Extracts the coherence values for each observed temporal distance

    Args:
        acq_sorted (np.array): sorted indices of SAR acquisitions (earlier to latest)
        Coh_array (np.array): coherence matrix between SAR acquisitions

    Returns:
        pd.DataFrame: Temporal distances and observed coherences
    """
    acq_sorted = acq_sorted.astype(np.int)
    n_bands = acq_sorted.shape[0]
    acquisition_dists = np.arange(1, n_bands)
    # acquisitions_sorted should be changed depending on the first image selection
    dist_list = []
    coh_list = []
    
    for dist in acquisition_dists:
        for img1_ind in range(n_bands):
            img2_ind = img1_ind+dist
            if img2_ind<n_bands:
                # get the index information from the unsorted initial acquisitions
                sorted_img1_ind = acq_sorted[img1_ind]
                sorted_img2_ind = acq_sorted[img2_ind]

                # find the coherence information
                Coh_img1_img2 = Coh_array[sorted_img1_ind,sorted_img2_ind]

                # append information
                dist_list.append(dist)
                coh_list.append(Coh_img1_img2)
    
    Coh_dist_df = pd.DataFrame()
    Coh_dist_df['dists']=dist_list
    Coh_dist_df['coh']=coh_list
    
    return Coh_dist_df


def find_sm_sorting(Datetimes:list,
                    coh_full_DS:np.array,
                    amp_DS:np.array,
                    ind_DS:int,
                    band_start:int,
                    band_end:int, 
                    driest_date:datetime.datetime, 
                    n_bands:int, 
                    n_iters:int,
                    denoise:bool = True)->np.array:

    """Ascending ordering of SAR acquisitions based on their surface soil moisture level (dry to wet)

    Args:
        Datetimes (list): datetime information for the SAR acquisitions
        coh_full_DS (np.array): the coherence matrix of the distributed scatterer (DS)
        amp_DS (np.array): the amplitude information of the DS
        ind_DS (int): the index of the DS that will be analyzed
        band_start (int): the index of first SAR acqusition
        band_end (int): the index of last SAR acqusition
        driest_date (datetime.datetime): the datetime of driest SAR acqusition
        n_bands (int): number of SARs that will be used
        n_iters (int): number of iterations related to sorting algorithm
        denoise (bool, optional): flag for denoising coherence matrix. Defaults to True.

    Returns:
        np.array: Ascending ordering of SAR acquisition based on their surface soil moisture level
    """
    #%% 1. Load data
    
    Coh_matrix = coh_full_DS[ind_DS,band_start:band_end,band_start:band_end].copy()

    Coh_matrix[np.diag_indices(Coh_matrix.shape[0])] = 0
    Datetimes_sel = Datetimes[band_start:band_end]
    dry_index = np.where(Datetimes_sel ==driest_date)[0][0]
    
    if denoise:
        u, s, vh = np.linalg.svd(Coh_matrix, full_matrices=True)
        use_bands = int(Coh_matrix.shape[0]/3)
        Coh_matrix = (u[:,:use_bands] @ np.diag(s[:use_bands])) @ vh[:use_bands,:]
        Coh_matrix[np.diag_indices(Coh_matrix.shape[0])] = 0

    amp_vector = amp_DS[ind_DS,:]
    
    #%% 2A. Find orderings based on amplitude information. We assume that the
    # monotonic relationship between soil moisture information.
    
    amp_dry = amp_vector[dry_index]
    amp_sm = amp_vector-amp_dry
    sm_amp_ordering = np.argsort(amp_sm)
    
    Coh_dist_df = coh_dist_calc(acq_sorted = sm_amp_ordering,
                                Coh_array = Coh_matrix)
    
    Max_Coh_dist_df = Coh_dist_df.groupby(by=['dists']).max()
    m_amp, b_amp = np.polyfit(Max_Coh_dist_df.index, Max_Coh_dist_df['coh'], 1)  

    #%% 3A. Find sm combinations based on coherence matrix
    
    sm_orderings = np.zeros((n_iters, n_bands), dtype=np.int32)
    
    for iteration in range(n_iters):
        date_index = dry_index
        sm_ordering = [dry_index]
        Coh_temp = Coh_matrix.copy()
        
        # Assign approximate sm levels according to coherence values.
        while np.count_nonzero(Coh_temp) > 0:
            # get the vector information from coherence matrix
            coh_vector = np.squeeze(Coh_temp[date_index,:])
            non_zero_inds = list(np.nonzero(coh_vector)[0])
            # Descending sorting
            similar_inds = np.argsort(coh_vector)[::-1][0:2]
            non_zero_similar_inds= [ind for ind in similar_inds if ind in non_zero_inds]
            coh_diff = np.abs(np.diff(coh_vector[non_zero_similar_inds]))
            if coh_diff < 0.1:
                sm_approx = np.random.choice(non_zero_similar_inds, 1, replace=False)[0]
                #print('We picked: {}'.format(sm_approx))
                
                # put zeros in column/row of the coherence 
                Coh_temp[date_index, :] = 0 
                Coh_temp[:, date_index] = 0 
                
                # update
                date_index = sm_approx
                sm_ordering.append(date_index)
                
            else:
                sm_approx = np.argmax(coh_vector)
                # put zeros in column/row of the coherence 
                Coh_temp[date_index, :] = 0 
                Coh_temp[:, date_index] = 0 
                
                # update
                date_index = sm_approx
                sm_ordering.append(date_index)

        sm_orderings[iteration,:] = sm_ordering
    
    #%% 3B. Calculate coherence slopes of all sm combinations
    slopes=np.zeros((n_iters))
    
    for iteration in range(n_iters):
        
        Coh_dist_df = coh_dist_calc(acq_sorted = sm_orderings[iteration,:],
                                    Coh_array = Coh_matrix)
    
        Max_Coh_dist_df = Coh_dist_df.groupby(by=['dists']).max()
        m, b = np.polyfit(Max_Coh_dist_df.index, Max_Coh_dist_df['coh'], 1)  
        
        slopes[iteration]=m
        
        # Max_Coh_dist_df['fitted'] = m*Max_Coh_dist_df.index + b
        # fig, ax = plt.subplots(figsize=(10,8))
        # ax.scatter(Coh_dist_df['dists'], Coh_dist_df['coh'])
        # Max_Coh_dist_df['fitted'].plot(style='-', color='r')
        # plt.xlabel('Acquisition_distances', fontsize=18)
        # plt.ylabel('Coherence', fontsize=16)
        # plt.title('Ordering: {} \n Fitted line: Max_coh={}*time+{}'.format(sm_orderings[iteration,:].astype(np.int), round(m,2), round(b,2)))

    #%% 3C. Select the best sorting based on frequency of best sm_combinations
    sm_coh_ordering = np.full((n_bands),-999, dtype=np.int32)
    
    has_duplicates = len(sm_coh_ordering) != len(np.unique(sm_coh_ordering))
    has_gaps = -999 in sm_coh_ordering
    n_iters = 10
    iter_ind = 1
    
    while (has_duplicates or  has_gaps) and (iter_ind<n_iters) :
        n_best = int(10*iter_ind)
        best_sm_orderings = sm_orderings[np.argsort(slopes)[:n_best], :]
        
        # Create a frequency array
        
        # the frequency array has two dimensions.
        # columns--> the position of the ordering
        # rows ----> the freqency of each band.
        # row index --> the number of the band
        
        freq_array = np.zeros((n_bands, n_bands), dtype=np.float32)
        for position in range(n_bands):
            a = best_sm_orderings[:,position]
            unique_bands, counts_bands = np.unique(a, return_counts=True)
            for ind, band in enumerate(unique_bands):
                freq_array[band,position] = counts_bands[ind]
                
                
        # For each position find the dominant image and assign to the 
        # sm_coh_ordering array

        positions_remaining = []
        for position in range(n_bands):
            best_image = np.bincount(best_sm_orderings[:,position]).argsort()[::-1][0]
            Freq_A = np.bincount(best_sm_orderings[:,position])[best_image]/n_best
            if Freq_A > 0.5:
                sm_coh_ordering[position] = best_image
                freq_array[best_image, :] = 0.
                freq_array[:, position] = 0.
            else:
                positions_remaining.append(position)
                continue
            
        bands_remaining = np.array([band for band in range(n_bands) if band not in sm_coh_ordering])
    
        for image in bands_remaining:  
            best_position  = np.argmax(freq_array[image, :])
            if best_position in positions_remaining:
                sm_coh_ordering[best_position] = image
                freq_array[image,:] = 0.
                freq_array[:,best_position] = 0.
                positions_remaining = np.setdiff1d(positions_remaining,best_position)     
            else:
                continue
            
        has_duplicates = len(sm_coh_ordering) != len(np.unique(sm_coh_ordering))
        has_gaps = -999 in sm_coh_ordering
        iter_ind = iter_ind + 1
    
    # If we still have gaps or duplicates we just select the steepest combinations
    if (has_duplicates or has_gaps) :
        #print('Sorting algorithm did not converged!')
        sm_coh_ordering = sm_orderings[np.argsort(slopes)[0], :]
        

    Coh_dist_df = coh_dist_calc(acq_sorted = sm_coh_ordering,
                                Coh_array = Coh_matrix)
    
    Max_Coh_dist_df = Coh_dist_df.groupby(by=['dists']).max()
    m_coh, b_coh = np.polyfit(Max_Coh_dist_df.index, Max_Coh_dist_df['coh'], 1)  

    #%% 3C. Select the best sorting based on frequency of best sm_combinations

    if np.abs(m_coh)>np.abs(m_amp):
        best_sorting = sm_coh_ordering
    else:
        best_sorting = sm_amp_ordering

    return best_sorting
