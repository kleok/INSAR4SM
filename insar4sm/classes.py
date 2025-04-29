#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg.lapack import zheevr 

#--------------------- INSAR4SM functionalities --------------
from insar4sm.prep_slc_data import tops_2_vrt
from insar4sm.prep_meteo import convert_to_df
from insar4sm.meteo_funcs import find_dry_SARs
from insar4sm.calc_int_data import calc_int_amp
from insar4sm.gridding import create_grid_xy
from insar4sm.DS_funcs import compute_Covar_Coh_weighted, get_DS_pixels, llh2xy
from insar4sm.DS_funcs import find_best_pixels_ph, find_best_pixels_amp_fast, find_best_pixels_amp
from insar4sm.sorting_funcs import find_sm_sorting
from insar4sm.coh_funcs import calc_SM_coherences
from insar4sm.calc_SM_index import calc_burgi_sm_index
from insar4sm.inversion import invert_sm
from insar4sm.Soilgrids.soil_funcs import get_soilgrids_value

class INSAR4SM_stack:
    """Constructing and processing InSAR4SM stack
    """
    def __init__(self, topstackDir, projectname, n_CPUs, AOI, meteo_file, sand, clay, orbit_time, days_back, export_dir):
        
        #----------------------Basic parms ------------
        self.projectname = projectname
        self.topstackDir = topstackDir
        self.export_dir = export_dir
        self.AOI = AOI
        self.meteo_file = meteo_file
        self.orbit_time = orbit_time
        self.buffer = 0.005
        self.CPUs = n_CPUs
        self.sand_data = sand
        self.clay_data = clay
        
        #----------------------Get directories names ------------
        self.projectDir ='{}/{}/INSAR4SM_datasets'.format(self.export_dir, self.projectname)
        self.WorkDir = '{}/{}/INSAR4SM_processing'.format(self.export_dir, self.projectname)
        self.plot_dir='{}/{}/plots'.format(self.export_dir, self.projectname)  
        self.SM_dir = os.path.join(self.WorkDir,'SM')

        #----------------------Create directory structure------------
        if not os.path.exists(self.projectDir): os.makedirs(self.projectDir)
        if not os.path.exists(self.WorkDir): os.makedirs(self.WorkDir)
        if not os.path.exists(self.SM_dir): os.makedirs(self.SM_dir)
        if not os.path.exists(self.plot_dir): os.makedirs(self.plot_dir)
        
        #----------------------Processing parms ------------
        self.days_back = days_back
        self.dry_dates_ratio = 0.3  # percent of SAR acquisitions that are assumed dry (<0.04 m3/m3)
        
    def prepare_datasets(self):
        """Prepares topstack datasets for InSAR4SM processing
        """
        
        tops_2_vrt(indir = os.path.join(self.topstackDir,'merged/'),
                   outdir = os.path.join(self.projectDir,'slcs'),
                   stackdir = os.path.join(self.projectDir,'coreg_stack'),
                   AOI_WGS84 = self.AOI,
                   geomdir = os.path.join(self.projectDir,'geometry'))
        
        # Find the indices that corresponds to the defined Start_time and End_time
        self.slc_filenames = glob.glob(os.path.join(self.projectDir,'slcs/*.vrt')) 
        self.slc_datetimes = pd.to_datetime(sorted([ os.path.basename(slc_filename).split('.')[0] for slc_filename in self.slc_filenames]))
        self.slc_dates = [slc_datetime.strftime('%Y%m%d') for slc_datetime in self.slc_datetimes]
        self.Start_time = self.slc_dates[0]
        self.End_time = self.slc_dates[-1]
        self.Start_time_str =  self.Start_time.replace('-','')
        self.End_time_str =  self.End_time.replace('-','')
        self.start_index = self.slc_dates.index(self.Start_time_str)
        self.end_index = self.slc_dates.index(self.End_time_str)+1
        self.nbands = self.end_index - self.start_index
        self.n_dry_bands = int(self.nbands*self.dry_dates_ratio)

    def plot(self):
        """Plotting functionalites
        """

        for file in glob.glob(os.path.join(self.projectDir,'geometry')+'/*npy'):
            temp_data=np.load(file)
        
            if len(temp_data.shape)==2:
                rows=temp_data.shape[0]
                columns=temp_data.shape[1]
                min_dimension=min(rows,columns)
                plot_y_size=2*int(rows/min_dimension)
                plot_x_size=2*int(columns/min_dimension)
                fig = plt.figure(figsize=(plot_x_size,plot_y_size))
                plt.imshow(temp_data)
                plt.colorbar(orientation="horizontal")
                band_name=os.path.basename(file)[:-4]
                plt.title(band_name)
                plt.savefig(os.path.join(self.plot_dir,band_name), dpi=100)
                plt.close()

    def get_dry_SARs(self, lowest_temp_K, tp_quantile):
        """
        Finds dry SAR acquisitions based on meteorological information
        """
        
        self.meteo_df = convert_to_df(self.meteo_file, self.AOI)
        
        self.dry_date_sel, self.dry_dates, self.meteo_sel_df = find_dry_SARs(meteo_df = self.meteo_df,
                                                                          slc_datetimes = self.slc_datetimes,
                                                                          days_back = self.days_back,
                                                                          orbit_time = self.orbit_time,
                                                                          lowest_temp_K = lowest_temp_K,
                                                                          tp_quantile = tp_quantile)
        

    def calc_insar_stack(self):
        """Computes interferometric observables.
        """
        
        self.primary_index = calc_int_amp(input_dir = self.projectDir,
                                         output_dir = self.WorkDir,
                                         primary_date = self.dry_date_sel)
        
        self.lat_data=np.load(os.path.join(self.WorkDir,'geometry/lat.npy'))
        self.lon_data=np.load(os.path.join(self.WorkDir,'geometry/lon.npy'))
        self.incLocal=np.load(os.path.join(self.WorkDir,'geometry/incLocal.npy'))
        self.perp_baseline_stack=np.load(os.path.join(self.WorkDir,'geometry/perp_baseline_stack.npy'))
        self.hgt = np.load(os.path.join(self.WorkDir,'geometry/hgt.npy'))
        self.slc_stack = np.load(os.path.join(self.WorkDir,'coreg_stack/slc_stack.npy'))[self.start_index:self.end_index,:,:]
        self.amp_stack = np.abs(self.slc_stack)
        
    def calc_grid(self, grid_size:int):
        """Calculates the centroids and polygons for each grid cell

        Args:
            grid_size (int): size of grid in meters
        """
        self.grid_size = grid_size
        self.sm_points, self.sm_polygons = create_grid_xy(Outdir = self.SM_dir,
                                                          AOI = self.AOI,
                                                          res = self.grid_size)
        
        assert len(self.sm_points) == len(self.sm_polygons)
        self.n_sm_points = len(self.sm_points)

#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
class SM_point:
    """Soil moisture estimation for a single DS
    """
    def __init__(self, insar4sm_stack:INSAR4SM_stack, sm_ind:int):
        
        # id
        self.WorkDir = insar4sm_stack.WorkDir
        self.id = sm_ind
        
        # space parms
        self.lon = insar4sm_stack.sm_points.iloc[self.id].x
        self.lat = insar4sm_stack.sm_points.iloc[self.id].y
        self.sm_geometry = insar4sm_stack.sm_polygons.iloc[self.id]

        # temporal parms
        self.slc_dates = insar4sm_stack.slc_datetimes
        self.start_index = insar4sm_stack.start_index
        self.end_index = insar4sm_stack.end_index
        self.dry_dates = insar4sm_stack.dry_dates
        
        # DS processing parms
        self.amp_sel = True
        self.ph_keep_percent = 1
        self.save = False
        self.DS_center_sel = True
        self.p_value_thres = 0.1
        self.similarity_perc = 0.5
        self.az_size = 1
        self.rg_size = 1

        # coherence processing parms
        self.n_iters = 500
        self.coh_denoise = False
        self.max_coh_flag = True
        self.simple_exp_model = False
        self.n_dry_bands = insar4sm_stack.n_dry_bands
        self.temp_thres = 0
        
        # inversion parms
        self.opt_parms = {}
        self.opt_parms['ftol'] = 10e-1
        self.opt_parms['eps'] = 0.03
        self.opt_parms['maxiter'] = 25

        self.opt_method = 'SLSQP' # or 'trust-constr'
        self.ph_closure_dist = 6 # distance between sar acquisitions
        self.weight_factor = 1 # the weighting factor of coherence cost
        self.sm_dry_state = 2.8  # in m3/m3
        self.freq_GHz = 5.405
        
        # read soil data
        if type(insar4sm_stack.sand_data) == np.int64:
            self.sand_pct = insar4sm_stack.sand_data
        elif type(insar4sm_stack.sand_data) == str:
            self.sand_pct = get_soilgrids_value(insar4sm_stack.sand_data, self.lon, self.lat)
        else:
            raise("please provide an integer value for sand or a path to soilgrids tiff file (https://soilgrids.org/).")
    
    
        if type(insar4sm_stack.clay_data) == np.int64:
            self.clay_pct = insar4sm_stack.clay_data
        elif type(insar4sm_stack.clay_data) == str:
            self.clay_pct = get_soilgrids_value(insar4sm_stack.clay_data, self.lon, self.lat)
        else:
            raise("please provide an integer value for clay or a path to soilgrids tiff file (https://soilgrids.org/).")

        # interferometric parms
        self.primary_index = insar4sm_stack.primary_index
        self.n_ifg = insar4sm_stack.nbands
        self.n_ds = 1
        self.DS_ind = 0
   
        # intialize datasets for SM_point
        self.bp_DS = np.zeros((self.n_ds, self.n_ifg), dtype=np.float64)
        self.inc_DS = np.zeros((self.n_ds), dtype=np.float64)
        self.ph_DS = np.zeros((self.n_ds,self.n_ifg,self.n_ifg), dtype=np.complex128)
        self.coh_full_DS = np.zeros((self.n_ds,self.n_ifg,self.n_ifg), dtype=np.float64)
        self.coh_DS = np.zeros((self.n_ds), dtype=np.float64)
        self.amp_DS = np.zeros((self.n_ds, self.n_ifg), dtype=np.float64)
        self.xy_DS = np.zeros((self.n_ds, 3), dtype=np.float64)
        self.ij_DS = np.zeros((self.n_ds, 3), dtype=np.int64)
        self.lonlat_DS = np.zeros((self.n_ds, 3), dtype=np.float64)
        self.hgt_DS = np.zeros((self.n_ds), dtype=np.float64)
        
        
    def get_DS_info(self, insar4sm_stack:INSAR4SM_stack):
        """Extracts the SLC SAR pixels that will be used for constructing distributed scatterer (DS).
         Extracts the inteferometric phase and amplitude of DS.

        Args:
            insar4sm_stack (INSAR4SM_stack): Object with data and attributes of InSAR4SM stack
        """
        
        # 1. get pixel coordinates and DS slc/amp values
        self.DS_coords_1, self.DS_coords_2, self.DS_slc_values, self.DS_amp_values, self.geometry_mask = get_DS_pixels(insar4sm_stack.slc_stack,
                                                                                                                       insar4sm_stack.amp_stack,
                                                                                                                       insar4sm_stack.lat_data,
                                                                                                                       insar4sm_stack.lon_data,
                                                                                                                       self.sm_geometry)
        
        # 2. For all combinations of pixels run amplitude statistical test to 
        #    find the similar ones.
        if self.amp_sel:
            if self.DS_center_sel:
                self.Best_pixels_amp = find_best_pixels_amp_fast(self.DS_coords_1,
                                                                self.DS_coords_2,
                                                                self.DS_amp_values,
                                                                insar4sm_stack.amp_stack,
                                                                p_value_thres = self.p_value_thres,
                                                                az_size = self.az_size,
                                                                rg_size = self.rg_size)
            else:
                self.Best_pixels_amp = find_best_pixels_amp(self.DS_coords_1,
                                                                self.DS_coords_2,
                                                                self.DS_amp_values,
                                                                p_value_thres = self.p_value_thres,
                                                                similarity_perc = self.similarity_perc)
        else:
            self.Best_pixels_amp = np.ones((self.DS_coords_1.shape[0]), dtype=np.bool_)
            
        # 3. For all combinations of amplitude similar pixels, calculate the 
        # DS coherence in order to find the ones that yield the highest 
        # temporal coherence.
        self.Best_pixels_ph = find_best_pixels_ph(self.DS_coords_1,
                                                  self.DS_coords_2,
                                                  self.DS_slc_values,
                                                  self.Best_pixels_amp,
                                                  keep_percent = self.ph_keep_percent)
        
        
        self.DS_coords_1_amp_ph = self.DS_coords_1 [self.Best_pixels_amp][self.Best_pixels_ph]
        self.DS_coords_2_amp_ph = self.DS_coords_2 [self.Best_pixels_amp][self.Best_pixels_ph]
        self.DS_slc_values_amp_ph = self.DS_slc_values[:,self.Best_pixels_amp][:,self.Best_pixels_ph]
        self.DS_amp_values_amp_ph = self.DS_amp_values[:,self.Best_pixels_amp][:,self.Best_pixels_ph] 

    def calc_covar_matrix(self):
        """Calculates the covariance and coherence matrix of distributed scatterer (DS)
        """
        
        # 1. Get the covariance and coherence matrix of the selected DS")
    
        #For all pixels
        self.nbands = self.DS_slc_values.shape[0]
        self.weights = np.ones(self.DS_slc_values.shape[1], dtype=np.float32)
        self.DS_Covar0, self.DS_Coh0 = compute_Covar_Coh_weighted(self.DS_slc_values,
                                                                  self.weights,
                                                                  self.nbands)
       
        #Based on amplitude/phase criteria
        self.nbands_ph = self.DS_slc_values_amp_ph.shape[0]
        self.weights = np.ones(self.DS_slc_values_amp_ph.shape[1], dtype=np.float32)
        self.DS_Covar_ph, self.DS_Coh_ph = compute_Covar_Coh_weighted(self.DS_slc_values_amp_ph,
                                                                      self.weights,
                                                                      self.nbands_ph)
                
        # 2. Keep the pixels that yield highest coherence
    
        self.DS_Covar = self.DS_Covar_ph
        self.DS_Coh = self.DS_Coh_ph
        self.DS_amp = np.mean(self.DS_amp_values_amp_ph, axis=1)
        self.DS_coords_1 = self.DS_coords_1_amp_ph
        self.DS_coords_2 = self.DS_coords_2_amp_ph
        
        # check if coords contains only one element
        if (self.DS_coords_1.shape[0] != 0) and (self.DS_coords_2.shape[0] != 0):
            #self.DS_coords_1_flag = np.all(self.DS_coords_1 == self.DS_coords_1[0]) or (not np.any(self.DS_coords_1))
            #self.DS_coords_2_flag = np.all(self.DS_coords_2 == self.DS_coords_2[0]) or (not np.any(self.DS_coords_2))
            self.DS_coords_1_flag = not np.any(self.DS_coords_1)
            self.DS_coords_2_flag = not np.any(self.DS_coords_2)
            
            self.non_coverage = self.DS_coords_1_flag or self.DS_coords_2_flag
            self.non_coverage = False
        else:
            self.non_coverage = True

        self.nbands = self.nbands_ph 

        # 3. calculate mean temporal coherence
        
        self.eigenValues, self.eigenVectors, c, d, e  = zheevr(np.asfortranarray(self.DS_Covar), range='I')
        self.largestEigenVector = self.eigenVectors[:,-1]
          # get the largest Eigenvector
        self.phase0_largestEigenVector=np.angle(self.largestEigenVector[self.primary_index])
        # this is like creating interferograms with primary the first image
        # phase diffs can be from -2pi to 2pi 
        self.phase_diffs=np.angle(self.largestEigenVector)-self.phase0_largestEigenVector     
        # by constructing the complex number with the phase difference is like
        # rewrapping the phase difference from -pi to pi    
        self.DS_slc_phase=np.exp(1J*self.phase_diffs)
        # normalizing
        self.DS_slc_phase = self.DS_slc_phase/np.abs(self.DS_slc_phase)    
        
        self.number_ints=int((self.nbands*(self.nbands-1))/2)
        self.temporal_coherence=np.zeros(self.number_ints, dtype=np.complex128) 
        int_index=0
        for image_i_index in range(0, self.nbands):
            for image_j_index in range(image_i_index+1, self.nbands):    
                self.evd_phase = np.angle(self.DS_slc_phase[image_i_index])-np.angle(self.DS_slc_phase[image_j_index])
                self.initial_phase = np.angle(self.DS_Covar[image_i_index,image_j_index])
                self.temporal_coherence[int_index] = np.exp(1j*(self.initial_phase-self.evd_phase))
                int_index+=1
        
        self.temporal_coherence_real= np.real(self.temporal_coherence)
        self.mean_temporal_coherence=np.mean(self.temporal_coherence_real)
    
    
        self.ph_DS[self.DS_ind,:,:] = self.DS_Covar
        self.coh_full_DS[self.DS_ind,:,:] = self.DS_Coh
        self.coh_DS[self.DS_ind] = self.mean_temporal_coherence
        self.amp_DS[self.DS_ind,:] = self.DS_amp

    def get_DS_geometry(self, insar4sm_stack:INSAR4SM_stack):
        """Extracts geometrical information for DS

        Args:
            insar4sm_stack (INSAR4SM_stack): Object with data and attributes of InSAR4SM stack
        """

        #ij 
        mean_i = int(np.nanmean(self.DS_coords_1))
        mean_j = int(np.nanmean(self.DS_coords_2))
        self.ij_DS[self.DS_ind,0] = self.DS_ind
        self.ij_DS[self.DS_ind,1] = mean_i
        self.ij_DS[self.DS_ind,2] = mean_j
        
        #lonlat
        self.lonlat_DS[self.DS_ind,0] = self.DS_ind
        self.lonlat_DS[self.DS_ind,1] = insar4sm_stack.lon_data[mean_i,mean_j]
        self.lonlat_DS[self.DS_ind,2] = insar4sm_stack.lat_data[mean_i,mean_j]
        
        #xy
        Xs, Ys = llh2xy(insar4sm_stack.lat_data,insar4sm_stack.lon_data)
        self.xy_DS[self.DS_ind,0] = self.DS_ind
        self.xy_DS[self.DS_ind,1] = Ys[mean_i,mean_j]
        self.xy_DS[self.DS_ind,2] = Xs[mean_i,mean_j]
        
        #incidence angle information
        self.inc_DS[self.DS_ind] = insar4sm_stack.incLocal[mean_i,mean_j]
        
        #baseline information
        self.DS_bperp = insar4sm_stack.perp_baseline_stack[self.start_index:self.end_index,mean_i,mean_j]
        
        # Latest image is selected as primary in order to calculate DS perpendicular
        # baseline information. We subtract the first image to re-reference
        self.bp_DS[self.DS_ind,:] = self.DS_bperp - self.DS_bperp[self.primary_index]
        
        # height
        self.hgt_DS[self.DS_ind] = insar4sm_stack.hgt[mean_i,mean_j]
        
    def calc_driest_date(self):
        """ Selection of the SAR acquisition that yields the highest coherence among the images that are characterized "dry". 
        Assuming that the majority of "dry" images correspond to dry condition, we ensure that the selected SAR acquisition is related to dry conditions.
        """
        
        self.dry_inds = [ np.where(self.slc_dates == dry_date)[0][0] for dry_date in self.dry_dates]
        
        # convert diagonal elements into nan values
        coh_nan = self.coh_full_DS[self.DS_ind,:,:].copy()
        coh_nan[np.diag_indices(coh_nan.shape[0])] = np.nan
        
        # subset only the dry dates from the coherence matrix
        coh_temp_dry = coh_nan[self.dry_inds,:][:,self.dry_inds]
        
        #calculate mean over columns
        coh_temp_dry_mean = np.nanmean(coh_temp_dry, axis=1)
        
        # get the driest date
        driest_ind = np.argmax(coh_temp_dry_mean)
        self.driest_date = self.dry_dates[driest_ind]

    def calc_sm_sorting(self):
        """Calculates ascending sorting of SAR acuqisition based on their soil moisture
        """
        
        self.best_sorting = find_sm_sorting(self.slc_dates,
                                            self.coh_full_DS,
                                            self.amp_DS,
                                            self.DS_ind,
                                            self.start_index,
                                            self.end_index,
                                            self.driest_date,
                                            self.nbands,
                                            self.n_iters,
                                            self.coh_denoise)
        
    def calc_sm_coherence(self):
        """Calculates coherence information related to surface soil moisture variations
        """
        self.coh_sm, self.coh_sm_mean, self.coh_sm_std = calc_SM_coherences(self.slc_dates,
                                                                            self.best_sorting,
                                                                            self.nbands,
                                                                            self.n_ds,
                                                                            self.DS_ind,
                                                                            self.coh_full_DS,
                                                                            self.start_index,
                                                                            self.end_index,
                                                                            self.n_dry_bands,
                                                                            self.max_coh_flag,
                                                                            self.simple_exp_model)
        
    def calc_sm_index(self):
        """Calculates soil moisture (proxy) index information
        """
        self.SM_index, self.SM0, self.dry_bands_inds, self.coh_sm_filled = calc_burgi_sm_index(self.coh_sm,
                                                                                               self.best_sorting,
                                                                                               self.n_dry_bands,
                                                                                               self.sm_dry_state,
                                                                                               self.DS_ind,
                                                                                               self.start_index,
                                                                                               self.end_index,
                                                                                               self.temp_thres)
        
    def inversion(self):
        """Invertion of De Zan`s model to estimation soil moisture.
        
        .. References:
                De Zan, F., Parizzi, A., Prats-Iraola, P., Lopez-Dekker, P., 2014. A SAR Interferometric Model for Soil Moisture. IEEE Trans. Geosci. Remote Sens. 52, 418-425. https://doi.org/10.1109/TGRS.2013.2241069   
        """
        self.sm_inverted = invert_sm(self.ph_DS,
                                     self.coh_sm_filled,
                                     self.SM0,
                                     self.dry_bands_inds,
                                     self.slc_dates,
                                     self.DS_ind,
                                     self.driest_date,
                                     self.inc_DS,
                                     self.start_index,
                                     self.end_index,
                                     self.nbands,
                                     self.opt_method,
                                     self.opt_parms,
                                     self.ph_closure_dist,
                                     self.weight_factor,
                                     self.sm_dry_state,
                                     self.freq_GHz,
                                     self.clay_pct,
                                     self.sand_pct)

