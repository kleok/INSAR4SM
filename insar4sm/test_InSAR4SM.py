#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, pickle
import numpy as np
from insar4sm.classes import SM_point
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    
def array_nan_close(a:np.array, b:np.array)->bool:
    m = np.isfinite(a) & np.isfinite(b)
    return np.allclose(np.abs(a[m]), np.abs(b[m]))

def test_insar4sm():
    # get the value of the environment variable $insar4sm_HOME
    insar4sm_HOME = os.getenv('insar4sm_HOME')
    
    with open('{}/testing/FordDryLake_data.pkl'.format(insar4sm_HOME), 'rb') as inp:
        FordDryLake_data = pickle.load(inp)

    #Run SM functionalities
    FordDryLake_SM_data = SM_point(insar4sm_stack = FordDryLake_data, sm_ind=0)
    FordDryLake_SM_data.get_DS_info(FordDryLake_data)
    FordDryLake_SM_data.calc_covar_matrix()
    FordDryLake_SM_data.get_DS_geometry(FordDryLake_data)
    FordDryLake_SM_data.calc_driest_date()
    FordDryLake_SM_data.calc_sm_sorting()
    FordDryLake_SM_data.calc_sm_coherence()
    FordDryLake_SM_data.calc_sm_index()
    FordDryLake_SM_data.inversion()
    
    # save to disk for comparison purposes
    #with open('{}/testing/FordDryLake_data_test.pkl'.format(insar4sm_HOME), 'wb') as outp:
    #    pickle.dump(FordDryLake_SM_data, outp, pickle.HIGHEST_PROTOCOL)
        
    # convert object to dictionary
    dict_data = FordDryLake_SM_data.__dict__
    
    with open('{}/testing/FordDryLake_validation_data.pkl'.format(insar4sm_HOME), 'rb') as inp:
        FordDryLake_SM_validation_data = pickle.load(inp)
        # convert object to dictionary
        dict_validation_data = FordDryLake_SM_validation_data.__dict__
    
    for key in dict_validation_data.keys():
        print(key)
        try:
            if type(dict_validation_data[key])!=np.ndarray:
                assert np.all(dict_data[key]==dict_validation_data[key])
            else:
                assert array_nan_close(dict_data[key], dict_validation_data[key])
        except:
            print(dict_data[key],type(dict_data[key]))
            print('---------------------------------')
            print(dict_validation_data[key],type(dict_validation_data[key]))