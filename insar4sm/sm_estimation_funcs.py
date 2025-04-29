from insar4sm.classes import INSAR4SM_stack, SM_point
import numpy as np
import pandas as pd

def sm_estimation(stack:INSAR4SM_stack, sm_ind:int, dry_date_manual_flag: bool, dry_date: str, DS_flag: bool = True)->np.array:
    """Estimates soil moisture using insar4sm functionalities

    Args:
        stack (INSAR4SM_stack): the object of the INSAR4SM_stack class
        sm_ind (int): index of soil moisture estimation point
        DS_flag (bool): flag that determines if distributed scatterers will be computed.

    Returns:
        np.array: soil moisture estimations from inversion
    """
    sm_point_ts = SM_point(stack, sm_ind)
    sm_point_ts.amp_sel = DS_flag
    sm_point_ts.get_DS_info(stack)
    sm_point_ts.calc_covar_matrix()
    if sm_point_ts.non_coverage or np.all(sm_point_ts.amp_DS==0):
        return np.full(sm_point_ts.n_ifg, np.nan)
    else:
        sm_point_ts.get_DS_geometry(stack)
        sm_point_ts.calc_driest_date()
        if dry_date_manual_flag: sm_point_ts.driest_date = pd.to_datetime(dry_date)
        sm_point_ts.calc_sm_sorting()
        sm_point_ts.calc_sm_coherence()
        sm_point_ts.calc_sm_index()
        sm_point_ts.inversion()
        return sm_point_ts.sm_inverted
    