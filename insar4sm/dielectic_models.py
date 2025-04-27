import numpy as np

def hallikainen_1985_calc(clay_pct:float, sand_pct:float, mv_pct:float, freq_GHz:float)->np.complex128:
    """Computes the complex dielectric constant based on the empirical relations from Hallikainen et al., 1985.

    Args:
        clay_pct (float): clay mass fraction (0-100)%
        sand_pct (float): sand mass fraction (0-100)%
        mv_pct (float): volumetric moisture content (0-100)%
        freq_GHz (float): frequency (GHz)

    Returns:
        np.complex: complex dielectric constant of soil/water mixture

    ..References:
          Hallikainen, M., Ulaby, F., Dobson, M., El-rayes, M., Wu, L., 1985. Microwave Dielectric Behavior of Wet Soil-390 Part 1: Empirical Models and Experimental Observations. IEEE Trans. Geosci. Remote Sens. GE-23, 25-34. 391 https://doi.org/10.1109/TGRS.1985.289497
    """
    mv_dec = mv_pct/100

    x = np.array([1, sand_pct, clay_pct])
    F = np.array([1.4, 4, 6, 8, 10, 12, 14, 16, 18])

    a_r = np.array([[+2.862, -0.012, +0.001],
                    [+2.927, -0.012, -0.001],
                    [+1.993, +0.002, +0.015],
                    [+1.997, +0.002, +0.018],
                    [+2.502, -0.003, -0.003],
                    [+2.200, -0.001, +0.012],
                    [+2.301, +0.001, +0.009],
                    [+2.237, +0.002, +0.009],
                    [+1.912, +0.007, +0.021]])

    a_i = np.array([[+0.356, -0.003, -0.008],
                    [+0.004, +0.001, +0.002],
                    [-0.123, +0.002, +0.003],
                    [-0.201, +0.003, +0.003],
                    [-0.070, +0.000, +0.001],
                    [-0.142, +0.001, +0.003],
                    [-0.096, +0.001, +0.002],
                    [-0.027, -0.001, +0.003],
                    [-0.071, +0.000, +0.003]])

    b_r = np.array([[+3.803,  +0.462, -0.341],
                    [+5.505,  +0.371, +0.062],
                    [+38.086, -0.176, -0.633],
                    [+25.579, -0.017, -0.412],
                    [+10.101, +0.221, -0.004],
                    [+26.473, +0.013, -0.523],
                    [+17.918, +0.084, -0.282],
                    [+15.505, +0.076, -0.217],
                    [+29.123, -0.190, -0.545]])

    b_i = np.array([[+5.507, +0.044, -0.002],
                    [+0.951, +0.005, -0.010],
                    [+7.502, -0.058, -0.116],
                    [+11.266, -0.085, -0.155],
                    [+6.620, +0.015, -0.081],
                    [+11.868, -0.059, -0.225],
                    [+8.583, -0.005, -0.153],
                    [+6.179, +0.074, -0.086],
                    [+6.938, +0.029, -0.128]])

    c_r = np.array([[+119.006, -0.500, +0.633],
                    [+114.826, -0.389, -0.547],
                    [+10.720, +1.256, +1.522],
                    [+39.793, +0.723, +0.941],
                    [+77.482, -0.061, -0.135],
                    [+34.333, +0.284, +1.062],
                    [+50.149, +0.012, +0.387],
                    [+48.260, +0.168, +0.289],
                    [+6.960, +0.822, +1.195]])

    c_i = np.array([[+17.753, -0.313, +0.206],
                    [+16.759, +0.192, +0.290],
                    [+2.942, +0.452, +0.543],
                    [+0.194, +0.584, +0.581],
                    [+21.578, +0.293, +0.332],
                    [+7.817, +0.570, +0.801],
                    [+28.707, +0.297, +0.357],
                    [+34.126, +0.143, +0.206],
                    [+29.945, +0.275, +0.377]])

    if type(mv_dec) == np.ndarray:
        
        ones = np.ones((mv_dec.shape[0], F.shape[0]))
        E_r_arr = (ones*(a_r @ x)).T +  (ones*(b_r @ x)).T * mv_dec +  (ones*(c_r @ x)).T * mv_dec ** 2
        E_i_arr = (ones*(a_i @ x)).T +  (ones*(b_i @ x)).T * mv_dec +  (ones*(c_i @ x)).T * mv_dec ** 2
        E_r_arr = E_r_arr.T
        E_i_arr = E_i_arr.T
        e_r_arr = np.array([np.interp(freq_GHz, F, E_r) for E_r in E_r_arr])
        e_i_arr = np.array([np.interp(freq_GHz, F, E_i) for E_i in E_i_arr])
        return e_r_arr - 1j * e_i_arr

    else:
        E_r = (a_r @ x) +  (b_r @ x) * mv_dec +  (c_r @ x) * mv_dec ** 2
        E_i = (a_i @ x) +  (b_i @ x) * mv_dec +  (c_i @ x) * mv_dec ** 2

        e_r = np.interp(freq_GHz, F, E_r)
        e_i = np.interp(freq_GHz, F, E_i)
    return e_r - 1j * e_i