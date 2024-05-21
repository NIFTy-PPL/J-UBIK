#!/usr/bin/env python3
import numpy as np
import jubik0 as ju
from astropy.io import fits

fbase = "data/LMC_SN1987A/processed/"
tms = ["tm1",  "tm3", "tm4", "tm6", "tm2"]

nrg = ["_pm00_700161_020_data_emin0.2_emax1.0.fits",
       "_pm00_700161_020_data_emin1.0_emax2.0.fits",
       "_pm00_700161_020_data_emin2.0_emax4.5.fits"]

fpath_list = [[fbase + tm + energy for tm in tms] for energy in nrg]

data_list = []
for fpathl in fpath_list:
    data_list_i = []
    for fpath in fpathl:
        with fits.open(fpath) as hdul:
            data = hdul[0].data
            data_list_i.append(data)
    data_arr_i = np.array(data_list_i)
    data_list.append(data_arr_i)

data_arr = np.array(data_list)
data_arr = data_arr.sum(1)
data_arr = data_arr*1e-10

# Plotting Config
plot_name = "prototyp"
Log = True
sigma = 0.02
max_percent = [1.412e-3, 9.6e-4, 3.00e-3]
Sat_max = [max_percent[i] * data_arr[i].max() for i in range(3)]
Sat_min = [2, 0, 2]

ju.plot_rgb(data_arr, plot_name, sat_min=None, sat_max=None, sigma=None, log=True)
