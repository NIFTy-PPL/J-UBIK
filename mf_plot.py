# Basic MF Plot

import jubik0 as ju
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import numpy as np
from functools import reduce


def get_gaussian_kernel(domain, sigma):
    """"2D Gaussian kernel for fft convolution"""
    border = (domain.shape * domain.distances // 2)
    x = np.linspace(-border[0], border[0], domain.shape[0])
    y = np.linspace(-border[1], border[1], domain.shape[1])
    xv, yv = np.meshgrid(x, y)
    kern = ju.operators.convolve_utils.gauss(xv, yv, sigma)
    kern = np.fft.fftshift(kern)
    dvol = reduce(lambda a, b: a*b, domain.distances)
    normalization = kern.sum() * dvol
    kern = kern * normalization ** -1
    return kern.T


Log = False
smooth = False
# Sat_max = [150, 100, 40] #Pretty
Sat_max = [60, 50, 27]
Sat_min = [2, 2, 2]
fbase = "data/LMC_SN1987A/processed/"
tms = ["tm1", "tm2", "tm3", "tm4", "tm6"]

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

sigma = 1
domain = ju.Domain(np.array([3, 512, 512]), np.array([1, 7.03125, 7.03125]))
gauss_domain = ju.Domain(np.array([512, 512]), np.array([7.03125, 7.03125]))

data_arr = np.array(data_list)
data_arr = data_arr.sum(1)

if smooth:
    smoothing_kernel = get_gaussian_kernel(gauss_domain, sigma)
    smooth_data = ju.jifty_convolve(data_arr, smoothing_kernel, domain, [1, 2])
    smooth_data = np.array(smooth_data)
    data_arr = smooth_data

# Sat
if Sat_max is not None:
    clipped = np.zeros(data_arr.shape)
    print("Change the Saturation")
    for i in range(3):
        clipped[i] = np.clip(data_arr[i], a_min=Sat_min[i], a_max=Sat_max[i])
        clipped[i] = clipped[i]-Sat_min[i]
    data_arr = clipped

if Log:
    data_arr[data_arr > 0] = np.log(data_arr[data_arr > 0])

data_arr = np.moveaxis(data_arr, 0, -1)

maxim = np.array([np.max(data_arr[:, :, i]) for i in range(3)])

plot_data = np.zeros(data_arr.shape)

for i in range(3):
    plot_data[:, :, i] = (data_arr[:, :, i] / maxim[i])

plt.imshow(plot_data, origin="lower", norm=LogNorm())
plt.savefig("prototyp.png", dpi=500)
plt.close()
