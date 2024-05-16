# Basic MF Plot

from functools import reduce
import numpy as np

from astropy.io import fits
import matplotlib.pyplot as plt

import jubik0 as ju


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


def _smooth(sig, x):
    domain = ju.Domain(x.shape, np.ones([3]))
    gauss_domain = ju.Domain(x.shape[1:], np.ones([2]))

    smoothing_kernel = get_gaussian_kernel(gauss_domain, sig)
    smoothing_kernel = smoothing_kernel[np.newaxis, ...]
    smooth_data = ju.jifty_convolve(x, smoothing_kernel, domain, [1, 2])
    return np.array(smooth_data)


def _clip(x, sat_min, sat_max):
    clipped = np.zeros(x.shape)
    print("Change the Saturation")
    for i in range(3):
        clipped[i] = np.clip(x[i], a_min=sat_min[i], a_max=sat_max[i])
        clipped[i] = clipped[i]-sat_min[i]
    return clipped


def _non_zero_log(x):
    x[x >= 1] = np.log(x[x >= 1])
    return x


def _norm_rgb_plot(x):
    plot_data = np.zeros(x.shape)
    maxim = np.array([np.max(x[:, :, i]) for i in range(3)])
    # norm on RGB to 0-1
    for i in range(3):
        plot_data[:, :, i] = (x[:, :, i] / maxim[i])
    return plot_data


def plot_rgb(x, name, sat_min, sat_max, sigma=None,log=False):
    """
    x: array with shape (RGB, Space, Space)
    name: str, name of the plot
    sigma: float or None, if None, no smoothing
    linsat: boolean, if it should be saturated
    log: boolean
    """
    if sigma is not None:
        x = _smooth(sigma, x)
    if sat_min and sat_max is not None:
        x = _clip(x, sat_min, sat_max)
    if log:
        x = _non_zero_log(x)
    x = np.moveaxis(x, 0, -1)
    plot_data = _norm_rgb_plot(x)
    plt.imshow(plot_data, origin="lower")
    plt.savefig(name + ".png", dpi=500)
    plt.close()

# Prep Data
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

# Plotting Config
plot_name = "prototyp"
Log = False
sigma = 0.02
max_percent = [1.412e-3, 9.6e-4, 3.00e-3]
Sat_max = [max_percent[i] * data_arr[i].max() for i in range(3)]
Sat_min = [2, 0, 2]

plot_rgb(data_arr, plot_name, Sat_min, Sat_max, sigma, log=Log)
