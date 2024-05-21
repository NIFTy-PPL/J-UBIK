# Basic MF Plot

from functools import reduce
import numpy as np

import matplotlib.pyplot as plt

from ..operators.jifty_convolution_operators import jifty_convolve
from ..operators.convolve_utils import gauss
from .data import Domain


def get_gaussian_kernel(domain, sigma):
    """"2D Gaussian kernel for fft convolution."""
    border = (domain.shape * domain.distances // 2)
    x = np.linspace(-border[0], border[0], domain.shape[0])
    y = np.linspace(-border[1], border[1], domain.shape[1])
    xv, yv = np.meshgrid(x, y)
    kern = gauss(xv, yv, sigma)
    kern = np.fft.fftshift(kern)
    dvol = reduce(lambda a, b: a*b, domain.distances)
    normalization = kern.sum() * dvol
    kern = kern * normalization ** -1
    return kern.T


def _smooth(sig, x):
    domain = Domain(x.shape, np.ones([3]))
    gauss_domain = Domain(x.shape[1:], np.ones([2]))

    smoothing_kernel = get_gaussian_kernel(gauss_domain, sig)
    smoothing_kernel = smoothing_kernel[np.newaxis, ...]
    smooth_data = jifty_convolve(x, smoothing_kernel, domain, [1, 2])
    return np.array(smooth_data)


def _clip(x, sat_min, sat_max):
    clipped = np.zeros(x.shape)
    print("Change the Saturation")
    for i in range(3):
        clipped[i] = np.clip(x[i], a_min=sat_min[i], a_max=sat_max[i])
        clipped[i] = clipped[i]-sat_min[i]
    return clipped


def _non_zero_log(x):
    x_arr = np.array(x)
    log_x = np.zeros(x_arr.shape)
    log_x[x_arr>0] = np.log(x_arr[x_arr>0])
    return log_x


def _norm_rgb_plot(x):
    plot_data = np.zeros(x.shape)
    x = np.array(x)
    # minim = np.array([np.min(x[:, :, i] for i in range(3))])
    # maxim = np.array([np.max(x[:, :, i]) for i in range(3)])
    # norm on RGB to 0-1
    for i in range(3):
        a = x[:, :, i]
        minim = a[a!=0].min()
        maxim = a[a!=0].max()
        a[a != 0] = (a[a != 0] - minim) / (maxim - minim)
        plot_data[:, :, i] = a
    return plot_data


def plot_rgb(x, name, sat_min=[0, 0, 0], sat_max=[1, 1, 1], sigma=None, log=False):
    """Routine for plotting RGB images.

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
