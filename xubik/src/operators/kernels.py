import numpy as np
import nifty8 as ift


def gauss(x, y, sig):
    const = 1 / (np.sqrt(2 * np.pi * sig ** 2))
    r = np.sqrt(x ** 2 + y ** 2)
    f = const * np.exp(-r ** 2 / (2 * sig ** 2))
    return f


def get_gaussian_kernel(width, domain):
    x = y = np.linspace(-width, width, domain.shape[1])
    xv, yv = np.meshgrid(x, y)
    kern = gauss(xv, yv, 1)
    kern = np.fft.fftshift(kern)
    kern = ift.makeField(domain[1], kern)
    kern = kern * (kern.integrate().val) ** -1
    explode_pad = ift.ContractionOperator(domain, spaces=0)
    res = explode_pad.adjoint(kern)
    return res
