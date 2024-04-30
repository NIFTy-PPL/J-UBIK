#!/usr/bin/env python3

from functools import reduce
import jax.numpy as jnp
import matplotlib.pyplot as plt
import scipy
import jubik0 as ju

# jax to float 64?

def get_gaussian_kernel(domain, sigma):
    """"2D Gaussian kernel for fft convolution"""
    border = (shape* distances // 2)
    x = jnp.linspace(-border[0], border[0], domain.shape[0])
    y = jnp.linspace(-border[1], border[1], domain.shape[1])
    xv, yv = jnp.meshgrid(x, y)
    kern = ju.operators.convolve_utils.gauss(xv, yv, sigma)
    kern = jnp.fft.fftshift(kern)
    dvol = reduce(lambda a,b: a*b, distances)
    normalization = kern.sum() * dvol
    kern = kern * normalization ** -1
    return kern.T

def naive_deconv(deg, psf, domain, axes, eps=None):
    """Deconvolve degraded images in a naive approach.

    Paramters:
    ---------
    deg: array, smeared out image
    psf: array, point spread function, same size as deg
    eps: additive epsilon, needed to prevent division by zero

    returns:
    -------
    recovered image
    """
    # FIXME weird integration, think about meaning
    dlist = [domain.distances[i] for i in axes]
    dvol = float(reduce(lambda a, b: a*b, dlist))
    deg_k = jnp.fft.fftn(deg, axes=axes)
    # normalization = (1/psf).sum() * dvol
    # psf = psf * normalization**-1
    psf_k = jnp.fft.fftn(psf, axes=axes) + eps
    res = jnp.fft.ifftn(deg_k / psf_k, axes=axes)
    res = res * dvol**-1
    return res.real

def RL_deconv(data, kernel, domain, n_it, init=None):
    kernel_T = jnp.flip(jnp.flip(kernel, axis=0), axis=1)
    if init is not None:
        s = init
    else:
        s = data
    ones = jnp.ones(data.shape)
    for i in range(n_it):
        t1 = ju.jifty_convolve(s, kernel, domain, axes=[0, 1])
        frac = data / t1
        nom = s * ju.jifty_convolve(frac, kernel_T, domain, axes=[0, 1])
        denom = ju.jifty_convolve(ones, kernel_T, domain, axes=[0, 1])
        s = nom/denom
    return s


# Load Data
racoon = scipy.datasets.face()[:, :, 0]

# Shapes and Distances, here trivial
shape = jnp.array(racoon.shape)
distances = jnp.array([1, 1])
domain = ju.Domain(shape, distances)

# get PSF
sigma = 10
gauss = get_gaussian_kernel(domain, sigma)

# Convolve
convolved = ju.jifty_convolve(racoon, gauss, domain, [0, 1])
exit()
# Deconvolve
naive_res = naive_deconv(convolved, gauss, domain, [0, 1], eps=1e-14)
ones = jnp.ones(racoon.shape)
RL_res = RL_deconv(convolved, gauss, domain, 400, init=ones)

# Plot
fig, ax = plt.subplots(3, 3, figsize=(10,10))

map00 = ax[0, 0].imshow(racoon)
ax[0, 0].set_title("Image")
fig.colorbar(map00, ax=ax[0, 0])

map10 = ax[1, 0].imshow(convolved)
ax[1, 0].set_title("Convolved")
fig.colorbar(map10, ax=ax[1, 0])

map20 = ax[2, 0].imshow(jnp.fft.fftshift(gauss))
ax[2, 0].set_title("Gaussian PSF")
fig.colorbar(map20, ax=ax[2, 0])

map01 = ax[0, 1].imshow(naive_res)
ax[0, 1].set_title("Naive Deconvolution")
fig.colorbar(map01, ax=ax[0, 1])

map11 = ax[1, 1].imshow(racoon-naive_res)
ax[1, 1].set_title("GT-Naive Deconv")
fig.colorbar(map11, ax=ax[1, 1])

map02 = ax[0, 2].imshow(RL_res)
ax[0, 2].set_title("RL Deconvolution")
fig.colorbar(map02, ax=ax[0, 2])

map12 = ax[1, 2].imshow(racoon-RL_res)
ax[1, 2].set_title("GT-RL")
fig.colorbar(map12, ax=ax[1, 2])

fig.tight_layout()
fig.savefig(fname="DeconvolutionMethods.png", dpi=600)

