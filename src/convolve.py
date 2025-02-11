# SPDX-License-Identifier: BSD-2-Clause
# Authors: Vincent Eberle, Philipp Frank

# Copyright(C) 2024 Max-Planck-Society

# %

from functools import reduce

import jax
import jax.lax
import jax.numpy as jnp
import numpy as np

from .data import Domain


def _bilinear_weights(shape):
    """Build bilinear interpolation kernel."""
    if shape[0]/2 != int(shape[0]/2):
        raise ValueError("this should happen")
    # FIXME get better weights for non by 2 divisible numbers
    a = np.linspace(0, 1, int(shape[0]/2), dtype="float64")
    b = np.concatenate([a, np.flip(a)])
    return np.outer(b, b)


def _prep_psfs_for_linpatch_convolve(psfs, domain, n_patches_per_axis,
                                     margin, normalize=True):
    margins = [[margin, margin],]*2
    kernelcut_x = int((domain.shape[-2]*(1 - 2 / n_patches_per_axis))//2) #- margin)
    kernelcut_y = int((domain.shape[-1]*(1 - 2 / n_patches_per_axis))//2) #- margin)

    roll_kernel = jnp.fft.fftshift(psfs, axes=(-2, -1))
    cut_kernel = roll_kernel[..., kernelcut_x:-kernelcut_x, kernelcut_y:-kernelcut_y]

    # FIXME Temp Fix for weird psfs/ We could / should leave it in.
    padding_for_extradims_width = [[0, 0],]*(len(cut_kernel.shape) - 2)
    pad_width_kernel = padding_for_extradims_width + margins

    pkernel = jnp.pad(cut_kernel,
                      pad_width=pad_width_kernel,
                      mode="constant",
                      constant_values=0)
    res = jnp.fft.ifftshift(pkernel, axes=(-2, -1))
    if normalize:
        return _naive_normalize_psf(res, domain)
    return res


def _naive_normalize_psf(psf, domain):
    summed = psf.sum((-2, -1))
    dvol = domain.distances[-2]*domain.distances[-1]
    integral = summed * np.array(dvol)
    integral = integral[..., np.newaxis, np.newaxis]
    normed_psf = psf * integral**-1
    return normed_psf


def slice_patches(x, shape, n_patches_per_axis, additional_margin):
    """Slice object into equal sized patches.

    Parameters:
    -----------
    x: input array
    shape: shape of the input array
    n_patches_per_axis: int
        number of patches after the slicing
    overlap_margin: int
        additional margin at the borders
    """
    dr = additional_margin
    dx = int((shape[-2] - 2 * dr) / n_patches_per_axis)
    dy = int((shape[-1] - 2 * dr) / n_patches_per_axis)

    pad_extra_dims = [[0, 0],]*(len(x.shape) - 2)
    pad_spaces = [[dx//2, ] * 2, [dy//2, ] * 2]
    pad_width = pad_extra_dims + pad_spaces

    padded_x = jnp.pad(x, pad_width=pad_width,
                       mode="constant", constant_values=0)
    trailing_pos = (0,)*(len(padded_x.shape)-2)
    trailing_slice_sizes = padded_x.shape[:-2]

    def slicer(x_pos, y_pos):
        return jax.lax.dynamic_slice(padded_x, start_indices=trailing_pos+(x_pos, y_pos),
                                     slice_sizes=trailing_slice_sizes+(2*dx + 2*dr, 2*dy + 2*dr))

    ids = (jnp.arange(n_patches_per_axis)*dx, jnp.arange(n_patches_per_axis)*dy)

    ndx = jnp.meshgrid(*ids, indexing="xy")
    f = jax.vmap(slicer, in_axes=(0, 0), out_axes=(0))
    return f(*(nn.flatten() for nn in ndx))


def linpatch_convolve(x, domain, kernel, n_patches_per_axis,
                      margin):
    """Functional version of linear patching convolution.

    Parameters:
    -----------
    x : input array
    domain: domain (shape, distances) of input array
    kernel: np.array
        Array containing the different kernels for the inhomogeneos convolution
    n_patches_per_axis: int
        Number of patches
    additional_margin: int
        Size of the margin. Number of pixels on one boarder.

    """
    if not isinstance(domain, Domain):
        raise ValueError("domain has to be an instance of jubik.Domain")

    spatial_shape = [domain.shape[-2], domain.shape[-1]]
    slices = slice_patches(x, domain.shape, n_patches_per_axis,
                           additional_margin=0)

    weight_shp = (2*(domain.shape[-2] / n_patches_per_axis),
                  2*(domain.shape[-1] / n_patches_per_axis))
    weights = _bilinear_weights(weight_shp)

    weighted_slices = weights * slices

    # TODO with less dependency? Helper func from domain /n patches per axis and margin
    padding_for_extradims_width = [[0, 0],]*(len(weighted_slices.shape) - 2)
    margins = [[margin, margin],]*2
    pad_width = padding_for_extradims_width + margins
    # END

    padded = jnp.pad(weighted_slices,
                     pad_width=pad_width,
                     mode="constant", constant_values=0)

    ndom = Domain((1, *domain.shape), (None, *domain.distances))
    convolved = convolve(kernel,
                         padded,
                         ndom,
                         axes=(-2, -1))
    # TODO Check these two also with speed!
    # convolved = jax.scipy.signal.fftconvolve(kernel, padded, mode="same", axes=(-2,-1))
    remaining_shape = list(domain.shape[:-2])
    padded_shape = remaining_shape + [ii+2*margin for ii in spatial_shape]

    def patch_w_margin(array):
        return slice_patches(array, padded_shape,
                             n_patches_per_axis, margin)

    primal = np.empty(padded_shape)
    overlap_add = jax.linear_transpose(patch_w_margin, primal)
    padded_res = overlap_add(convolved)[0]
    # TODO This slicing is not good
    res = padded_res[..., margin:-margin, margin:-margin]
    return res


def convolve(x, y, domain, axes):
    """Perform an FFT convolution.
    #TODO implement jnp.convolve wrapper

    Parameters:
    -----------
    x: numpy.array
        input array
    y: numpy.array
        kernel array
    domain: Domain(NamedTuple)
        containing the information about distances and shape of the domain.
    axes: tuple
        axes for the convolution
    """
    dlist = [domain.distances[i] for i in axes]
    dvol = float(reduce(lambda a, b: a*b, dlist))

    hx = jnp.fft.fftn(x, axes=axes)
    hy = jnp.fft.fftn(y, axes=axes)
    if len(y.shape) > len(x.shape):
        print("kernel_shape:", x.shape)
        print("signal_shape:", y.shape)
        print("Dimension Inconsistency. Broadcasting PSFs")
        prod = hx[..., np.newaxis, :, :]*hy
    else:
        prod = hx*hy
    res = jnp.fft.ifftn(prod, axes=axes)
    res = dvol*res
    return res.real


def integrate(x, domain, axes):
    sumation = np.sum(x, axis=tuple(axes))
    dlist = [domain.distances[i] for i in axes]
    dvol = float(reduce(lambda a, b: a*b, dlist))
    return dvol*sumation

