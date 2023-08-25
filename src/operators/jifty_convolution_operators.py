import jax
import jax.lax
import jax.numpy as jnp

import numpy as np
from functools import reduce

from ..library.data import Domain


def _bilinear_weights(shape):
    """Build bilinear interpolation kernel."""
    if shape[0]/2 != int(shape[0]/2):
        raise ValueError("this should happen")
    # FIXME get better weights for non by 2 divisible numbers
    a = np.linspace(0, 1, int(shape[0]/2), dtype="float64")
    b = np.concatenate([a, jnp.flip(a)])
    return np.outer(b, b)


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
    dx = int((shape[0] - 2 * dr) / n_patches_per_axis)
    dy = int((shape[1] - 2 * dr) / n_patches_per_axis)
    padded_x = jnp.pad(x, pad_width=((dx//2, ) * 2, (dy//2, ) * 2),
                       mode="constant", constant_values=0)

    def slicer(x_pos, y_pos):
        return jax.lax.dynamic_slice(padded_x, start_indices=(x_pos, y_pos),
                                     slice_sizes=(2*dx + 2*dr, 2*dy + 2*dr))

    ids = (np.arange(n_patches_per_axis)*dx, np.arange(n_patches_per_axis)*dy)

    ndx = np.meshgrid(*ids, indexing="xy")
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
    shape = domain.shape
    slices = slice_patches(x, shape, n_patches_per_axis,
                           additional_margin=0)
    weights = _bilinear_weights(slices[0].shape)
    weighted_slices = weights * slices
    padded = jnp.pad(weighted_slices,
                     pad_width=((0, 0), (margin, margin), (margin, margin)),
                     mode="constant", constant_values=0)

    # dx = int(shape[0] / n_patches_per_axis)
    # dy = int(shape[1] / n_patches_per_axis)

    # kernelcuts = (shape[0] - 2*dx) // 2
    # roll_kernel = np.fft.fftshift(kernel, axes=(1, 2))
    # cut_kernel = roll_kernel[:, kernelcuts:-kernelcuts, kernelcuts:-kernelcuts]
    # rollback_kernel = np.fft.ifftshift(cut_kernel, axes=(1, 2))

    # pkernel = jnp.pad(rollback_kernel,
    #                   pad_width=((0, 0), (margin, margin), (margin, margin)),
    #                   mode="constant",
    #                   constant_values=0)
    # TODO Prep Kernel Norm

    pkernel = kernel
    ndom = Domain((1, *shape), (None, *domain.distances))
    convolved = jifty_convolve(pkernel, padded,
                               ndom, axes=(1, 2))
    padded_shape = [ii+2*margin for ii in shape]

    def patch_w_margin(array):
        return slice_patches(array, padded_shape,
                             n_patches_per_axis, margin)

    primal = np.empty(padded_shape)
    overlap_add = jax.linear_transpose(patch_w_margin, primal)
    padded_res = overlap_add(convolved)[0]
    res = padded_res[margin:-margin, margin:-margin]
    return res


def jifty_convolve(x, y, domain, axes):
    """Perform an FFT convolution.

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
    res = jnp.fft.ifftn(hx*hy, axes=axes)
    res = dvol*res
    return res.real
