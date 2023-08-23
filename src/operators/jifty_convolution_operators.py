import jax.numpy as jnp
import jax.lax
import jax

import numpy as np
import nifty8.re as jft

from ..library.utils import convolve_field_operator


def _bilinear_weights(shape):
    """Build bilinear interpolation kernel."""
    if shape[0]/2 != int(shape[0]/2):
        raise ValueError("this should happen")
    # FIXME get better weights for non by 2 divisible numbers
    a = np.linspace(0, 1, int(shape[0]/2), dtype="float64")
    b = np.concatenate([a, jnp.flip(a)])
    return np.outer(b, b)


def linpatch_convolve(x, shape, kernel, n_patches_per_axis,
                      margin):
    """Functional version of linear patching convolution.

    Parameters:
    -----------
    x : input array
    shape: shape of input array
    kernel: np.array
        Array containing the different kernels for the inhomogeneos convolution
    n_patches_per_axis: int
        Number of patches
    additional_margin: int
        Size of the margin. Number of pixels on one boarder.

    """
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
    convolved = jifty_convolve(pkernel, padded, axes=(1, 2))
    padded_shape = [ii+2*margin for ii in shape]

    def patch_w_margin(array):
        return slice_patches(array, padded_shape,
                             n_patches_per_axis, margin)

    primal = np.empty(padded_shape)
    overlap_add = jax.linear_transpose(patch_w_margin, primal)
    padded_res = overlap_add(convolved)[0]
    res = padded_res[margin:-margin, margin:-margin]
    return res


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

    ndx = np.meshgrid(*ids, indexing="ij")
    f = jax.vmap(slicer, in_axes=(0, 0), out_axes=(0))
    return f(*(nn.flatten() for nn in ndx))


def jifty_convolve(x, y, axes):
    """Perform an FFT convolution."""
    hx = jnp.fft.fftn(x, axes=axes)
    hy = jnp.fft.fftn(y, axes=axes)
    res = jnp.fft.ifftn(hx*hy, axes=axes)
    # FIXME VOLUME FACTOR MISSING
    return res.real
