import jax.numpy as jnp
import jax.lax as jlax

import numpy as np
import nifty8.re as jft

from .zero_padder import MarginZeroPadder
from ..library.utils import convolve_field_operator


def _bilinear_weights(shape):
    """Build bilinear interpolation kernel."""
    if shape/2 != int(shape/2):
        raise ValueError("this should happen")
    a = np.linspace(0, 1, int(shape/2), dtype="float64")
    b = np.concatenate([a, jnp.flip(a)])
    return np.outer(b, b)


def linpatch_convolve(signal, kernel"""Functional version of linear patching convolution.

    This is a longer part of the docstring:

    """
    signal
    return kernel


def slice_patches(x, shape, n_patches_per_axis, overlap_margin):
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
    dr = overlap_margin
    dx = [int((shape[0] - 2 * dr) / n_patches_per_axis)]
    dy = [int((shape[1] - 2 * dr) / n_patches_per_axis)]
    padded_x = jnp.pad(x, pad_width=((dx//2, dx//2), (dy//2, dy//2)),
                       mode="constant", constant_values=0)

    def slicer(xi, yi, dx, dy):
        return jlax.dynamic_slice(image, start_indices=(xi,yi), slice_sizes=(dx,dy))

    return padded_x

        #     listing = []
        #     for l in range(self.sqrt_n_patch):
        #         y_i = l * dy
        #         y_f = y_i + 2 * dy + 2 * self.dr
        #         for k in range(self.sqrt_n_patch):
        #             x_i = k * dx
        #             x_f = x_i + 2 * dx + 2 * self.dr
        #             tmp = xplus[x_i:x_f, y_i:y_f]
        #             listing.append(tmp)
        #     res = ift.Field.from_raw(self._target, np.array(listing))
        # else:
        #     taped = np.zeros([self._domain.shape[0] + self.dx] * 2)
        #     i = 0
        #     for n in range(self.sqrt_n_patch):
        #         y_i = n * dy
        #         y_f = y_i + 2 * dy + 2 * self.dr
        #         for m in range(self.sqrt_n_patch):
        #             x_i = m * dx
        #             x_f = x_i + 2 * dx + 2 * self.dr
        #             taped[x_i:x_f, y_i:y_f] += val[i]
        #             i += 1
        #     taped_s = np.zeros(self.domain.shape)
        #     taped_s += taped[self.dx // 2: -self.dx // 2, self.dy // 2: -self.dy // 2]
        #     res = ift.Field.from_raw(self._domain, taped_s)
        # return res
