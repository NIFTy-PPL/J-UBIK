# SPDX-License-Identifier: BSD-2-Clause
# Authors: Julian Rüstig

# Copyright(C) 2025 Max-Planck-Society

# %
from functools import partial
from typing import Callable

import nifty.re as jft
import numpy as np
from numpy.typing import ArrayLike
from jax import vmap
from jax.scipy.signal import fftconvolve
from jax.tree_util import Partial


def _build_vmap_apply(psf_kernel_shape: tuple[int]) -> Callable[[ArrayLike], ArrayLike]:
    if len(psf_kernel_shape) == 2:
        return partial(fftconvolve, mode="same")
    elif len(psf_kernel_shape) == 3:
        return vmap(Partial(fftconvolve, mode="same"), in_axes=(0, 0))
    else:
        raise ValueError("Unknown psf_kernel shape")


class PsfStatic(jft.Model):
    """Implements the convolution by a static psf kernel"""

    def __init__(
        self,
        sky_shape_with_dtype: jft.ShapeWithDtype,
        psf_kernel: np.ndarray | None,
    ):
        """
        Parameters
        ----------
        sky_shape_with_dtype: jft.ShapeWithDtype
            The `ShapeWithDtype` of the sky.
        psf_kernel: np.ndarray | None
            If None, the apply will just return the input field.
            Else, the input field will by convolved by the `psf_kernel`.
        """

        self.kernel = psf_kernel
        if psf_kernel is not None:
            self._convolve = _build_vmap_apply(psf_kernel.shape)

        super().__init__(domain=(sky_shape_with_dtype, {}))

    def __call__(self, x):
        field, _ = x
        if self.kernel is None:
            return field
        return self._convolve(field, self.kernel)
