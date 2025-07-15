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

from .psf_learning import LearnablePsf


def _build_vmap_apply(psf_kernel_shape: tuple[int]) -> Callable[ArrayLike, ArrayLike]:
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


class PsfDynamic(jft.Model):
    """Implements the convolution by a dynamic psf kernel, i.e. a psf kernel that is
    learned."""

    def __init__(
        self,
        sky_shape_with_dtype: jft.ShapeWithDtype,
        psf_kernel: LearnablePsf | np.ndarray | None,
    ):
        """
        Parameters
        ----------
        sky_shape_with_dtype: jft.ShapeWithDtype
            The `ShapeWithDtype` of the sky.
        psf_kernel: LearnablePsf | np.ndarray | None
            If None, the apply will just return the input field.
            Elif np.ndarray, the input field will by convolved by the `psf_kernel`.
        """

        assert isinstance(psf_kernel, LearnablePsf)

        self.kernel = psf_kernel
        self._convolve = _build_vmap_apply(self.kernel.shape)

        super().__init__(
            domain=(sky_shape_with_dtype, self.kernel.domain), white_init=True
        )

    def __call__(self, x):
        field, psf_kernel_x = x
        psf_kernel = self.model(psf_kernel_x)
        return self._convolve(field, psf_kernel)


def build_psf_operator_strategy(
    sky_shape_with_dtype: jft.ShapeWithDtype,
    psf_kernel: np.ndarray | LearnablePsf | None,
) -> PsfDynamic | PsfStatic:
    """Build either a static or a dynamic (with learned kernel) PsfDynamic.

    Parameters
    ----------
    sky_shape_with_dtype: jft.ShapeWithDtype
        The shape and dtype of the sky.
    psf_kernel:  np.ndarray | LearnablePsf | None
        The psf kernel to by applied to the sky.
    """
    assert hasattr(sky_shape_with_dtype, "shape")
    assert hasattr(sky_shape_with_dtype, "dtype")

    if isinstance(psf_kernel, np.ndarray) or psf_kernel is None:
        return PsfStatic(sky_shape_with_dtype, psf_kernel)

    return PsfDynamic(sky_shape_with_dtype, psf_kernel)
