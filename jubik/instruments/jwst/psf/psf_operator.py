# SPDX-License-Identifier: BSD-2-Clause
# Authors: Julian Rüstig

# Copyright(C) 2025 Max-Planck-Society

# %
from functools import partial
from typing import Callable

import jax.numpy as jnp
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


def _center_even_axes(kernel: np.ndarray) -> np.ndarray:
    """Recenter even kernel axes onto the middle pixel of an odd grid.

    PSF kernels are centered at `(K - 1) / 2`. For even `K` that falls between two
    pixels, and `fftconvolve(mode="same")` then shifts the image by half a pixel.
    Each even axis is zero-padded in front to `K + 1` (center at `K / 2 + 0.5`) and
    Fourier-shifted by -0.5 pixel onto the middle pixel `K / 2`. An odd grid has no
    Nyquist bin, so the result is real, and the sum (DC term) is unchanged.
    """
    for axis in (-2, -1):
        size = kernel.shape[axis]
        if size % 2:
            continue
        pad = [(0, 0)] * kernel.ndim
        pad[axis] = (1, 0)
        kernel = np.pad(kernel, pad)
        phase = np.exp(1j * np.pi * np.fft.fftfreq(size + 1))
        phase = np.expand_dims(phase, tuple(range(axis + 1, 0)))
        kernel = np.fft.ifft(np.fft.fft(kernel, axis=axis) * phase, axis=axis).real
    return kernel


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
            self.kernel = _center_even_axes(np.asarray(psf_kernel))
            self._convolve = _build_vmap_apply(psf_kernel.shape)

        super().__init__(domain=(sky_shape_with_dtype, {}))

    def __call__(self, x):
        field, _ = x
        if self.kernel is None:
            return field
        return self._convolve(field, self.kernel)


class PsfGaussianBroadening(jft.Model):
    """Convolution by a static psf kernel broadened by a Gaussian of learnable width.

    The Gaussian is the inverse FFT of its transfer function on the (odd, centered)
    kernel grid, so it is symmetric and sums to one for every width. Its full linear
    convolution with the kernel thus keeps the kernel's sum and centroid, and width 0
    gives back the static kernel.
    """

    def __init__(
        self,
        sky_shape_with_dtype: jft.ShapeWithDtype,
        psf_kernel: np.ndarray,
        width: jft.Model,
    ):
        """
        Parameters
        ----------
        sky_shape_with_dtype: jft.ShapeWithDtype
            The `ShapeWithDtype` of the sky.
        psf_kernel: np.ndarray
            The static kernel, 2-D or one per observation (3-D).
        width: jft.Model
            Standard deviation of the Gaussian in sky pixels, a scalar or one per
            observation.
        """
        self.kernel = _center_even_axes(np.asarray(psf_kernel))
        self.width = width
        self._convolve = _build_vmap_apply(self.kernel.shape)
        fy, fx = (np.fft.fftfreq(n) for n in self.kernel.shape[-2:])
        self._frequency2 = fy[:, None] ** 2 + fx[None, :] ** 2

        super().__init__(domain=(sky_shape_with_dtype, width.domain))

    def broadened_kernel(self, params):
        # shortcut: the Gaussian wraps on the kernel grid, so widths near the kernel
        # half-size stop being Gaussian (still normalized); pad the grid if needed.
        sigma = jnp.broadcast_to(self.width(params), self.kernel.shape[:-2])
        transfer = jnp.exp(
            -2 * jnp.pi**2 * sigma[..., None, None] ** 2 * self._frequency2
        )
        gauss = jnp.fft.fftshift(jnp.fft.ifft2(transfer).real, axes=(-2, -1))
        return fftconvolve(self.kernel, gauss, mode="full", axes=(-2, -1))

    def __call__(self, x):
        field, params = x
        return self._convolve(field, self.broadened_kernel(params))
