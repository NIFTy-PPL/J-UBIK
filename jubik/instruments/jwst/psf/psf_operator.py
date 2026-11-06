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
from scipy.fft import next_fast_len


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

        self.kernel_fft = None
        if psf_kernel is not None:
            if np.ndim(psf_kernel) not in (2, 3) or (
                np.shape(psf_kernel)[:-2] != sky_shape_with_dtype.shape[:-2]
            ):
                raise ValueError(
                    "psf_kernel must be 2-D, or 3-D with one kernel per sky observation"
                )
            kernel = _center_even_axes(np.asarray(psf_kernel))
            sky = sky_shape_with_dtype.shape[-2:]
            # Kernel pixels more than N - 1 from the (odd) centre never reach the output.
            size = tuple(min(k, 2 * n - 1) for n, k in zip(sky, kernel.shape[-2:]))
            cut = tuple(
                slice((k - m) // 2, (k + m) // 2)
                for k, m in zip(kernel.shape[-2:], size)
            )
            kernel = kernel[(..., *cut)]
            # `fftconvolve(mode="same")` with the kernel transformed once, cropped to the
            # sky around the centre. On a grid of at least N + (K - 1) / 2 the circular
            # wrap-around only reaches discarded pixels; a 5-smooth grid is fast. The trim
            # to 2N - 1 keeps that grid at least K wide.
            self._fft_shape = tuple(
                next_fast_len(n + (k - 1) // 2, real=True) for n, k in zip(sky, size)
            )
            self._crop = tuple(
                slice((k - 1) // 2, (k - 1) // 2 + n) for n, k in zip(sky, size)
            )
            self.kernel_fft = jnp.asarray(np.fft.rfft2(kernel, self._fft_shape))

        super().__init__(domain=(sky_shape_with_dtype, {}))

    def __call__(self, x):
        field, _ = x
        if self.kernel_fft is None:
            return field
        return self._convolve(field, self.kernel_fft)

    def _convolve(self, field, kernel_fft):
        field_fft = jnp.fft.rfft2(field, self._fft_shape)
        convolved = jnp.fft.irfft2(field_fft * kernel_fft, self._fft_shape)
        return convolved[(..., *self._crop)]


class PsfKernelMix(jft.Model):
    """Convolution by a learnable mix of two static kernels, `(1 - t) K_a + t K_b`.

    For example a broad halo `K_a` and a sharp core `K_b`, so that `t` is a
    Strehl-like core fraction, or the PSF-star kernels of two nights. Both
    kernels sum to one and share their shape, so every `t` keeps the flux;
    kernels centred on `(K - 1) / 2`, as `PsfStatic` expects, keep their centroid
    for every `t`. Outside [0, 1] the mix extrapolates: sharper than the sharper
    kernel, broader than the broader one, and negative where the two differ most;
    `t` is not bounded here, nor the model clipped (the shot-noise variance only
    floors model + sky at 0). Both transforms are cached as in `PsfStatic` and
    mixed in Fourier space, so a call is one forward and one inverse FFT.
    """

    def __init__(
        self,
        sky_shape_with_dtype: jft.ShapeWithDtype,
        psf_kernel: np.ndarray,
        second_kernel: np.ndarray,
        weight: jft.Model,
    ):
        """
        Parameters
        ----------
        sky_shape_with_dtype: jft.ShapeWithDtype
            The `ShapeWithDtype` of the sky.
        psf_kernel, second_kernel: np.ndarray
            `K_a` and `K_b`, 2-D or one per observation (3-D), of the same shape
            and summing to one (per observation).
        weight: jft.Model
            The mixing weight `t`, a scalar or one per observation; `weight(x)`
            gives the `t` of a sample `x`.
        """
        _check_pair(psf_kernel, second_kernel)
        static = PsfStatic(sky_shape_with_dtype, psf_kernel)
        self._convolve = static._convolve
        self.kernel_fft = static.kernel_fft
        self.difference_fft = (
            PsfStatic(sky_shape_with_dtype, second_kernel).kernel_fft - self.kernel_fft
        )
        self.weight = weight

        super().__init__(domain=(sky_shape_with_dtype, weight.domain))

    def __call__(self, x):
        field, params = x
        t = jnp.broadcast_to(self.weight(params), self.kernel_fft.shape[:-2])
        return self._convolve(
            field, self.kernel_fft + t[..., None, None] * self.difference_fft
        )


def mixed_kernel(
    psf_kernel: np.ndarray, second_kernel: np.ndarray, weight: jft.Model
) -> jft.Model:
    """The kernel `(1 - t) K_a + t K_b` as a model of the parameters of `weight`.

    Checked and centred as in `PsfKernelMix`; for `PsfGaussianBroadening`, or
    evaluated once at a fixed `t` (a `weight` without parameters).
    """
    _check_pair(psf_kernel, second_kernel)
    a, b = (_center_even_axes(np.asarray(k)) for k in (psf_kernel, second_kernel))

    def kernel(x):
        t = jnp.broadcast_to(weight(x), a.shape[:-2])[..., None, None]
        return a + t * (b - a)

    return jft.Model(kernel, domain=weight.domain)


def _check_pair(psf_kernel, second_kernel):
    if np.shape(psf_kernel) != np.shape(second_kernel):
        raise ValueError(
            f"the kernels differ in shape, {np.shape(psf_kernel)} and "
            f"{np.shape(second_kernel)}"
        )
    for kernel in (psf_kernel, second_kernel):
        if not np.allclose(np.sum(kernel, axis=(-2, -1)), 1.0):
            raise ValueError("both kernels must sum to one (per observation)")


class PsfGaussianBroadening(jft.Model):
    """Convolution by a psf kernel broadened by a Gaussian of learnable width.

    The Gaussian is the inverse FFT of its transfer function on the (odd, centered)
    kernel grid, so it is symmetric and sums to one for every width. Its full linear
    convolution with the kernel thus keeps the kernel's sum and centroid, and width 0
    gives back the static kernel.
    """

    def __init__(
        self,
        sky_shape_with_dtype: jft.ShapeWithDtype,
        psf_kernel: np.ndarray | jft.Model,
        width: jft.Model,
    ):
        """
        Parameters
        ----------
        sky_shape_with_dtype: jft.ShapeWithDtype
            The `ShapeWithDtype` of the sky.
        psf_kernel: np.ndarray | jft.Model
            The static kernel, 2-D or one per observation (3-D), or a model of an
            odd, centred kernel, such as `mixed_kernel`, whose parameters join the
            domain.
        width: jft.Model
            Standard deviation of the Gaussian in pixels of the kernel grid (the
            subsampled data pixels), a scalar or one per observation. The gradient
            vanishes at width 0, so use a positive prior such as a log-normal.
        """
        if not isinstance(psf_kernel, jft.Model):
            centred = _center_even_axes(np.asarray(psf_kernel))
            psf_kernel = jft.Model(lambda _: centred, domain={})
        self.kernel = psf_kernel
        self.width = width
        shape = psf_kernel.target.shape
        if any(n % 2 == 0 for n in shape[-2:]):
            raise ValueError(
                "a kernel model must be odd and centred, e.g. mixed_kernel"
            )
        self._convolve = _build_vmap_apply(shape)
        fy, fx = (np.fft.fftfreq(n) for n in shape[-2:])
        self._frequency2 = fy[:, None] ** 2 + fx[None, :] ** 2

        super().__init__(
            domain=(sky_shape_with_dtype, psf_kernel.domain | width.domain)
        )

    def broadened_kernel(self, params):
        # shortcut: the Gaussian wraps on the kernel grid, so widths near the kernel
        # half-size stop being Gaussian (still normalized); pad the grid if needed.
        # Sub-pixel widths ring slightly negative (band-limited transfer function).
        kernel = self.kernel(params)
        sigma = jnp.broadcast_to(self.width(params), kernel.shape[:-2])
        transfer = jnp.exp(
            -2 * jnp.pi**2 * sigma[..., None, None] ** 2 * self._frequency2
        )
        gauss = jnp.fft.fftshift(jnp.fft.ifft2(transfer).real, axes=(-2, -1))
        return fftconvolve(kernel, gauss, mode="full", axes=(-2, -1))

    def __call__(self, x):
        field, params = x
        return self._convolve(field, self.broadened_kernel(params))
