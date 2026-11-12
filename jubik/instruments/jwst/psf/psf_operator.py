# SPDX-License-Identifier: BSD-2-Clause
# Authors: Julian Rüstig

# Copyright(C) 2025 Max-Planck-Society

# %
import jax.numpy as jnp
import nifty.re as jft
import numpy as np
from scipy.fft import next_fast_len

# FFT length per axis, for a sky axis of n pixels and a centered kernel of k pixels.
# "fast" is the deliberate default: the kernel center sits at index 0, so n + k // 2
# avoids wrap-around, rounded up to a 5-smooth length. "exact" is the length jax
# `fftconvolve` uses, n + k - 1; tests set it to compare with `fftconvolve(mode="same")`.
FFT_LENGTH = "fast"
_FFT_LENGTH_RULES = {
    "fast": lambda n, k: next_fast_len(max(n + k // 2, k), real=True),
    "exact": lambda n, k: n + k - 1,
}


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


def _create_spectrum(kernel: np.ndarray, fft_shape: tuple[int, int]) -> np.ndarray:
    """`rfft2` of the centered `kernel`, zero-padded to `fft_shape` with its center
    pixel `k // 2` rolled to index 0. The circular convolution with this spectrum,
    cropped to the first `n` pixels per axis, is `fftconvolve(mode="same")`."""
    k = kernel.shape[-2:]
    pad = [(0, 0)] * (kernel.ndim - 2) + [(0, n - m) for n, m in zip(fft_shape, k)]
    kernel = np.roll(np.pad(kernel, pad), (-(k[0] // 2), -(k[1] // 2)), axis=(-2, -1))
    return np.fft.rfft2(kernel)


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

        self._spectrum = None
        if psf_kernel is not None:
            kernel = _center_even_axes(np.asarray(psf_kernel))
            shape, k = sky_shape_with_dtype.shape, kernel.shape[-2:]
            if np.broadcast_shapes(kernel.shape[:-2], shape[:-2]) != shape[:-2]:
                raise ValueError(f"psf kernel {kernel.shape} does not fit sky {shape}")
            rule = _FFT_LENGTH_RULES[FFT_LENGTH]
            self._fft_shape = tuple(rule(n, m) for n, m in zip(shape[-2:], k))
            # shortcut: the spectrum is baked into every jitted function (~4 MB per exposure
            # at L=720, complex128); cast to the field's complex dtype if memory bites.
            self._spectrum = _create_spectrum(kernel, self._fft_shape)

        super().__init__(domain=(sky_shape_with_dtype, {}))

    def __call__(self, x):
        field, _ = x
        if self._spectrum is None:
            return field
        s = self._fft_shape
        out = jnp.fft.irfft2(jnp.fft.rfft2(field, s=s) * self._spectrum, s=s)
        return out[..., : field.shape[-2], : field.shape[-1]]
