# SPDX-License-Identifier: BSD-2-Clause
# Authors: Julian Rüstig

# Copyright(C) 2025 Max-Planck-Society

# %
import jax.numpy as jnp
import nifty.re as jft
import numpy as np
from scipy.fft import next_fast_len


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

        self._spectrum = None
        if psf_kernel is not None:
            kernel = _center_even_axes(np.asarray(psf_kernel))
            shape, k = sky_shape_with_dtype.shape, kernel.shape[-2:]
            if np.broadcast_shapes(kernel.shape[:-2], shape[:-2]) != shape[:-2]:
                raise ValueError(f"psf kernel {kernel.shape} does not fit sky {shape}")
            # Kernel center rolled to index 0: the circular convolution equals
            # `fftconvolve(mode="same")` on [:n] once L >= n + k // 2.
            self._fft_shape = tuple(
                next_fast_len(max(n + m // 2, m), real=True) for n, m in zip(shape[-2:], k)
            )
            pad = [(0, 0)] * (kernel.ndim - 2) + [
                (0, n - m) for n, m in zip(self._fft_shape, k)
            ]
            kernel = np.roll(np.pad(kernel, pad), (-(k[0] // 2), -(k[1] // 2)), axis=(-2, -1))
            # shortcut: the spectrum is baked into every jitted function (~4 MB per exposure
            # at L=720, complex128); cast to the field's complex dtype if memory bites.
            self._spectrum = np.fft.rfft2(kernel)

        super().__init__(domain=(sky_shape_with_dtype, {}))

    def __call__(self, x):
        field, _ = x
        if self._spectrum is None:
            return field
        s = self._fft_shape
        out = jnp.fft.irfft2(jnp.fft.rfft2(field, s=s) * self._spectrum, s=s)
        return out[..., : field.shape[-2], : field.shape[-1]]
