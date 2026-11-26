# SPDX-License-Identifier: BSD-2-Clause
# Authors: Matteo Guardiani

# Copyright(C) 2026 Max-Planck-Society

# %%

"""The sky, convolved by the PSF and averaged over each detector pixel, computed
in Fourier space with one NUFFT point per detector pixel.

For exposure e and detector pixel p, with centre x_p in sky pixels,

    d_p = sum_k c(k) T_e(k) B_e(k) exp(-2 pi i k . (x_p + shift_e)),

summed over the sky's Fourier modes k inside the band limit. `c` are the modes of
the sky zero-padded so that the PSF cannot wrap around the sky box (the sky is
zero outside it, as NUFFT mode `constant`); `T_e` is the transfer function of the
exposure's kernel at the rotated frequencies `J_e^T k`; `B_e` is the detector
pixel's box, `sinc((J_e^T k)_0) sinc((J_e^T k)_1)`; the shift is a phase. `J_e`
maps detector pixels to sky pixels. `d_p` is the mean over the pixel of the
PSF-convolved sky: the exact pixel integral, with no sub-pixel grid to alias.
"""

import jax.numpy as jnp
import nifty.re as jft
import numpy as np
from astropy import units as u
from scipy.fft import next_fast_len

from ..parse.rotation_and_shift.rotation_and_shift import NufftConfig
from ..psf.psf_operator import _center_even_axes
from .coordinates_correction import (
    Coordinates,
    CoordinatesCorrectedShiftOnly,
    prior_extent,
)


class FourierResponse(jft.Model):
    """The sky in the data pixels, see the module docstring. Built by
    `build_fourier_response`."""

    def __init__(
        self,
        sky_domain: dict[str, jft.ShapeWithDtype],
        coordinates: Coordinates | CoordinatesCorrectedShiftOnly,
        transfer: np.ndarray,
        frequencies: tuple[np.ndarray, np.ndarray],
        points: tuple[np.ndarray, np.ndarray],
        dft: tuple[np.ndarray, np.ndarray],
        out_shape: tuple[int, int, int],
    ):
        """
        Parameters
        ----------
        sky_domain : dict
            The single sky key and its 2-D shape.
        coordinates : Coordinates | CoordinatesCorrectedShiftOnly
            Kept as `self.coordinates`, as `RotationAndShift` does (callers read
            its shift model there); its shift, if any, enters as a phase.
        transfer : np.ndarray
            `T_e B_e` on the cropped modes, zero outside the band, `(n, My, Mx)`.
        frequencies : tuple
            The cropped modes' `k_y` `(My, 1)` and `k_x` `(1, Mx)`, cycles per sky pixel.
        points : tuple
            The pixel centres times 2 pi / the padded grid, `(n, H * W)` each.
        dft : tuple
            `(A_y, A_x)`, `(My, N_y)` and `(Mx, N_x)`: `A_y sky A_x^T` are the
            cropped modes of the zero-padded sky's `ifft2`, centred.
        out_shape : tuple
            `(n, H, W)`.
        """
        self._sky_key = next(iter(sky_domain))
        self.coordinates = coordinates
        self._shift = getattr(coordinates, "shift_in_pixels", None)
        self._transfer = jnp.asarray(transfer)
        self._ky, self._kx = (jnp.asarray(k) for k in frequencies)
        self._points = tuple(jnp.asarray(p) for p in points)
        self._dft = tuple(jnp.asarray(a) for a in dft)
        self._out_shape = out_shape
        super().__init__(domain=sky_domain | coordinates.domain)

    def __call__(self, x):
        from jax_finufft import nufft2

        # FINUFFT modeord=0: centred k, as in build_nufft_rotation_and_shift
        modes = self._dft[0] @ x[self._sky_key] @ self._dft[1].T
        modes = modes * self._transfer
        if self._shift is not None:
            shift = self._shift(x)[..., None, None]  # sky pixels
            modes = modes * jnp.exp(-2j * np.pi * shift[:, 0] * self._ky)
            modes = modes * jnp.exp(-2j * np.pi * shift[:, 1] * self._kx)
        return nufft2(modes, *self._points).real.reshape(self._out_shape)


def build_fourier_response(
    sky_domain: dict[str, jft.ShapeWithDtype],
    coordinates: Coordinates | CoordinatesCorrectedShiftOnly,
    kernel: np.ndarray,
    subsample: int,
    algorithm_config: NufftConfig,
    sky_pixel_scales_yx: u.Quantity,
    mask: np.ndarray | None = None,
) -> FourierResponse:
    """The sky in the data pixels, PSF-convolved and pixel-averaged in Fourier space.

    Parameters
    ----------
    sky_domain : dict
        The single sky key and its 2-D shape.
    coordinates : Coordinates | CoordinatesCorrectedShiftOnly
        The subsampled data pixel centres on the sky grid, `(n, 2, H s, W s)`, and
        the shift correction. They give the pixel centres (each pixel's mean) and
        `J`, the detector-to-sky Jacobian (the cutout's mean).
    kernel : np.ndarray
        The PSF kernel of each exposure on its subsampled grid, `(n, K, K)` or
        `(K, K)`, as `PsfStatic` takes it.
    subsample : int
        Kernel pixels per detector pixel.
    algorithm_config : NufftConfig
        Its `band_limit` (cycles per arcsec) sets the modes; mode `constant`.
    sky_pixel_scales_yx : astropy.units.Quantity
        To read the band limit in cycles per sky pixel.
    mask : np.ndarray | None
        The data mask, `(n, H, W)`. The sky is padded so that no light wraps
        around onto an unmasked pixel (None: onto any pixel) while the shift
        stays within its prior's reach (|mean| + 3 sigma, or the bounds).
    """
    band_limit = algorithm_config.band_limit
    if band_limit is None or band_limit <= 0 or algorithm_config.mode != "constant":
        raise ValueError(
            "the Fourier response needs a positive band_limit and mode 'constant'"
        )
    if not isinstance(coordinates, (Coordinates, CoordinatesCorrectedShiftOnly)):
        raise NotImplementedError(
            "the Fourier response takes the shift correction model only: a "
            "rotation would turn the kernel's and the pixel's transfer functions"
        )
    s = subsample
    shifted = isinstance(coordinates, CoordinatesCorrectedShiftOnly)
    fixed = np.asarray(coordinates._coordinates if shifted else coordinates.coordinates)
    n, _, hs, ws = fixed.shape
    out_shape = (n, hs // s, ws // s)
    centres = fixed.reshape(n, 2, out_shape[1], s, out_shape[2], s).mean((3, 5))
    # J[e, i, j]: sky pixels along axis i per detector pixel along axis j
    jac = s * np.stack([np.diff(fixed, axis=a).mean((-2, -1)) for a in (-2, -1)], -1)

    kernel = _center_even_axes(np.asarray(kernel, float))
    kernel = np.broadcast_to(kernel, (n,) + kernel.shape[-2:])
    half = (np.array(kernel.shape[-2:]) - 1) / 2 / s + 0.5  # kernel + pixel, det px
    reach = np.max(np.abs(jac) @ half, axis=0)  # along each sky axis, sky pixels
    if shifted:
        shift = prior_extent(coordinates.shift_and_rotation.prior.shift)
        reach = reach + shift * coordinates._1_over_pixel_distance

    sky_shape = np.array(next(iter(sky_domain.values())).shape)
    sel = np.ones(out_shape, bool) if mask is None else np.asarray(mask, bool)
    used = centres.transpose(1, 0, 2, 3)[:, sel]  # (2, unmasked pixels)
    lo = np.minimum(used.min(axis=1), 0)
    hi = np.maximum(used.max(axis=1), sky_shape - 1)
    # light from sky pixel y reaches q directly; its copy at y +- L must not
    pad = tuple(
        next_fast_len(int(r + max(nn - 1 - l_, h_)) + 1)
        for r, nn, l_, h_ in zip(reach, sky_shape, lo, hi)
    )

    band = band_limit * sky_pixel_scales_yx.to_value(u.arcsec)
    ky, kx = (np.fft.fftshift(np.fft.fftfreq(m)) for m in pad)
    disk = (ky[:, None] / band[0]) ** 2 + (kx / band[1]) ** 2 <= 1
    rows, cols = np.flatnonzero(disk.any(1)), np.flatnonzero(disk.any(0))
    crop = (slice(rows[0], rows[-1] + 1), slice(cols[0], cols[-1] + 1))
    ky, kx, keep = ky[crop[0], None], kx[None, crop[1]], disk[crop]
    k = np.stack(np.broadcast_arrays(ky, kx))
    det = np.einsum("nij,i...->nj...", jac, k)  # J^T k, cycles per detector pixel
    if np.abs(det[:, :, keep]).max() >= s / 2:
        raise ValueError(
            f"band_limit {band_limit} cycles/arcsec reaches the "
            "kernel grid's Nyquist frequency: lower it or raise the subsample"
        )

    from jax_finufft import nufft2

    # the kernel's DTFT at J^T k / s: a kernel pixel m moves light by +J m / s
    otf = nufft2(
        jnp.asarray(kernel, complex),
        *(2 * np.pi / s * det[:, j].reshape(n, -1) for j in (0, 1)),
        iflag=1,
        eps=1e-12,
    )
    transfer = np.asarray(otf).reshape(det[:, 0].shape) * np.prod(np.sinc(det), 1)
    transfer = np.where(keep, transfer, 0)

    points = tuple(
        2 * np.pi * np.mod(centres[:, i], m).reshape(n, -1) / m
        for i, m in enumerate(pad)
    )
    # the band of the padded sky's ifft2 by matrix DFTs: cheaper than the FFT,
    # which computes the whole padded grid
    dft = tuple(
        np.exp(2j * np.pi * np.outer(f, np.arange(nn))) / m
        for f, nn, m in zip((ky.ravel(), kx.ravel()), sky_shape, pad)
    )
    return FourierResponse(
        sky_domain, coordinates, transfer, (ky, kx), points, dft, out_shape
    )
