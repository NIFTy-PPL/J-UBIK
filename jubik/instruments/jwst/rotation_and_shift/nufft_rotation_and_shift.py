# SPDX-License-Identifier: BSD-2-Clause
# Authors: Julian Rüstig and Matteo Guardiani

# Copyright(C) 2024 Max-Planck-Society

# %%

import jax
import jax.numpy as jnp
from jax.numpy.fft import fftshift, ifft2
import numpy as np
from numpy.typing import ArrayLike
from astropy.units import Quantity

from typing import Union, Callable
from functools import reduce


def build_nufft_rotation_and_shift(
    sky_shape: tuple[int, int],
    out_shape: tuple[int, int],
    mode: str = "constant",
    band_limit: tuple[float, float] | None = None,
    capacity: int | None = None,
) -> Callable[[ArrayLike], ArrayLike]:
    """Builds non-uniform FFT interpolation model.

    Parameters
    ----------
    sky_shape: Tuple[int, int]
        The shape of the reconstruction array (sky shape)
    out_shape: Tuple[int, int]
        The shape of the subsample array.
    mode: str
        The mode of the interpolation. ['wrap', 'constant']
    band_limit: tuple[float, float] | None
        ``(b_y, b_x)`` in cycles per sky pixel, each in (0, 0.5). The sky's
        Fourier modes outside the ellipse ``(k_y / b_y)^2 + (k_x / b_x)^2 <= 1``
        are dropped before the interpolation, so sampling the subsampled grid
        cannot alias sky structure beyond the band into the data. None keeps
        every mode.
    capacity: int | None
        With mode ``constant``, interpolate only at the points inside the sky,
        gathered into this many slots (the others read zero anyway); when more
        points are inside, interpolate at every point. The result is the same
        either way. None interpolates at every point.

    Returns
    -------
    rotate_shift_subsample : function
        The interpolation function ``(field, subsample_centers_yx) -> out``.
        ``subsample_centers_yx`` has shape ``(2, *out_shape)`` or, for stacked
        exposures, ``(n, 2, *out_shape)``; ``out`` drops the coordinate axis.

    Notes
    -----
    The sky is the reconstruction array, we assume a one-to-one relation
    between the sky brightness (flux density) and the flux:
        flux(x, y) = sky(x, y) * sky_dvol
    """
    from jax_finufft import nufft2
    from jax_finufft.options import Opts

    # the points come in the detector grids' raster order, so cuFINUFFT's sort
    # each call only costs (V100, 21 F277W exposures: ~25% of a likelihood metric)
    opts = Opts(gpu_sort=False)

    xy_conversion = 2 * np.pi / np.array(sky_shape)[:, None]
    out_shape = tuple(out_shape)
    crop, keep = (...,), 1.0
    if band_limit is not None:
        # unit conversion can round a Nyquist band to just below 0.5
        if not all(0 < b < 0.5 * (1 - 1e-9) for b in band_limit):
            raise ValueError(
                f"band_limit {band_limit} cycles per sky pixel: each must lie in "
                "(0, 0.5), below the sky grid's Nyquist frequency"
            )
        ky, kx = (np.fft.fftshift(np.fft.fftfreq(n)) for n in sky_shape)
        disk = (ky[:, None] / band_limit[0]) ** 2 + (kx / band_limit[1]) ** 2 <= 1
        # symmetric about k = 0, so the crop is odd and centred as FINUFFT reads it
        rows, cols = np.flatnonzero(disk.any(1)), np.flatnonzero(disk.any(0))
        crop = (slice(rows[0], rows[-1] + 1), slice(cols[0], cols[-1] + 1))
        keep = jnp.asarray(disk[crop])

    @jax.custom_batching.custom_vmap
    def fits(n_inside):
        return n_inside <= capacity

    # vmapped (NIFTy's kl_map over samples): one branch for all samples, since a
    # batched predicate turns the cond into a select that runs both branches
    @fits.def_vmap
    def _fits_batched(axis_size, in_batched, n_inside):
        return jnp.all(n_inside <= capacity), False

    def rotate_shift_subsample(field, subsample_centers_yx):
        actual_shape = tuple(subsample_centers_yx.shape[-2:])
        if actual_shape != out_shape:
            raise ValueError(
                f"subsample_centers_yx trailing shape {actual_shape} "
                f"does not match out_shape {out_shape}"
            )
        f_field = fftshift(ifft2(field))[crop] * keep  # FINUFFT modeord=0: centred k
        # Stacked exposures go into one nufft2 call: under vmap jax_finufft
        # copies the sky per exposure and runs one setpts/execute each.
        centers_yx = jnp.moveaxis(subsample_centers_yx, -3, 0)
        coords = xy_conversion * centers_yx.reshape(2, -1)

        if mode == "constant":
            mask = jnp.any((coords > 2 * np.pi) + (coords < 0), axis=0)
            coords = jnp.where(mask, 0.0, coords)
        elif mode != "wrap":
            raise ValueError("mode must either be `wrap` or `constant`.")

        def nufft(c):
            return nufft2(f_field, c[0], c[1], opts=opts).real

        if capacity is None or mode != "constant" or capacity >= coords.shape[1]:
            out = nufft(coords)
        else:
            n_inside = jnp.sum(~mask)

            def inside_only(coords):
                idx = jnp.nonzero(~mask, size=capacity, fill_value=0)[0]
                vals = jnp.where(
                    jnp.arange(capacity) < n_inside, nufft(coords[:, idx]), 0
                )
                # add, not set: the unused slots repeat index 0 (with value 0)
                return jnp.zeros(coords.shape[1], vals.dtype).at[idx].add(vals)

            out = jax.lax.cond(fits(n_inside), inside_only, nufft, coords)
        if mode == "constant":
            out = jnp.where(mask, 0.0, out)
        return jnp.reshape(out, subsample_centers_yx.shape[:-3] + out_shape)

    return rotate_shift_subsample
