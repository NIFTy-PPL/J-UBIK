# SPDX-License-Identifier: BSD-2-Clause
# Authors: Julian Rüstig and Matteo Guardiani

# Copyright(C) 2024 Max-Planck-Society

# %%

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

    xy_conversion = 2 * np.pi / np.array(sky_shape)[:, None]
    out_shape = tuple(out_shape)

    def rotate_shift_subsample(field, subsample_centers_yx):
        actual_shape = tuple(subsample_centers_yx.shape[-2:])
        if actual_shape != out_shape:
            raise ValueError(
                f"subsample_centers_yx trailing shape {actual_shape} "
                f"does not match out_shape {out_shape}"
            )
        f_field = fftshift(ifft2(field))  # FINUFFT modeord=0 wants centered k
        # Stacked exposures go into one nufft2 call: under vmap jax_finufft
        # copies the sky per exposure and runs one setpts/execute each.
        centers_yx = jnp.moveaxis(subsample_centers_yx, -3, 0)
        coords = xy_conversion * centers_yx.reshape(2, -1)

        if mode == "constant":
            mask = jnp.any((coords > 2 * np.pi) + (coords < 0), axis=0)
            coords = jnp.where(mask, 0.0, coords)
        elif mode != "wrap":
            raise ValueError("mode must either be `wrap` or `constant`.")

        out = nufft2(f_field, coords[0], coords[1]).real
        if mode == "constant":
            out = jnp.where(mask, 0.0, out)
        return jnp.reshape(out, subsample_centers_yx.shape[:-3] + out_shape)

    return rotate_shift_subsample
