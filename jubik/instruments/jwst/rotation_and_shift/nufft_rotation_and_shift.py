# SPDX-License-Identifier: BSD-2-Clause
# Authors: Julian Rüstig and Matteo Guardiani

# Copyright(C) 2024 Max-Planck-Society

# %%

import jax.numpy as jnp
from jax.numpy.fft import ifftshift, ifft2
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
        The interpolation function.

    Notes
    -----
    The sky is the reconstruction array, we assume a one-to-one relation
    between the sky brightness (flux density) and the flux:
        flux(x, y) = sky(x, y) * sky_dvol
    """
    from jax_finufft import nufft2

    xy_conversion = 2 * np.pi / np.array(sky_shape)[:, None]

    def rotate_shift_subsample(field, subsample_centers_yx):
        f_field = ifftshift(ifft2(field))
        coords = xy_conversion * subsample_centers_yx.reshape(2, -1)

        if mode == "constant":
            mask = jnp.any((coords > 2 * np.pi) + (coords < 0), axis=0)
            coords = jnp.where(mask, 0.0, coords)
        elif mode != "wrap":
            raise ValueError("mode must either be `wrap` or `constant`.")

        out = nufft2(f_field, coords[0], coords[1]).real
        if mode == "constant":
            out = jnp.where(mask, 0.0, out)
        return jnp.reshape(out, out_shape)

    return rotate_shift_subsample
