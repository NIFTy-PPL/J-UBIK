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
    indexing: str = "ij",
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

    # TODO: Check why we need the subsample centers swapped.
    # 07-03-25: It seems that the linear & finufft interpolation needs the
    # input points swapped.
    # Maybe: this comes from the matrix style indexing?
    # 16-03-25: Yes, always take matrix style indexing (meshgrid='ij')!

    if indexing == "ij":

        def rotate_shift_subsample(field, subsample_centers):
            f_field = ifftshift(ifft2(field))
            xy_finufft = xy_conversion * subsample_centers.reshape(2, -1)

            if mode == "constant":
                mask = jnp.any((xy_finufft > 2 * np.pi) + (xy_finufft < 0), axis=0)
                xy_finufft = jnp.where(mask, 0.0, xy_finufft)
            elif mode != "wrap":
                raise ValueError("mode must either be `wrap` or `constant`.")

            out = nufft2(f_field, xy_finufft[0], xy_finufft[1]).real

            if mode == "constant":
                out = jnp.where(mask, 0.0, out)

            return jnp.reshape(out, out_shape)

    elif indexing == "xy":
        out_shape = out_shape[1], out_shape[0]

        def rotate_shift_subsample(field, subsample_centers):
            f_field = ifftshift(ifft2(field.T))
            xy_finufft = xy_conversion * subsample_centers.reshape(2, -1)

            if mode == "constant":
                mask = jnp.any((xy_finufft > 2 * np.pi) + (xy_finufft < 0), axis=0)
                xy_finufft = jnp.where(mask, 0.0, xy_finufft)
            elif mode != "wrap":
                raise ValueError("mode must either be `wrap` or `constant`.")

            out = nufft2(f_field, xy_finufft[1], xy_finufft[0]).real

            if mode == "constant":
                out = jnp.where(mask, 0.0, out)

            return jnp.reshape(out, out_shape)
    else:
        raise ValueError("Need either provide `ij` or `xy` indexing.")

    return rotate_shift_subsample
