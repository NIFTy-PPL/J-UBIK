# SPDX-License-Identifier: BSD-2-Clause
# Authors: Julian Rüstig and Matteo Guardiani

# Copyright(C) 2024 Max-Planck-Society

# %%

from jax.numpy import reshape
from jax.numpy.fft import ifftshift, ifft2
import numpy as np
from numpy.typing import ArrayLike
from astropy.units import Quantity

from typing import Union, Callable
from functools import reduce


def build_nufft_rotation_and_shift(
    sky_dvol: ArrayLike,
    sub_dvol: ArrayLike,
    sky_shape: tuple[int, int],
    sky_distances: Union[tuple[Quantity, Quantity], tuple[float, float]],
    out_shape: tuple[int, int],
    indexing: str = 'ij',
    sky_as_brightness: bool = False
) -> Callable[[ArrayLike], ArrayLike]:
    """
    Builds non-uniform FFT interpolation model.

    Parameters
    ----------
    sky_dvol: float
        The volume of the sky/reconstruction pixels
    sub_dvol: float
        The volume of the subsample pixels.
        Typically, the data pixel is subsampled.
    sky_shape: Tuple[int, int]
        The shape of the reconstruction array (sky shape)
    sky_distances: Tuple[Quantity, Quantity]
        The sky_distances are needed to check for the consistency of the 
        `xy_conversion`.
    out_shape: Tuple[int, int]
        The shape of the subsample array.
    sky_as_brightness: bool
        If True, the sky will be treated as a brightness distribution.
        This is the same as setting sky_dvol = 1.

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

    if isinstance(sky_distances[0], Quantity):
        assert reduce(lambda x, y: x.value == y.value, sky_distances)
    else:
        assert reduce(lambda x, y: x == y, sky_distances)
    assert len(sky_distances) == len(out_shape) == len(sky_shape) == 2

    # The conversion factor from sky to subpixel
    # (flux = sky_brightness * flux_conversion)
    if sky_as_brightness:
        sky_dvol = 1
    flux_conversion = sub_dvol / sky_dvol

    xy_conversion = 2 * np.pi / np.array(sky_shape)[:, None]

    # TODO: Check why we need the subsample centers swapped.
    # 07-03-25: It seems that the linear & finufft interpolation needs the
    # input points swapped.
    # Maybe: this comes from the matrix style indexing?
    # 16-03-25: Yes, always take matrix style indexing (meshgrid='ij')!

    if indexing == 'ij':
        def rotate_shift_subsample(field, subsample_centers):
            f_field = ifftshift(ifft2(field))
            xy_finufft = xy_conversion * subsample_centers.reshape(2, -1)
            out = nufft2(f_field, xy_finufft[0], xy_finufft[1]).real
            return reshape(out, out_shape) * flux_conversion
    elif indexing == 'xy':
        out_shape = out_shape[1], out_shape[0]

        def rotate_shift_subsample(field, subsample_centers):
            f_field = ifftshift(ifft2(field.T))
            xy_finufft = xy_conversion * subsample_centers.reshape(2, -1)
            out = nufft2(f_field, xy_finufft[1], xy_finufft[0]).real
            return reshape(out, out_shape).T * flux_conversion
    else:
        raise ValueError('Need either provide `ij` or `xy` indexing.')

    return rotate_shift_subsample
