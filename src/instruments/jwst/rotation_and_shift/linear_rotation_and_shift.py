# SPDX-License-Identifier: BSD-2-Clause
# Authors: Julian Rüstig and Matteo Guardiani

# Copyright(C) 2024 Max-Planck-Society

# %%

from functools import partial
from typing import Callable

from jax.scipy.ndimage import map_coordinates
from numpy.typing import ArrayLike


def build_linear_rotation_and_shift(
    sky_dvol: float,
    sub_dvol: float,
    order: int = 1,
    sky_as_brightness: bool = False,
    mode='wrap'
) -> Callable[ArrayLike, ArrayLike]:
    """
    Builds linear (higher orders not yet supported) rotation_and_shift model.

    Parameters
    ----------
    sky_dvol: float
        The volume of the sky/reconstruction pixels
    sub_dvol: float
        The volume of the subsample pixels.
        Typically, the data pixel is subsampled
    order: int
        The order of the rotation_and_shift scheme
        (only linear supported by JAX)
    sky_as_brightness:
        If True, the sky will be treated as a brightness distribution.
        This is the same as setting sky_dvol = 1.
    mode: str
        The mode of the interpolation.

    Returns
    -------
    rotation_shift_subsample : function
        The rotation_and_shift function

    Notes
    -----
    The sky is the reconstruction array, we assume a one-to-one relation
    between the sky brightness and the flux:
        flux(x, y) = sky(x, y) * sky_dvol
    """

    # The conversion factor from sky to subpixel
    # (flux = sky_brightness * flux_conversion)
    if sky_as_brightness:
        sky_dvol = 1
    flux_conversion = sub_dvol / sky_dvol

    rotation_and_shift = partial(map_coordinates, order=order, mode=mode)

    # TODO: Check why we need the subsample centers swapped.
    # 07-03-25: It seems that the linear & finufft interpolation needs the
    # input points swapped.
    # Maybe: this comes from the matrix style indexing?
    def rotation_shift_subsample(field, subsample_centers):
        out = rotation_and_shift(field, subsample_centers)
        # TODO : Strange Transpose
        return out.T * flux_conversion

    return rotation_shift_subsample
