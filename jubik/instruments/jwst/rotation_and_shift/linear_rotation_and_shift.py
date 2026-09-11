# SPDX-License-Identifier: BSD-2-Clause
# Authors: Julian Rüstig and Matteo Guardiani

# Copyright(C) 2024 Max-Planck-Society

# %%

from functools import partial
from typing import Callable

from jax.scipy.ndimage import map_coordinates
from numpy.typing import ArrayLike


def build_linear_rotation_and_shift(
    out_shape: tuple[int, int],
    order: int = 1,
    mode="wrap",
) -> Callable[ArrayLike, ArrayLike]:
    """
    Builds linear (higher orders not yet supported) rotation_and_shift model.

    Parameters
    ----------
    out_shape: tuple[int, int]
        Expected trailing YX shape of the subsample coordinate grid.
    order: int
        The order of the rotation_and_shift scheme
        (only linear supported by JAX)
    mode: str
        The mode of the interpolation. ['wrap', 'constant']

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

    rotation_and_shift = partial(map_coordinates, order=order, mode=mode)
    out_shape = tuple(out_shape)

    def rotation_shift_subsample(field, subsample_centers_yx):
        actual_shape = tuple(subsample_centers_yx.shape[-2:])
        if actual_shape != out_shape:
            raise ValueError(
                f"subsample_centers_yx trailing shape {actual_shape} "
                f"does not match out_shape {out_shape}"
            )
        return rotation_and_shift(field, subsample_centers_yx)

    return rotation_shift_subsample
