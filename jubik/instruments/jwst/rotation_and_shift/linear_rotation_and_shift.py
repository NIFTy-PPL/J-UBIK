# SPDX-License-Identifier: BSD-2-Clause
# Authors: Julian Rüstig and Matteo Guardiani

# Copyright(C) 2024 Max-Planck-Society

# %%

from functools import partial
from typing import Callable

import jax.numpy as jnp
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
        The interpolation function ``(field, subsample_centers_yx) -> out``.
        ``subsample_centers_yx`` has shape ``(2, *out_shape)`` or, for stacked
        exposures, ``(n, 2, *out_shape)``; ``out`` drops the coordinate axis.

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
        # (..., 2, ny, nx) -> (2, ..., ny, nx); map_coordinates keeps the leading shape
        return rotation_and_shift(field, jnp.moveaxis(subsample_centers_yx, -3, 0))

    return rotation_shift_subsample
