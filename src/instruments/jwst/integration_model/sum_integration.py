# SPDX-License-Identifier: BSD-2-Clause
# Authors: Julian Rüstig and Matteo Guardiani

# Copyright(C) 2024 Max-Planck-Society

# %%


def build_sum(high_res_shape: tuple, reduction_factor: int):
    """
    Builds a function that sums over a high-resolution grid, reducing it by a
    specified factor. This can be understood as an integration over the field,
    assuming that the volume factor is part/inside of the pixel value.

    This function reshapes the input data from a high-resolution grid to a
    lower-resolution grid by dividing each axis by the reduction factor. The
    data is then summed over blocks of pixels to create a downsampled version.

    Parameters
    ----------
    high_res_shape : tuple of int
        The shape of the high-resolution grid (height, width).
    reduction_factor : int
        The factor by which to reduce each dimension of the grid.
        It must evenly divide both the height and width of the input grid.

    Returns
    -------
    callable
        A function that, when applied to an array of shape `high_res_shape`,
        reduces its resolution by the specified reduction factor and returns
        the sum over the pixel blocks.

    Raises
    ------
    ValueError
        If the `reduction_factor` does not evenly divide both dimensions
        of `high_res_shape`.

    Example
    -------
    Given an input shape of (100, 100) and a reduction factor of 10,
    the resulting callable reshapes the input into blocks of shape (10, 10)
    and sums them, reducing the resolution to (10, 10).
    """
    if (high_res_shape[0] % reduction_factor != 0) or (
        high_res_shape[1] % reduction_factor != 0
    ):
        raise ValueError("The reduction factor must evenly divide both dimensions")

    new_shape = (
        high_res_shape[0] // reduction_factor,
        reduction_factor,
        high_res_shape[1] // reduction_factor,
        reduction_factor,
    )

    return lambda x: x.reshape(new_shape).sum(axis=(1, 3))
