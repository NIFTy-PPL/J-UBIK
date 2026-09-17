# SPDX-License-Identifier: BSD-2-Clause
# Authors: Julian Rüstig and Matteo Guardiani
# Copyright(C) 2024 Max-Planck-Society

# %

from typing import Union

import numpy as np
from astropy.coordinates import SkyCoord
from .wcs_jwst_data import WcsJwstData
from .wcs_astropy import WcsAstropy


def subsample_pixel_centers(
    bounding_indices: tuple[int, int, int, int] | np.ndarray,
    to_be_subsampled_grid_wcs: Union[WcsAstropy, WcsJwstData],
    subsample: int,
    as_pixel_values: bool = False,
) -> SkyCoord | np.ndarray:
    """This function finds the world coordinate centers of a subsampled grid, the
    `to_be_subsampled_grid`, which is typically the data grid.

    Parameters
    ----------
    bounding_indices: tuple[int]
        The min_row, max_row, min_column, max_column indices of the bounding box.
    to_be_subsampled_grid_wcs: Union[WcsAstropy, WcsJwstData]
        The world coordinate system associated with the grid to be subsampled.
    subsample: int
        The multiplicity of the subsampling along each axis. How many
        sub-pixels will a single pixel in the to_be_subsampled_grid have along
        each axis.
    as_pixel_values: bool, optional
        If True, the pixel values of the subsample centers are returned.
        If False, the world coordinates of the subsample centers are returned.

    Returns
    -------
    subsample_centers: SkyCoord | np.ndarray
        The world coordinates or pixel values (if as_pixel_values=True) of subsampled
        pixel centers of the `to_be_subsampled_grid`.
    """

    # NOTE : GWCS.wcs expects `xy` indexing. Other arrays are not tested.
    tbsg_pixcenter_indices = (
        to_be_subsampled_grid_wcs.pixel_grid_xy_from_bounding_indices(
            *bounding_indices
        )
    )

    ps = np.arange(0.5 / subsample, 1, 1 / subsample) - 0.5
    ms = np.vstack(np.array(np.meshgrid(ps, ps, indexing="xy")).T)

    subsample_centers = np.zeros(
        (
            tbsg_pixcenter_indices.shape[0],
            tbsg_pixcenter_indices.shape[1] * subsample,
            tbsg_pixcenter_indices.shape[2] * subsample,
        )
    )
    for ii, ps in enumerate(ms):
        xx = ii % subsample
        yy = ii // subsample
        subsample_centers[:, xx::subsample, yy::subsample] = (
            tbsg_pixcenter_indices + ps[:, None, None]
        )

    if as_pixel_values:
        return subsample_centers

    return to_be_subsampled_grid_wcs.pixel_to_world(*subsample_centers)
