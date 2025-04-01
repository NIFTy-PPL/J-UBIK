# SPDX-License-Identifier: BSD-2-Clause
# Authors: Julian Rüstig and Matteo Guardiani

# Copyright(C) 2024 Max-Planck-Society

# %

from typing import Tuple, Union

import numpy as np
from astropy.coordinates import SkyCoord
from numpy.typing import ArrayLike

from .wcs_jwst_data import WcsJwstData
from .wcs_astropy import WcsAstropy


def subsample_grid_centers_in_index_grid(
    world_corners: Tuple[SkyCoord, SkyCoord, SkyCoord, SkyCoord],
    to_be_subsampled_grid_wcs: Union[WcsAstropy, WcsJwstData],
    index_grid_wcs: Union[WcsAstropy, WcsJwstData],
    subsample: int,
    indexing: str,
) -> ArrayLike:
    """
    This function finds the index positions for the centers of a subsampled
    grid (the to_be_subsampled_grid, typcially the data_grid) inside another
    grid (the index_grid, typically the reconstruction_grid).

    Parameters
    ----------
    world_corners: SkyCoord
        The sky/world positions of the extrema inside which to find the
        subsampling centers.
        Works also if they are outside the grids.
    to_be_subsampled_grid_wcs: WcsBase
        The world coordinate system associated with the grid to be subsampled.
    index_grid_wcs: WcsBase
        The world coordinate system associated with the grid which will be
        indexed into. This will typically be the reconstruction_grid. The
        subsample centers will be in units/indices of this grid.
    subsample:
        The multiplicity of the subsampling along each axis. How many
        sub-pixels will a single pixel in the to_be_subsampled_grid have along
        each axis.
    indexing:
        The indexing convention, either `ij` for matrix style or `xy` for physics style.
    """

    # NOTE : GWCS.wcs expects `xy` indexing. Other arrays are not tested.
    tbsg_pixcenter_indices = to_be_subsampled_grid_wcs.index_grid_from_world_extrema(
        world_corners, indexing="xy"
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

    wl_subsample_centers = to_be_subsampled_grid_wcs.pixel_to_world(*subsample_centers)
    indices_xy = np.array(index_grid_wcs.world_to_pixel(wl_subsample_centers))

    if indexing == "xy":
        return indices_xy
    elif indexing == "ij":
        return indices_xy[::-1]

    raise ValueError("Either `ij` or `xy` indexing.")
