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


def subsample_pixel_centers(
    bounding_indices: tuple[int, int, int, int],
    to_be_subsampled_grid_wcs: Union[WcsAstropy, WcsJwstData],
    subsample: int,
    as_pixel_values: bool = False,
) -> SkyCoord:
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
    tbsg_pixcenter_indices = to_be_subsampled_grid_wcs.index_grid_from_bounding_indices(
        *bounding_indices, indexing="xy"
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


def world_coordinates_to_index_grid(
    world_coordinates: Union[SkyCoord, list[SkyCoord]],
    index_grid_wcs: Union[WcsAstropy, WcsJwstData],
    indexing: str,
):
    """Transform world coordinates into pixels coordinates in the index grid.
    Subsequently, these pixel coordinates can be used to interpolate the values that
    live on the index grid onto the world coordinates.

    Parameters
    ----------
    world_coordinates: SkyCoord
        The world coordinates of pixels or subpixels.
    index_grid_wcs: Union[WcsAstropy, WcsJwstData],
        The wcs of the index grid. This is needed in order to find out where the world
        coordinates fall into in the index grid.
    indexing: str
        The index convention used. Either `ij` or `xy` indexing.
    """

    if isinstance(world_coordinates, SkyCoord):
        indices_xy = np.array(index_grid_wcs.world_to_pixel(world_coordinates))
    elif isinstance(world_coordinates, list):
        indices_xy = np.array(
            [index_grid_wcs.world_to_pixel(wc) for wc in world_coordinates]
        )
    else:
        raise ValueError(
            "`world_coordinates` should either be SkyCoord or list[SkyCoord]."
            f"\ntype(world_coordinates)={type(world_coordinates)}"
        )

    if indexing == "xy":
        return indices_xy
    elif indexing == "ij":
        return indices_xy[..., ::-1, :, :]

    raise ValueError("Either `ij` or `xy` indexing.")
