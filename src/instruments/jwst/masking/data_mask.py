from typing import Union

import numpy as np
from astropy.coordinates import SkyCoord

from ....wcs.wcs_astropy import WcsAstropy
from ....wcs.wcs_jwst_data import WcsJwstData
from ....wcs.wcs_subsample_centers import subsample_grid_centers_in_index_grid


def get_mask_from_index_centers_within_rgrid(
    world_corners: SkyCoord,
    data_wcs: WcsJwstData,
    index_grid_wcs: WcsAstropy,
) -> np.ndarray:
    """Get mask which is true where the index grid falls within the data grid.

    Parameters
    ----------
    data_pix_centers_in_index_grid: np.ndarray
        The data-pixel-centers in units of rgrid pixels.
    rgrid_shape: tuple[int, int]
        The shape of rgrid.
    """

    data_pix_centers_in_index_grid: np.ndarray = subsample_grid_centers_in_index_grid(
        world_corners=world_corners,
        to_be_subsampled_grid_wcs=data_wcs,
        index_grid_wcs=index_grid_wcs,
        subsample=1,
        indexing="xy",
    )

    return (
        (data_pix_centers_in_index_grid[0] > 0)
        * (data_pix_centers_in_index_grid[1] > 0)
        * (data_pix_centers_in_index_grid[0] < index_grid_wcs.shape[0])
        * (data_pix_centers_in_index_grid[1] < index_grid_wcs.shape[1])
    )


def get_mask_from_mask_corners(
    data_shape: tuple[int],
    data_wcs: Union[WcsJwstData, WcsAstropy],
    world_corners: tuple[SkyCoord],
    mask_corners: tuple[SkyCoord],
):
    """Build a mask that masks the pixels inside the data which are bounded by the
    `mask_corners`.

    Paramters
    ---------
    data_shape: tuple[int]
        The shape of the data.
    data_wcs: Union[WcsJwstData, WcsAstropy]
        The wcs of the data.
    world_corners:

    """
    min_x, _, min_y, _ = data_wcs.bounding_box_indices_from_world_extrema(world_corners)
    centers = subsample_grid_centers_in_index_grid(
        world_corners=mask_corners,
        to_be_subsampled_grid_wcs=data_wcs,
        index_grid_wcs=data_wcs,
        subsample=1,
        indexing="xy",
    )
    centers = np.array(centers, dtype=int) - np.array((min_x, min_y))[:, None, None]
    min_x_mask, max_x_mask, min_y_mask, max_y_mask = (
        centers[0].min(),
        centers[0].max(),
        centers[1].min(),
        centers[1].max(),
    )

    mask = np.full(data_shape, False, dtype=bool)
    mask[min_y_mask:max_y_mask, min_x_mask:max_x_mask] = True

    return mask
