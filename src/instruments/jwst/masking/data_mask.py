from typing import Union

import numpy as np
from astropy.coordinates import SkyCoord

from ....wcs.wcs_astropy import WcsAstropy
from ....wcs.wcs_jwst_data import WcsJwstData
from ....wcs.wcs_subsample_centers import (
    subsample_pixel_centers,
    world_coordinates_to_index_grid,
)


def get_mask_from_index_centers_within_rgrid(
    bounding_box_xmin_xmax_ymin_ymax: tuple[int] | np.ndarray,
    data_wcs: WcsJwstData,
    index_grid_wcs: WcsAstropy | None,
) -> np.ndarray:
    """Get mask which is true where the index grid falls within the data grid.

    Parameters
    ----------
    bounding_box_xmin_xmax_ymin_ymax: tuple[int] | np.ndarray
        The pixel postions edges of the bounding box, inside the data (corresponding
        from the data_wcs).
        Previously cacuclated by,
        data_wcs.bounding_box_indices_from_world_extrema(world_corners).
    data_wcs: WcsJwstData
    index_grid_wcs: WcsAstropy | None
    """

    data_pix_centers_world_coordinates = subsample_pixel_centers(
        bounding_box_xmin_xmax_ymin_ymax,
        data_wcs,
        subsample=1,
    )
    if index_grid_wcs is None:
        return np.full(data_pix_centers_world_coordinates.shape, True)

    data_pix_centers_in_index_grid = world_coordinates_to_index_grid(
        data_pix_centers_world_coordinates, index_grid_wcs=index_grid_wcs, indexing="xy"
    )

    return (
        (data_pix_centers_in_index_grid[0] > 0)
        * (data_pix_centers_in_index_grid[1] > 0)
        * (data_pix_centers_in_index_grid[0] < index_grid_wcs.shape[0])
        * (data_pix_centers_in_index_grid[1] < index_grid_wcs.shape[1])
    )


def get_mask_from_mask_corners(
    data_shape: tuple[int, int] | np.ndarray,
    data_wcs: Union[WcsJwstData, WcsAstropy],
    bounding_box_minmax_row_column: tuple[int] | np.ndarray,
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
    world_corners: SkyCoord
        The world corners of the reconstruction grid.
    mask_corners: SkyCoord
        The world corners of the mask.
    """

    min_y, _, min_x, _ = bounding_box_minmax_row_column
    centers = subsample_pixel_centers(
        data_wcs.bounding_indices_from_world_extrema(mask_corners),
        data_wcs,
        subsample=1,
        as_pixel_values=True,
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
