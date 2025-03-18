# SPDX-License-Identifier: BSD-2-Clause
# Authors: Julian Rüstig and Matteo Guardiani

# Copyright(C) 2024 Max-Planck-Society

# %%

from typing import Tuple

import numpy as np
from jax.experimental.sparse import BCOO
from numpy.typing import ArrayLike
from scipy.sparse import coo_matrix

try:
    from shapely.geometry import Polygon, box
except ImportError:
    print("shapely not installed. Some JWST functions will not work.")
    pass


class _ValueCalculator:
    def __init__(self, coord_grid, points):
        self.data_pixel_polygon = Polygon([p for p in points])
        self.hside = (
            abs(coord_grid[0][0, 1] - coord_grid[0][0, 0]) / 2,
            abs(coord_grid[1][0, 0] - coord_grid[1][1, 0]) / 2,
        )

        minx = min([p[0] for p in points]) - self.hside[0]
        miny = min([p[1] for p in points]) - self.hside[1]
        maxx = max([p[0] for p in points]) + self.hside[0]
        maxy = max([p[1] for p in points]) + self.hside[1]
        mask = (
            (coord_grid[0] >= minx)
            * (coord_grid[0] <= maxx)
            * (coord_grid[1] >= miny)
            * (coord_grid[1] <= maxy)
        )
        self.masked_grid = coord_grid[0][mask], coord_grid[1][mask]

    def get_pixel_extrema(self, pix_cntr):
        """Find the extrema of the pixel."""
        minx, miny = pix_cntr[0] - self.hside[0], pix_cntr[1] - self.hside[1]
        maxx, maxy = pix_cntr[0] + self.hside[0], pix_cntr[1] + self.hside[1]
        return minx, miny, maxx, maxy

    def calculate_values(self, minimum=1e-11):
        """Calculate the fractional area of the data pixel."""
        values = {}
        for pix_cntr in zip(self.masked_grid[0], self.masked_grid[1]):
            minxy_maxxy = self.get_pixel_extrema(pix_cntr)
            sky_pixel_polygon = box(*minxy_maxxy)
            fractional_area = (
                self.data_pixel_polygon.intersection(sky_pixel_polygon).area
                / sky_pixel_polygon.area
            )
            values[pix_cntr] = fractional_area
        return {k: v for k, v in values.items() if v > minimum}


def _get_nearest_index(coord: Tuple[float, float], grid: ArrayLike):
    """Find the index of the nearest grid point to the given
    coordinate."""
    dist_squared = (grid[0] - coord[0]) ** 2 + (grid[1] - coord[1]) ** 2
    return np.unravel_index(np.argmin(dist_squared), grid[0].shape)


def build_sparse_rotation_and_shift(
    index_grid: ArrayLike,
    subsample_corners: ArrayLike,
):
    """
    Builds a sparse interpolation matrix to apply rotation and shift
    transformations to an input grid using subsample-corner information.

    The function calculates a sparse matrix that can later be used to apply a
    rotation and shift operation on a field by performing sparse matrix
    multiplication.
    It uses the corner points of a subsampled grid for precise interpolation.

    Parameters
    ----------
    index_grid : ArrayLike
        The grid of indices representing the input field (e.g., a pixel grid).
    subsample_corners : ArrayLike
        A set of subsampled corners that represent the transformation applied to
        the field. This should have the shape (2, 2, ...) where the third
        dimension corresponds to the data field.

    Returns
    -------
    Callable[[ArrayLike, Any], ArrayLike]
        A function that takes in the field and applies the rotation and shift
        transformation using sparse matrix multiplication.
    """
    print("Calculating sparse interpolation matrix...")
    data_shape = subsample_corners.shape[2:]
    subsample_corners = subsample_corners.reshape(*subsample_corners.shape[:2], -1)

    data_length = subsample_corners[0, 0].size
    assert data_length == subsample_corners[0, 1].size
    assert data_length == subsample_corners[0, 0].shape[0]

    rows = []
    cols = []
    data = []

    for ii, pixel_edges in enumerate(subsample_corners.T):
        vc = _ValueCalculator(index_grid, pixel_edges.T)
        values = vc.calculate_values()

        for (index_x, index_y), val in values.items():
            index_x, index_y = _get_nearest_index((index_x, index_y), index_grid)
            ind = np.ravel_multi_index((index_x, index_y), index_grid[0].shape)
            rows.append(ii)
            cols.append(ind)
            data.append(val)

    mat = coo_matrix((data, (rows, cols)), shape=(data_length, index_grid[0].size))
    sparse_matrix = BCOO.from_scipy_sparse(mat)

    return lambda field, _: (sparse_matrix @ field.reshape(-1)).reshape(data_shape)
