# SPDX-License-Identifier: BSD-2-Clause
# Authors: Julian Rüstig and Matteo Guardiani

# Copyright(C) 2024 Max-Planck-Society

# %%

from typing import Optional
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord, Angle
from numpy.typing import ArrayLike


class WcsBase:
    """An interface class for converting between world coordinates and pixel
    coordinates. Child classes need to provide a `pixel_to_world` and a
    `world_to_pixel` method."""

    def pixel_to_world(self, *index: ArrayLike) -> SkyCoord:
        """
        Convert pixel coordinates to world coordinates.

        Parameters
        ----------
        index : ArrayLike
            Pixel coordinates in the data grid.

        Returns
        -------
        wl : SkyCoord
        """
        pass

    def world_to_pixel(self, world: SkyCoord) -> ArrayLike:
        """
        Convert world coordinates to pixel coordinates.

        Parameters
        ----------
        wl : SkyCoord

        Returns
        -------
        index : ArrayLike
        """
        pass


class WcsMixin:
    """A mixin class providing WCS functionality, assuming pixel_to_world and
    world_to_pixel are implemented."""

    def bounding_indices_from_world_extrema(
        self, world_extrema: SkyCoord, shape_check: Optional[tuple[int, int]] = None
    ) -> tuple[int, int, int, int]:
        """Find the pixels (edges of the pixels) of the bounding box that contains the
        `world_extrema`.

        Parameters
        ----------
        world_extrema : List[SkyCoord]
            List of SkyCoord objects, representing the world location of the
            reconstruction grid corners.
        shape_check : Optional[Tuple[int, int]]
            When provided the world_extrema are checked for consistency with
            the underlying data array.

        Returns
        -------
        min_row, max_row, min_column, max_column: Tuple[int, int, int, int]
            The array slice for the grid, corresponding to the corners of a grid,
            typically the data grid, which contain the `world_extrema`.

        Note
        ----
        row = y, column = x
        """

        edge_points = np.array([self.world_to_pixel(wex) for wex in world_extrema])

        if shape_check is not None:
            check = (
                np.any(edge_points < 0)
                or np.any(edge_points >= shape_check[0])
                or np.any(edge_points >= shape_check[1])
            )
            if check:
                o = f"""One of the wcs world_extrema is outside the data grid
                {edge_points}"""
                raise ValueError(o)

        min_column = int(np.floor(np.min(edge_points[:, 0])))
        max_column = int(np.ceil(np.max(edge_points[:, 0])))
        min_row = int(np.floor(np.min(edge_points[:, 1])))
        max_row = int(np.ceil(np.max(edge_points[:, 1])))
        return min_row, max_row, min_column, max_column

    def index_grid_from_bounding_indices(
        self,
        min_row: int,
        max_row: int,
        min_column: int,
        max_column: int,
        indexing: str,
    ) -> np.ndarray:
        """Return index array from the bounding indices.

        Paramaters
        ----------
        min_column: int
        max_column: int
        min_row: int
        max_row: int
        indexing: str
            Either `xy` or `ij`.
            Note astropy.wcs uses always xy indexing, hence, the shapes have to be
            according to this.
        """

        x_indices = np.arange(min_column, max_column + 1)
        y_indices = np.arange(min_row, max_row + 1)

        if indexing == "xy":
            return np.array(np.meshgrid(x_indices, y_indices, indexing="xy"))
        elif indexing == "ij":
            return np.array(np.meshgrid(y_indices, x_indices, indexing="ij"))

        raise ValueError("Either `ij` or `xy` indexing.")
