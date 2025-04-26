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

    def bounding_box_indices_from_world_extrema(
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
        min_x, max_x, min_y, max_y : Tuple[int, int, int, int]
            The corners of the pixels of the (typically data) grid that contains the
            `world_extrema`.
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

        # TODO : Is round the best ?
        min_x = int(np.floor(np.min(edge_points[:, 0])))
        max_x = int(np.ceil(np.max(edge_points[:, 0])))
        min_y = int(np.floor(np.min(edge_points[:, 1])))
        max_y = int(np.ceil(np.max(edge_points[:, 1])))
        return min_x, max_x, min_y, max_y

    def bounding_box_index_grid_from_world_extrema(
        self,
        world_extrema: SkyCoord,
        indexing: str,
        shape_check: tuple[int, int] | None = None,
    ) -> ArrayLike:
        """
        Find the pixel indices of the bounding box that contain the world
        location world_extrema (wl_extrema).

        Parameters
        ----------
        world_extrema : List[SkyCoord]
            List of SkyCoord objects, representing the world location of the
            reconstruction grid corners.
        indexing: str
            Which indexing format for meshgrid, either 'ij' or 'xy'.
        shape_check : Optional[tuple[int, int]]
            When provided the world_extrema are checked for consistency with
            the underlying data array.

        Returns
        -------
        min_x, max, min_y, max_y : tuple[int, int, int, int]
            Minimum and maximum pixel coordinates of the data grid that contain
            the edge points.
        """

        min_x, max_x, min_y, max_y = self.bounding_box_indices_from_world_extrema(
            world_extrema, shape_check
        )

        x_indices = np.arange(min_x, max_x + 1)
        y_indices = np.arange(min_y, max_y + 1)
        xy = np.array(np.meshgrid(x_indices, y_indices, indexing="xy"))

        if indexing == "xy":
            return xy
        elif indexing == "ij":
            return xy[::-1]

        raise ValueError("Either `ij` or `xy` indexing.")
