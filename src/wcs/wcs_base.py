# SPDX-License-Identifier: BSD-2-Clause
# Authors: Julian Rüstig and Matteo Guardiani

# Copyright(C) 2024 Max-Planck-Society

# %%

from abc import ABC, abstractmethod
from typing import List, Union, Tuple, Optional

import numpy as np
from astropy.coordinates import SkyCoord
from numpy.typing import ArrayLike


class WcsBase(ABC):
    """An interface class for converting between world coordinates and pixel
    coordinates. Inherited classes need to provide a `wl_from_index` and an
    `index_from_wl` method.
    """
    @abstractmethod
    def wl_from_index(
        self, index: ArrayLike
    ) -> Union[SkyCoord, List[SkyCoord]]:
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

    @abstractmethod
    def index_from_wl(
        self, wl: Union[SkyCoord, List[SkyCoord]]
    ) -> Union[ArrayLike, List[ArrayLike]]:
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

    def index_from_wl_extrema(
        self,
        world_extrema: SkyCoord,
        shape_check: Optional[Tuple[int, int]] = None
    ) -> Tuple[int, int, int, int]:
        """
        Find the minimum and maximum pixel indices of the bounding box that
        contain the world location world_extrema (wl_extrema).

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
        minx, maxx, miny, maxy : Tuple[int, int, int, int]
            Minimum and maximum pixel coordinates of the data grid that contain
            the edge points.
        """

        edges_dgrid = self.index_from_wl(world_extrema)

        if shape_check is not None:
            check = (
                np.any(edges_dgrid < 0) or
                np.any(edges_dgrid >= shape_check[0]) or
                np.any(edges_dgrid >= shape_check[1])
            )
            if check:
                o = f"""One of the wcs world_extrema is outside the data grid
                {edges_dgrid}"""
                raise ValueError(o)

        # FIXME: What to do here... round, ceil, or floor?
        minx = int(np.round(edges_dgrid[:, 0].min()))
        maxx = int(np.round(edges_dgrid[:, 0].max()))
        miny = int(np.round(edges_dgrid[:, 1].min()))
        maxy = int(np.round(edges_dgrid[:, 1].max()))
        return minx, maxx, miny, maxy

    def index_grid_from_wl_extrema(
        self,
        world_extrema: SkyCoord,
        shape_check: Optional[Tuple[int, int]] = None
    ) -> np.typing.ArrayLike:
        """
        Find the pixel indices of the bounding box that contain the world
        location world_extrema (wl_extrema).

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
        minx, max, miny, maxy : Tuple[int, int, int, int]
            Minimum and maximum pixel coordinates of the data grid that contain
            the edge points.
        """

        minx, maxx, miny, maxy = self.index_from_wl_extrema(
            world_extrema, shape_check)

        return np.array(np.meshgrid(np.arange(minx, maxx, 1),
                                    np.arange(miny, maxy, 1)))
