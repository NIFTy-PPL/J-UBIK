from abc import ABC, abstractmethod

from typing import List, Union, Tuple, Optional
from astropy.coordinates import SkyCoord
from numpy.typing import ArrayLike

import numpy as np


class WcsBase(ABC):
    def __init__(self, wcs):
        self._wcs = wcs

    @abstractmethod
    def wl_from_index(
        self, index: ArrayLike
    ) -> Union[SkyCoord, List[SkyCoord]]:
        '''Convert pixel coordinates to world coordinates.

        Parameters
        ----------
        index : ArrayLike
            Pixel coordinates in the data grid.

        Returns
        -------
        wl : SkyCoord
        '''
        pass

    @abstractmethod
    def index_from_wl(
        self, wl: Union[SkyCoord, List[SkyCoord]]
    ) -> Union[ArrayLike, List[ArrayLike]]:
        '''Convert world coordinates to pixel coordinates.

        Parameters
        ----------
        wl : SkyCoord

        Returns
        -------
        index : ArrayLike
        '''
        pass

    def index_from_wl_extrema(
        self,
        extrema: SkyCoord,
        shape: Optional[Tuple[int, int]] = None
    ) -> Tuple[int, int, int, int]:
        '''Find the minimum and maximum pixel indices of the bounding box that
        contain the world location extrema (wl_extrema).

        Parameters
        ----------
        extrema : List[SkyCoord]
            List of SkyCoord objects, representing the world location of the
            reconstruction grid corners.
        shape : Optional[Tuple[int, int]]
            When provided the extrema are checked for consistency with the
            underlying data array.

        Returns
        -------
        minx, maxx, miny, maxy : Tuple[int, int, int, int]
            Minimum and maximum pixel coordinates of the data grid that contain
            the edge points.
        '''

        edges_dgrid = self.index_from_wl(extrema)

        if shape is not None:
            check = (
                np.any(edges_dgrid < 0) or
                np.any(edges_dgrid >= shape[0]) or
                np.any(edges_dgrid >= shape[1])
            )
            if check:
                o = f"""One of the wcs extrema is outside the data grid
                {edges_dgrid}"""
                raise ValueError(o)

        # FIXME: What to do here... round, ceil, or floor?
        minx = int(np.round(edges_dgrid[:, 0].min()))
        maxx = int(np.round(edges_dgrid[:, 0].max()))
        miny = int(np.round(edges_dgrid[:, 1].min()))
        maxy = int(np.round(edges_dgrid[:, 1].max()))
        return minx, maxx, miny, maxy
