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
            reconstruction grid edges.
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

    def wl_pixelcenter_and_edges(
        self,
        extrema: SkyCoord,
        array_shape: Optional[Tuple[int, int]] = None,
    ) -> Tuple[SkyCoord, Tuple[SkyCoord, SkyCoord, SkyCoord, SkyCoord]]:
        '''Find the world locations of the pixel centers and edges of the data
        grid inside the extrema.

        Parameters
        ----------
        extrema : List[SkyCoord]
            List of SkyCoord objects, representing the world location of the
            reconstruction grid edges.

        array_shape: tuple
            When provided the extremas are checked for consistency with the
            underlying (data) grid. In case the extrema fall outside the (data)
            array's domain, the algorithm throws a ValueError.

        Returns
        -------
        pix_center : SkyCoord
            pixel centers of the data grid in global wcs

        e00, e01, e10, e11 : SkyCoord
            pixel edges of the data grid in global wcs

        '''
        minx, maxx, miny, maxy = self.index_from_wl_extrema(extrema)

        pix_center = np.meshgrid(np.arange(minx, maxx, 1),
                                 np.arange(miny, maxy, 1))
        e00 = pix_center - np.array([0.5, 0.5])[:, None, None]
        e01 = pix_center - np.array([0.5, -0.5])[:, None, None]
        e10 = pix_center - np.array([-0.5, 0.5])[:, None, None]
        e11 = pix_center - np.array([-0.5, -0.5])[:, None, None]

        return (self.wl_from_index([pix_center])[0],
                self.wl_from_index([e00, e01, e10, e11]))

    def wl_subsample_centers(
        self,
        extrema: Tuple[SkyCoord, SkyCoord, SkyCoord, SkyCoord],
        subsample: int,
        array_shape: Optional[Tuple[int, int]] = None,
    ) -> List[SkyCoord]:
        '''Find the location of (subsampled) pixel centers inside the extremas.

        Parameters
        ----------
        extrema : Tuple[SkyCoord]
            World locations of the extrema for the grid.

        subsample : int
            subsample factor of the pixel grid

        array_shape: tuple
            shape of the underlying array on which we find the subsample
            centers. If no array shape is provided the pixel centers are not
            checked for consistency with the underlying grid.


        Returns
        -------
        pix_center : SkyCoord
            pixel centers of the data grid in global wcs

        '''
        minx, maxx, miny, maxy = self.index_from_wl_extrema(
            extrema, array_shape)

        pix_center = np.array(np.meshgrid(np.arange(minx, maxx, 1),
                                          np.arange(miny, maxy, 1)))
        ps = np.arange(0.5/subsample, 1, 1/subsample) - 0.5
        ms = np.vstack(np.array(np.meshgrid(ps, ps)).T)
        subsample_centers = ms[:, :, None, None] + pix_center

        return self.wl_from_index(subsample_centers)
