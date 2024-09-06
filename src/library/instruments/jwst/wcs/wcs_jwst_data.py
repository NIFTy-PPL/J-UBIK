import numpy as np

from typing import List, Union
from astropy.coordinates import SkyCoord
from numpy.typing import ArrayLike

from gwcs import WCS

from .wcs_base import WcsBase


class WcsJwstData(WcsBase):
    def __init__(self, wcs: WCS):
        self._wcs = wcs

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
        shp = np.shape(index)
        if (len(shp) == 2) or ((len(shp) == 3) and (shp[0] == 2)):
            return self._wcs(*index, with_units=True)
        return [self._wcs(*p, with_units=True) for p in index]

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
        if isinstance(wl, SkyCoord):
            wl = [wl]
        return np.array([self._wcs.world_to_pixel(w) for w in wl])
