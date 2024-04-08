import numpy as np

from numpy.typing import ArrayLike
from typing import List, Union, Tuple

from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.units import Unit
from astropy import units

from .wcs_base import WcsBase


class WcsAstropy(WcsBase):
    def __init__(self, wcs):
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
        if len(np.shape(index)) == 1:
            index = [index]
        return [self._wcs.array_index_to_world(*idx) for idx in index]

    def index_from_wl(self, wl_array: List[SkyCoord]) -> ArrayLike:
        '''Convert world coordinates to pixel coordinates.

        Parameters
        ----------
        wl : SkyCoord

        Returns
        -------
        index : ArrayLike
        '''
        if isinstance(wl_array, SkyCoord):
            wl_array = [wl_array]
        return np.array([self._wcs.world_to_pixel(wl) for wl in wl_array])


def build_astropy_wcs(
    center: SkyCoord,
    shape: Tuple[int, int],
    fov: Tuple[Unit, Unit]
) -> WCS:

    # Create a WCS object
    w = WCS(naxis=2)

    # Set up ICRS system
    w.wcs.crpix = [shape[0] / 2, shape[1] / 2]
    w.wcs.cdelt = [-fov[0].to(units.deg).value / shape[0],
                   fov[1].to(units.deg).value / shape[1]]
    w.wcs.crval = [center.ra.deg, center.dec.deg]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]

    return WcsAstropy(w)
