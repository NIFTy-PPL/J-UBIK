# SPDX-License-Identifier: BSD-2-Clause
# Authors: Julian Rüstig and Matteo Guardiani

# Copyright(C) 2024 Max-Planck-Society

# %%

from typing import List, Union

import numpy as np
from astropy.coordinates import SkyCoord
from numpy.typing import ArrayLike

from .wcs_base import WcsBase


class WcsJwstData(WcsBase):
    """
    A wrapper around the gwcs, in order to define a common interface
    with the astropy wcs.
    """

    def __init__(self, wcs):
        try:
            from gwcs import WCS
        except ImportError:
            raise ImportError("gwcs not installed."
                              "Please install via 'pip install gwcs'.")

        if not isinstance(wcs, WCS):
            raise TypeError('wcs must be a gwcs.WCS')

        self.wcs = wcs

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
        shp = np.shape(index)
        if (len(shp) == 2) or ((len(shp) == 3) and (shp[0] == 2)):
            return self.wcs(*index, with_units=True)
        return [self.wcs(*p, with_units=True) for p in index]

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
        if isinstance(wl, SkyCoord):
            wl = [wl]
        return np.array([self.wcs.world_to_pixel(w) for w in wl])

    def to_header(self):
        return self.wcs.to_fits()[0]
