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
            raise ImportError(
                "gwcs not installed. Please install via 'pip install gwcs'."
            )

        if not isinstance(wcs, WCS):
            raise TypeError("wcs must be a gwcs.WCS")

        self.wcs = wcs

    def index_to_world_location(self, index: ArrayLike) -> SkyCoord:
        """
        Convert pixel coordinates to world coordinates.

        Parameters
        ----------
        index : ArrayLike
            Pixel coordinates in the data grid.

        Returns
        -------
        wl : SkyCoord

        Note
        ----
        Since wcs expects pixel indices, the wcs expects (x, y) ordering.
        See https://gwcs.readthedocs.io/en/latest/index.html (last visited 25.03.25).
        """
        return self.wcs(*index, with_units=True)

    def world_location_to_index(self, world: SkyCoord) -> ArrayLike:
        """
        Convert world coordinates to pixel coordinates.

        Parameters
        ----------
        wl : SkyCoord

        Returns
        -------
        index : ArrayLike
        """
        return self.wcs.world_to_pixel(world)

    def to_header(self):
        return self.wcs.to_fits()[0]
