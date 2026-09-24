# SPDX-License-Identifier: BSD-2-Clause
# Authors: Julian Rüstig and Matteo Guardiani

# Copyright(C) 2024 Max-Planck-Society

# %%

import numpy as np
from ...wcs.wcs_base import WcsMixin
from astropy.coordinates import SkyCoord


_HAS_GWCS = False
try:
    from gwcs import WCS

    _HAS_GWCS = True
except ImportError:
    pass


class WcsJwstData(WcsMixin):
    """
    A wrapper around gwcs that provides compatible interface through duck typing.
    """

    def __init__(self, wcs):
        if not _HAS_GWCS:
            raise ImportError(
                "gwcs not installed. Please install via 'pip install gwcs'."
            )

        if not isinstance(wcs, WCS):
            raise TypeError("wcs must be a gwcs.WCS")

        self._wcs = wcs

    # Custom method
    def to_header(self):
        return self._wcs.to_fits()[0]

    def __getattr__(self, name):
        return getattr(self._wcs, name)

    def pixel_to_world(self, *index):
        """Forward pixel coordinates to the wrapped GWCS."""
        return self._wcs.pixel_to_world(*index)

    def world_to_pixel(self, world):
        """Forward world coordinates to the wrapped GWCS."""
        return self._wcs.world_to_pixel(world)

    def world_corners(
        self,
        extension_factor: float = 1,
        extension_value: tuple[int, int] | None = None,
    ) -> np.ndarray:
        xx, yy = self.bounding_box.bounding_box()
        bounds = np.array([[xx[0], xx[0], xx[1], xx[1]], [yy[0], yy[1], yy[1], yy[0]]])
        ra, dec = self.transform("detector", "world", *bounds)
        return SkyCoord(ra=ra, dec=dec, unit="deg")
