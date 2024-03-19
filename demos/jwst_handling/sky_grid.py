from astropy import units
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.units import Unit

import numpy as np

from numpy.typing import ArrayLike
from typing import Tuple, List


class SkyGrid:
    def __init__(
        self,
        center: SkyCoord,
        shape: Tuple[int, int],
        fov: Tuple[Unit, Unit]
    ):
        self.shape = shape
        self.wcs = self._get_wcs(
            center,
            shape,
            (fov[0].to(units.deg), fov[1].to(units.deg))
        )

    def _get_wcs(
        self,
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

        return w

    @property
    def world_extrema(self) -> ArrayLike:
        return [self.wcs.array_index_to_world(*(0, 0)),
                self.wcs.array_index_to_world(*(self.shape[0], 0)),
                self.wcs.array_index_to_world(*(0, self.shape[1])),
                self.wcs.array_index_to_world(*(self.shape[0], self.shape[1]))]

    def index_grid(self, extend_factor=1) -> Tuple[ArrayLike, ArrayLike]:
        extent = [int(s * extend_factor) for s in self.shape]
        extent = [(e - s) // 2 for s, e in zip(self.shape, extent)]
        x, y = [np.arange(-e, s+e) for e, s in zip(extent, self.shape)]
        x, y = np.roll(x, -extent[0]), np.roll(y, -extent[1])  # to bottom left
        return np.meshgrid(x, y, indexing='xy')

    def indices_from_wl_array(self, wl_array: List[SkyCoord]) -> ArrayLike:
        if isinstance(wl_array, SkyCoord):
            wl_array = [wl_array]
        return np.array([self.wcs.world_to_pixel(wl) for wl in wl_array])
