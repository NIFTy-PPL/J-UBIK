import numpy as np

from astropy.coordinates import SkyCoord
from astropy.units import Unit
from astropy import units

from typing import Tuple
from numpy.typing import ArrayLike

from .wcs.wcs_astropy import build_astropy_wcs


class ReconstructionGrid:
    def __init__(
        self,
        center: SkyCoord,
        shape: Tuple[int, int],
        fov: Tuple[Unit, Unit]
    ):
        self.shape = shape
        self.distances = [f.to(units.deg)/s for f, s in zip(fov, shape)]
        self.wcs = build_astropy_wcs(
            center, shape, (fov[0].to(units.deg), fov[1].to(units.deg)))

    @property
    def dvol(self) -> Unit:
        return self.distances[0] * self.distances[1]

    @property
    def world_extrema(self) -> ArrayLike:
        return self.wcs.world_of_index_extrema(self.shape)

    def index_grid(self, extend_factor=1) -> Tuple[ArrayLike, ArrayLike]:
        extent = [int(s * extend_factor) for s in self.shape]
        extent = [(e - s) // 2 for s, e in zip(self.shape, extent)]
        x, y = [np.arange(-e, s+e) for e, s in zip(extent, self.shape)]
        x, y = np.roll(x, -extent[0]), np.roll(y, -extent[1])  # to bottom left
        return np.meshgrid(x, y, indexing='xy')
