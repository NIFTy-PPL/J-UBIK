import numpy as np

from astropy.coordinates import SkyCoord
from astropy.units import Unit
from astropy import units

from typing import Tuple, Optional
from numpy.typing import ArrayLike

from .wcs.wcs_astropy import build_astropy_wcs


class Grid:
    '''Grid wraps a 2d array with a world coordinate system.'''

    def __init__(
        self,
        center: SkyCoord,
        shape: Tuple[int, int],
        fov: Tuple[Unit, Unit],
        rotation: Unit = 0.0 * units.deg,
    ):
        self.shape = shape
        self.fov = fov
        self.distances = [f.to(units.deg)/s for f, s in zip(fov, shape)]
        self.center = center
        self.wcs = build_astropy_wcs(
            center=center,
            shape=shape,
            fov=(fov[0].to(units.deg), fov[1].to(units.deg)),
            rotation=rotation
        )

    @property
    def dvol(self) -> Unit:
        return self.distances[0] * self.distances[1]

    def world_extrema(
        self,
        extend_factor: float = 1,
        ext: Optional[tuple[int, int]] = None
    ) -> ArrayLike:
        '''The world location of the center of the pixels with the index
        locations = ((0, 0), (0, -1), (-1, 0), (-1, -1))

        Note
        ----
        The indices are assumed to coincide with the convention of the first
        index (x) aligning with the columns and the second index (y) aligning
        with the rows.
        '''
        if ext is None:
            ext0, ext1 = [int(shp*extend_factor-shp)//2 for shp in self.shape]
        else:
            ext0, ext1 = ext

        xmin = -ext0
        xmax = self.shape[0] + ext1  # - 1 FIXME: Which of the two
        ymin = -ext1
        ymax = self.shape[1] + ext1  # - 1
        return self.wcs.wl_from_index([
            (xmin, ymin), (xmin, ymax), (xmax, ymin), (xmax, ymax)])

    def extent(self, unit=units.arcsec):
        '''Convinience method which gives the extent of the grid in physical units.'''
        distances = [d.to(unit).value for d in self.distances]
        shape = self.shape
        halfside = np.array(shape)/2 * np.array(distances)
        return (-halfside[0], halfside[0], -halfside[1], halfside[1])

    def index_grid(self, extend_factor=1, to_bottom_left=True) -> Tuple[ArrayLike, ArrayLike]:
        '''Calculate the index array of the grid.

        Parameters
        ----------
        extend_factor: float
            A factor by which to increase the grid.

        to_bottom_left: bool
            Flag which controlls if the indices of the extended array which are
            relative to the un-extended array co-align such that the (0, 0)
            index of the extended and un-extended array are in the upper left
            corner of the matrix.

            Example:
            unextended = (0, 1, 2)
            extended_centered = (-1, 0, 1, 2, 3)
            extended_bottom_left = (0, 1, 2, 3, -1)
        '''
        extent = [int(s * extend_factor) for s in self.shape]
        extent = [(e - s) // 2 for s, e in zip(self.shape, extent)]
        x, y = [np.arange(-e, s+e) for e, s in zip(extent, self.shape)]
        if to_bottom_left:
            x, y = np.roll(x, -extent[0]), np.roll(y, -extent[1])
        return np.meshgrid(x, y, indexing='xy')

    def wl_coords(self, extend_factor=1, to_bottom_left=True) -> SkyCoord:
        indices = self.index_grid(extend_factor, to_bottom_left=to_bottom_left)
        return self.wcs.wl_from_index([indices])[0]

    def rel_coords(
        self,
        extend_factor=1,
        unit=units.arcsec,
        to_bottom_left=True
    ) -> ArrayLike:
        wl_coords = self.wl_coords(
            extend_factor, to_bottom_left=to_bottom_left)
        r = wl_coords.separation(self.center)
        phi = wl_coords.position_angle(self.center)
        x = r.to(unit) * np.sin(phi.to(units.rad).value)
        y = -r.to(unit) * np.cos(phi.to(units.rad).value)
        return np.array((x.value, y.value))
