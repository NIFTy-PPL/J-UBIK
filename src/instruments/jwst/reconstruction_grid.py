# SPDX-License-Identifier: BSD-2-Clause
# Authors: Julian Rüstig and Matteo Guardiani

# Copyright(C) 2024 Max-Planck-Society

# %

from typing import Tuple, Optional

import numpy as np
from astropy import units
from astropy.coordinates import SkyCoord
from astropy.units import Unit
from numpy.typing import ArrayLike

from .wcs.wcs_astropy import build_astropy_wcs


class Grid:
    """
    Grid that wraps a 2D array with a world coordinate system (WCS).

    This class represents a grid in the sky with a world coordinate system (WCS)
    centered around a given sky location.
    It provides methods to calculate physical properties of the grid,
    such as distances between pixels, and allows easy access to the world and
    relative coordinates of the grid points.
    """

    def __init__(
        self,
        center: SkyCoord,
        shape: Tuple[int, int],
        fov: Tuple[Unit, Unit],
        rotation: Unit = 0.0 * units.deg,
    ):
        """
        Initialize the Grid with a center, shape, field of view,
        and optional rotation.

        Parameters
        ----------
        center : SkyCoord
            The central sky coordinate of the grid.
        shape : tuple of int
            The shape of the grid, specified as (rows, columns).
        fov : tuple of Unit
            The field of view of the grid in angular units for both axes
            (width, height).
        rotation : Unit, optional
            The rotation of the grid in degrees, counterclockwise from north.
            Default is 0 degrees.
        """
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
        """Computes the area of a grid cell (pixel) in angular units."""
        return self.distances[0] * self.distances[1]

    def world_extrema(
        self,
        extend_factor: float = 1,
        ext: Optional[tuple[int, int]] = None
    ) -> ArrayLike:
        """
        The world location of the center of the pixels with the index
        locations = ((0, 0), (0, -1), (-1, 0), (-1, -1))

        Parameters
        ----------
        extend_factor : float, optional
            A factor by which to extend the grid. Default is 1.
        ext : tuple of int, optional
            Specific extension values for the grid's rows and columns.

        Returns
        -------
        ArrayLike
            The world coordinates of the corner pixels.

        Note
        ----
        The indices are assumed to coincide with the convention of the first
        index (x) aligning with the columns and the second index (y) aligning
        with the rows.
        """
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
        """Convenience method which gives the extent of the grid in
        physical units."""
        distances = [d.to(unit).value for d in self.distances]
        shape = self.shape
        halfside = np.array(shape)/2 * np.array(distances)
        return -halfside[0], halfside[0], -halfside[1], halfside[1]

    def index_grid(
        self,
        extend_factor=1,
        to_bottom_left=True
    ) -> Tuple[ArrayLike, ArrayLike]:
        """
        Compute the grid of indices for the array.

        Parameters
        ----------
        extend_factor : float, optional
            A factor to increase the grid size. Default is 1 (no extension).
        to_bottom_left : bool, optional
            Whether to shift the indices of the extended array such that (0, 0)
            is aligned with the upper left corner of the unextended array.
            Default is True.

        Returns
        -------
        tuple of ArrayLike
            The meshgrid of index arrays for the extended grid.

        Example
        -------
            un_extended = (0, 1, 2)
            extended_centered = (-1, 0, 1, 2, 3)
            extended_bottom_left = (0, 1, 2, 3, -1)
        """
        extent = [int(s * extend_factor) for s in self.shape]
        extent = [(e - s) // 2 for s, e in zip(self.shape, extent)]
        x, y = [np.arange(-e, s+e) for e, s in zip(extent, self.shape)]
        if to_bottom_left:
            x, y = np.roll(x, -extent[0]), np.roll(y, -extent[1])
        return np.meshgrid(x, y, indexing='xy')

    def wl_coords(self, extend_factor=1, to_bottom_left=True) -> SkyCoord:
        """
        Get the world coordinates of the grid points.

        Parameters
        ----------
        extend_factor : float, optional
            A factor by which to extend the grid. Default is 1.
        to_bottom_left : bool, optional
            Whether to align the extended grid indices with the unextended grid.
            Default is True.

        Returns
        -------
        SkyCoord
            The world coordinates of each grid point.
        """
        indices = self.index_grid(extend_factor, to_bottom_left=to_bottom_left)
        return self.wcs.wl_from_index([indices])[0]

    def rel_coords(
        self,
        extend_factor=1,
        unit=units.arcsec,
        to_bottom_left=True
    ) -> ArrayLike:
        """
        Get the relative coordinates of the grid points in a specified unit.

        Parameters
        ----------
        extend_factor : float, optional
            A factor by which to extend the grid. Default is 1.
        unit : Unit, optional
            The physical unit for the output coordinates. Default is arcseconds.
        to_bottom_left : bool, optional
            Whether to align the extended grid indices with the unextended grid.
            Default is True.

        Returns
        -------
        ArrayLike
            A 2D array of relative coordinates (x, y) for each grid point in the
            specified unit.
        """
        wl_coords = self.wl_coords(
            extend_factor, to_bottom_left=to_bottom_left)
        r = wl_coords.separation(self.center)
        phi = wl_coords.position_angle(self.center)
        x = r.to(unit) * np.sin(phi.to(units.rad).value)
        y = -r.to(unit) * np.cos(phi.to(units.rad).value)
        return np.array((x.value, y.value))
