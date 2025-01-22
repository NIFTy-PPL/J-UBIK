# SPDX-License-Identifier: BSD-2-Clause
# Authors: Julian Rüstig and Matteo Guardiani

# Copyright(C) 2024 Max-Planck-Society

# %%
from .wcs_base import WcsBase
from ..parse.wcs.coordinate_system import (
    CoordinateSystemModel, CoordinateSystems)
from ..parse.wcs.wcs_astropy import WcsModel

import numpy as np

from numpy.typing import ArrayLike
from typing import List, Union, Optional

from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy import units as u


class WcsAstropy(WCS, WcsBase):
    """
    A wrapper around the astropy wcs, in order to define a common interface
    with the gwcs.
    """

    def __init__(
        self,
        center: SkyCoord,
        shape: tuple[int, int],
        fov: tuple[u.Quantity, u.Quantity],
        rotation: u.Quantity = 0.0 * u.deg,
        coordinate_system: Optional[CoordinateSystemModel] = CoordinateSystems.icrs.value
    ):
        """
        Create FITS header, use it to instantiate an WcsAstropy.

        Parameters
        ----------
        center : SkyCoord
            The value of the center of the coordinate system (crval).
        shape : tuple
            The shape of the grid.
        fov : tuple
            The field of view of the grid. Typically given in degrees.
        rotation : u.Quantity
            The rotation of the grid WCS, in degrees.
        coordinate_system : CoordinateSystemConfig
            Coordinate system to use ('icrs', 'fk5', 'fk4', 'galactic')
        equinox : float, optional
            Equinox for FK4/FK5 systems (e.g., 2000.0 for J2000)
        """

        self.shape = shape
        self.fov = fov
        self.distances = [
            f.to(u.deg)/s for f, s in zip(fov, shape)]
        self.center = center

        # Calculate rotation matrix
        rotation_value = rotation.to(u.rad).value
        pc11 = np.cos(rotation_value)
        pc12 = -np.sin(rotation_value)
        pc21 = np.sin(rotation_value)
        pc22 = np.cos(rotation_value)

        # Transform center coordinates if necessary
        if coordinate_system.radesys == CoordinateSystems.galactic.value.radesys:
            lon = center.galactic.l.deg
            lat = center.galactic.b.deg
        else:
            lon = center.ra.deg
            lat = center.dec.deg

        # Build the header dictionary
        header = {
            'WCSAXES': 2,
            'CTYPE1': coordinate_system.ctypes[0],
            'CTYPE2': coordinate_system.ctypes[1],
            'CRPIX1': shape[0] / 2 + 0.5,
            'CRPIX2': shape[1] / 2 + 0.5,
            'CRVAL1': lon,
            'CRVAL2': lat,
            'CDELT1': -fov[0].to(u.deg).value / shape[0],
            'CDELT2': fov[1].to(u.deg).value / shape[1],
            'PC1_1': pc11,
            'PC1_2': pc12,
            'PC2_1': pc21,
            'PC2_2': pc22,
            'RADESYS': coordinate_system.radesys,
            'CUNIT1': 'deg',
            'CUNIT2': 'deg',
        }

        # Set equinox if needed for FK4/FK5
        if coordinate_system.radesys in [
                CoordinateSystems.fk4.value.radesys,
                CoordinateSystems.fk5.value.radesys]:
            header['EQUINOX'] = coordinate_system.equinox

        super().__init__(header)

    @classmethod
    def from_wcs_model(cls, wcs_model: WcsModel):
        return WcsAstropy(
            wcs_model.center,
            wcs_model.shape,
            wcs_model.fov,
            wcs_model.rotation,
            wcs_model.coordinate_system,
        )

    # TODO: Check output axis, RENAME index_from_world_location
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

        Note
        ----
        We use the convention of x aligning with the columns, second dimension,
        and y aligning with the rows, first dimension.
        """
        if len(np.shape(index)) == 1:
            index = [index]
        return [self.pixel_to_world(*idx) for idx in index]
        # return [self.array_index_to_world(*idx) for idx in index]

    # TODO: Check output axis, RENAME index_from_world_location
    def index_from_wl(self, wl_array: List[SkyCoord]) -> ArrayLike:
        """
        Convert world coordinates to pixel coordinates.

        Parameters
        ----------
        wl : SkyCoord

        Returns
        -------
        index : ArrayLike

        Note
        ----
        We use the convention of x aligning with the columns, second dimension,
        and y aligning with the rows, first dimension.
        """
        if isinstance(wl_array, SkyCoord):
            wl_array = [wl_array]
        return np.array([self.world_to_pixel(wl) for wl in wl_array])

    @property
    def dvol(self) -> u.Quantity:
        """Computes the area of a grid cell (pixel) in angular u."""
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
            ext0, ext1 = [int(shp*extend_factor-shp) //
                          2 for shp in self.shape]
        else:
            ext0, ext1 = ext

        xmin = -ext0
        xmax = self.shape[0] + ext1  # - 1 FIXME: Which of the two
        ymin = -ext1
        ymax = self.shape[1] + ext1  # - 1
        return self.wl_from_index([
            (xmin, ymin), (xmin, ymax), (xmax, ymin), (xmax, ymax)])

    def index_grid(
        self,
        extend_factor=1,
        to_bottom_left=True
    ) -> tuple[ArrayLike, ArrayLike]:
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

    def distances_in(self, unit: u.Unit) -> list[float]:
        return [d.to(unit).value for d in self.distances]

    def extent(self, unit=u.arcsec):
        """Convenience method which gives the extent of the grid in
        physical units."""
        distances = [d.to(unit).value for d in self.distances]
        halfside = np.array(self.shape)/2 * np.array(distances)
        return -halfside[0], halfside[0], -halfside[1], halfside[1]

    def relative_coordinates(
        self,
        reference_point: Optional[SkyCoord] = None,
        unit: u.Unit = u.arcsec,
    ) -> ArrayLike:
        """Calculate relative angular offsets from a reference point on the
        sky.

        Computes the spherical offsets between each pixel's world coordinates
        and a reference position. The offsets are measured in the longitudinal
        (x) and latitudinal (y) directions using proper spherical geometry.

        Parameters
        ----------
        reference_point : astropy.coordinates.SkyCoord, optional
            Reference position on the sky to measure offsets from.
            If None, uses `self.center`.
        unit : astropy.units.Unit, optional
            Unit for the output coordinates. Default is arcseconds.
            Must be an angular unit (e.g., degree, arcmin, radian).

        Returns
        -------
        numpy.ndarray
            Array of shape (2, ny, nx) containing the relative coordinates,
            where axis 0 contains:
                [0]: longitudinal (x) offsets
                [1]: latitudinal (y) offsets
            All offsets are in the specified unit.
        """

        coordinates_world = self.pixel_to_world(*self.index_grid())

        if reference_point is None:
            reference_point = self.center

        dlon, dlat = coordinates_world.spherical_offsets_to(reference_point)
        return np.stack([dlon.to(unit).value, dlat.to(unit).value])


def WcsAstropy_from_wcs(wcs: WCS) -> WcsAstropy:
    """
    Get basic WCS information: center coordinates, field of view, and array shape.

    Parameters:
    -----------
    wcs : astropy.wcs.WCS
        The WCS object to analyze

    Returns:
    --------
    center : SkyCoord
        Center/reference pixel coordinates
    fov : tuple[Quantity, Quantity]
        Field of view in (width, height)
    shape : list[int]
        Shape of the array (nx, ny)
    """
    # Get array shape
    nx, ny = wcs.array_shape

    # Get center coordinate
    center = SkyCoord(
        wcs.wcs.crval[0], wcs.wcs.crval[1],
        unit='deg', frame=wcs.wcs.radesys.lower())

    # Calculate FOV
    corners_pix = np.array([[0, 0], [nx-1, 0], [nx-1, ny-1], [0, ny-1]])
    corners_world = wcs.wcs_pix2world(corners_pix, 0)
    corners = SkyCoord(corners_world, unit='deg',
                       frame=wcs.wcs.radesys.lower())

    # Calculate width (average of top and bottom)
    width = (corners[0].separation(corners[1]) +
             corners[3].separation(corners[2])) / 2

    # Calculate height (average of left and right)
    height = (corners[0].separation(corners[3]) +
              corners[1].separation(corners[2])) / 2

    return WcsAstropy(center, [nx, ny], (width, height))
