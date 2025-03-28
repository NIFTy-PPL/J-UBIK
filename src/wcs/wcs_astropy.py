# SPDX-License-Identifier: BSD-2-Clause
# Authors: Julian Rüstig and Matteo Guardiani

# Copyright(C) 2024 Max-Planck-Society

# %%
from .wcs_base import WcsBase
from ..parse.wcs.coordinate_system import CoordinateSystemModel, CoordinateSystems
from ..parse.wcs.spatial_model import SpatialModel

import numpy as np

from numpy.typing import ArrayLike
from typing import List, Union, Optional

from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy import units as u


class WcsAstropy(WCS, WcsBase):
    """
    A wrapper around the astropy.wcs.WCS, in order to define a common interface
    with the gwcs.
    """

    def __init__(
        self,
        center: SkyCoord,
        shape: tuple[int, int],
        fov: tuple[u.Quantity, u.Quantity],
        rotation: u.Quantity = 0.0 * u.deg,
        coordinate_system: Optional[
            CoordinateSystemModel
        ] = CoordinateSystems.icrs.value,
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
        self.distances = [f.to(u.deg) / s for f, s in zip(fov, shape)]
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
            "WCSAXES": 2,
            "CTYPE1": coordinate_system.ctypes[0],
            "CTYPE2": coordinate_system.ctypes[1],
            "CRPIX1": shape[0] / 2 + 0.5,
            "CRPIX2": shape[1] / 2 + 0.5,
            "CRVAL1": lon,
            "CRVAL2": lat,
            "CDELT1": -fov[0].to(u.deg).value / shape[0],
            "CDELT2": fov[1].to(u.deg).value / shape[1],
            "PC1_1": pc11,
            "PC1_2": pc12,
            "PC2_1": pc21,
            "PC2_2": pc22,
            "RADESYS": coordinate_system.radesys,
            "CUNIT1": "deg",
            "CUNIT2": "deg",
        }

        # Set equinox if needed for FK4/FK5
        if coordinate_system.radesys in [
            CoordinateSystems.fk4.value.radesys,
            CoordinateSystems.fk5.value.radesys,
        ]:
            header["EQUINOX"] = coordinate_system.equinox

        super().__init__(header)

    @classmethod
    def from_spatial_model(cls, spatial_model: SpatialModel):
        return WcsAstropy(
            spatial_model.wcs_model.center,
            spatial_model.shape,
            spatial_model.fov,
            spatial_model.wcs_model.rotation,
            spatial_model.wcs_model.coordinate_system,
        )

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
        We use the convention of x aligning with the columns, second dimension,
        and y aligning with the rows, first dimension.
        """
        return self.pixel_to_world(*index)

        # TODO : DELETE
        # if len(np.shape(index)) == 1:
        #     index = [index]
        # return [self.pixel_to_world(*idx) for idx in index]
        # # return [self.array_index_to_world(*idx) for idx in index]

    def world_location_to_index(self, world: SkyCoord) -> np.ndarray:
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
        return self.world_to_pixel(world)

    @property
    def dvol(self) -> u.Quantity:
        """Computes the area of a grid cell (pixel) in angular u."""
        return self.distances[0] * self.distances[1]

    def world_extrema(
        self, extend_factor: float = 1, ext: Optional[tuple[int, int]] = None
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
            ext0, ext1 = [int(shp * extend_factor - shp) // 2 for shp in self.shape]
        else:
            ext0, ext1 = ext

        xmin = -ext0
        xmax = self.shape[0] + ext0 - 1
        ymin = -ext1
        ymax = self.shape[1] + ext1 - 1

        return [
            self.index_to_world_location(min_max)
            for min_max in [(xmin, ymin), (xmin, ymax), (xmax, ymin), (xmax, ymax)]
        ]

    def distances_in(self, unit: u.Unit) -> list[float]:
        return [d.to(unit).value for d in self.distances]

    def extent(self, unit=u.arcsec):
        """Convenience method which gives the extent of the grid in
        physical units."""
        distances = [d.to(unit).value for d in self.distances]
        halfside = np.array(self.shape) / 2 * np.array(distances)
        return -halfside[0], halfside[0], -halfside[1], halfside[1]


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
        wcs.wcs.crval[0], wcs.wcs.crval[1], unit="deg", frame=wcs.wcs.radesys.lower()
    )

    # Calculate FOV
    corners_pix = np.array([[0, 0], [nx - 1, 0], [nx - 1, ny - 1], [0, ny - 1]])
    corners_world = wcs.wcs_pix2world(corners_pix, 0)
    corners = SkyCoord(corners_world, unit="deg", frame=wcs.wcs.radesys.lower())

    # Calculate width (average of top and bottom)
    width = (corners[0].separation(corners[1]) + corners[3].separation(corners[2])) / 2

    # Calculate height (average of left and right)
    height = (corners[0].separation(corners[3]) + corners[1].separation(corners[2])) / 2

    return WcsAstropy(center, [nx, ny], (width, height))
