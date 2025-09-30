# SPDX-License-Identifier: BSD-2-Clause
# Authors: Julian Rüstig and Matteo Guardiani

# Copyright(C) 2024 Max-Planck-Society

# %%
from typing import List, Optional, Union

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord, distances
from astropy.wcs import WCS
from numpy.typing import ArrayLike

from ..parse.wcs.coordinate_system import CoordinateSystemModel, CoordinateSystems
from ..parse.wcs.spatial_model import SpatialModel
from .wcs_base import WcsMixin


class WcsAstropy(WCS, WcsMixin):
    """
    A wrapper around the astropy.wcs.WCS, in order to define a common interface
    with the gwcs.
    """

    def __init__(
        self,
        center: SkyCoord,
        shape: tuple[int, int] | list[int],
        fov: u.Quantity | tuple[u.Quantity, u.Quantity],
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

        if isinstance(fov, u.Quantity):
            assert fov.shape == 2 or fov.shape == (2,)

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

        if np.isnan(lon) or np.isnan(lat):
            lon = lat = None

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

    @property
    def dvol(self) -> u.Quantity:
        """Computes the area of a grid cell (pixel) in angular u."""
        return self.distances[0] * self.distances[1]

    def world_corners(
        self,
        extension_value: Optional[tuple[int, int]] = None,
        extension_factor: float = 1,
    ) -> list[SkyCoord]:
        """
        The world location of the center of the pixels with the index
        locations = ((0, 0), (0, -1), (-1, 0), (-1, -1))

        Parameters
        ----------
        extension_value : tuple of int, optional
            Specific extension values for the grid's rows and columns.
        extension_factor : float, optional
            A factor by which to extend the grid. Default is 1.

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
        # NOTE : renamed ext -> extension_value

        if extension_value is None:
            ext0, ext1 = [int(shp * extension_factor - shp) // 2 for shp in self.shape]
        else:
            ext0, ext1 = extension_value

        xmin = -ext0 + 0.5
        xmax = self.shape[0] + ext0 - 1 + 0.5
        ymin = -ext1 + 0.5
        ymax = self.shape[1] + ext1 - 1 + 0.5

        points = np.array(((xmin, ymin), (xmin, ymax), (xmax, ymin), (xmax, ymax)))
        return self.pixel_to_world(*points.T)

    def extent(self, unit=u.Unit("arcsec")):
        """Convenience method which gives the extent of the grid in
        physical units."""
        distances = [d.to(unit).value for d in self.distances]
        halfside = np.array(self.shape) / 2 * np.array(distances)
        return -halfside[0], halfside[0], -halfside[1], halfside[1]

    def get_xycoords(self, centered: bool = True, unit: u.Unit = u.Unit("arcsec")):
        shape = self.shape
        distances = (u.Quantity(self.fov) / np.array(self.shape)).to(unit).value
        x_direction = coords(shape[0], distances[0])
        y_direction = coords(shape[1], distances[1])
        fieldcentered = np.array(np.meshgrid(x_direction, y_direction, indexing="xy"))

        if centered:
            return fieldcentered
        else:
            npix = shape[0]
            if not npix == shape[1]:
                raise NotImplementedError("Not implemented for rectangular grids.")

            if npix % 2 == 0:
                return np.fft.fftshift(
                    fieldcentered - np.array(distances)[:, None, None] / 2.0
                )
            else:
                return np.fft.fftshift(fieldcentered)


def coords(shape: int, distance: float) -> ArrayLike:
    """Returns coordinates such that the edge of the array is
    shape/2*distance"""
    halfside = shape / 2 * distance
    return np.linspace(-halfside + distance / 2, halfside - distance / 2, shape)


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
