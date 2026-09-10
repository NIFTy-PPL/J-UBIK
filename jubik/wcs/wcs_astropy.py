# SPDX-License-Identifier: BSD-2-Clause
# Authors: Julian Rüstig and Matteo Guardiani

# Copyright(C) 2024 Max-Planck-Society

# %%
from typing import Optional

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS

from ..parse.wcs.coordinate_system import CoordinateSystemModel, CoordinateSystems
from ..parse.wcs.spatial_model import SpatialModel
from .frame import SpatialGeometry
from .wcs_base import WcsMixin


class WcsAstropy(WCS, WcsMixin):
    """
    A wrapper around the astropy.wcs.WCS, in order to define a common interface
    with the gwcs.

    The pixel grid is owned by :attr:`geometry`, a :class:`SpatialGeometry`.
    This class adds astrometry to it: the sky center, the position angle and
    the coordinate system, and the projection astropy builds from them.

    Public constructor arguments are Cartesian ``(x, y)``: ``shape = (nx, ny)``
    and ``fov = (fov_x, fov_y)``. Celestial offsets use ``x = East`` and
    ``y = North``. Sky arrays are canonical ``(..., y, x)``; increasing columns
    move West on a zero-position-angle sky.
    """

    def __init__(
        self,
        center: SkyCoord,
        shape: int | tuple[int, int] | list[int],
        fov: u.Quantity | tuple[u.Quantity, u.Quantity],
        position_angle: u.Quantity = 0.0 * u.deg,
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
        shape : int or tuple
            Public grid shape ``(nx, ny)``. A scalar creates a square grid.
        fov : quantity or tuple
            Public field of view ``(fov_x, fov_y)``. A scalar is broadcast.
        position_angle : u.Quantity
            Astronomical position angle, measured from North toward East.
        coordinate_system : CoordinateSystemConfig
            Coordinate system to use ('icrs', 'fk5', 'fk4', 'galactic')
        """
        if isinstance(coordinate_system, CoordinateSystems):
            coordinate_system = coordinate_system.value

        self.geometry = SpatialGeometry.from_xy(shape, fov)
        self.center = center
        self.position_angle = u.Quantity(position_angle)
        if not self.position_angle.isscalar:
            raise ValueError("position_angle must be a scalar angle")
        if not self.position_angle.unit.is_equivalent(u.rad):
            raise u.UnitConversionError("position_angle must carry angular units")
        self.coordinate_system = coordinate_system

        header = self.geometry.fits_header(center, self.position_angle, coordinate_system)
        super().__init__(header)
        self.pixel_shape = self.geometry.shape_xy

    @classmethod
    def from_geometry(
        cls,
        geometry: SpatialGeometry,
        center: SkyCoord,
        position_angle: u.Quantity = 0.0 * u.deg,
        coordinate_system: Optional[CoordinateSystemModel] = CoordinateSystems.icrs.value,
    ) -> "WcsAstropy":
        """Attach astrometry to an existing pixel grid."""
        return cls(center, geometry.shape_xy, geometry.fov_xy, position_angle, coordinate_system)

    @classmethod
    def from_spatial_model(cls, spatial_model: SpatialModel):
        return WcsAstropy(
            spatial_model.wcs_model.center,
            spatial_model.shape_xy,
            spatial_model.fov_xy,
            spatial_model.wcs_model.position_angle,
            spatial_model.wcs_model.coordinate_system,
        )

    # ------------------------------------------------------------------ geometry forwards
    @property
    def shape_xy(self) -> tuple[int, int]:
        return self.geometry.shape_xy

    @property
    def shape_yx(self) -> tuple[int, int]:
        return self.geometry.shape_yx

    @property
    def fov_xy(self) -> u.Quantity:
        return self.geometry.fov_xy

    @property
    def fov_yx(self) -> u.Quantity:
        return self.geometry.fov_yx

    @property
    def pixel_scales_xy(self) -> u.Quantity:
        return self.geometry.pixel_scales_xy

    @property
    def pixel_scales_yx(self) -> u.Quantity:
        return self.geometry.pixel_scales_yx

    @property
    def dvol(self) -> u.Quantity:
        """Pixel area in square degrees, preserving the historical unit contract."""
        scales_deg = self.geometry.pixel_scales_xy.to(u.deg)
        return scales_deg[0] * scales_deg[1]

    # ------------------------------------------------------------------ astrometry
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
        list[SkyCoord]
            The world coordinates of the corner pixels.

        Note
        ----
        ``extension_value`` is NumPy ``(row, column)`` order. Astropy pixel
        coordinates are ``(column, row)`` / ``(x, y)``.
        """
        g = self.geometry
        if extension_value is None:
            ext_dec = int(g.n_dec * extension_factor - g.n_dec) // 2
            ext_ra = int(g.n_ra * extension_factor - g.n_ra) // 2
        else:
            ext_dec, ext_ra = extension_value

        xmin = -ext_ra + 0.5
        xmax = g.n_ra + ext_ra - 1 + 0.5
        ymin = -ext_dec + 0.5
        ymax = g.n_dec + ext_dec - 1 + 0.5

        points = np.array(((xmin, ymin), (xmin, ymax), (xmax, ymin), (xmax, ymax)))
        return self.pixel_to_world(*points.T)

    def extent(self, unit=u.Unit("arcsec")):
        """Matplotlib extent for a North-up, East-left zero-PA image."""
        pa = self.position_angle.to_value(u.deg) % 360.0
        if not np.isclose(pa, 0.0):
            raise ValueError("extent() is only valid at position_angle=0; use WCSAxes")
        return self.geometry.imshow_extent(unit)

    def world_to_offsets_xy(self, world: SkyCoord):
        """Return unit-bearing ``(East, North)`` offsets from the grid center."""
        return self.center.spherical_offsets_to(world)

    def offsets_xy_to_world(self, x_east: u.Quantity, y_north: u.Quantity):
        """Convert unit-bearing ``(East, North)`` offsets to world coordinates."""
        return self.center.spherical_offsets_by(x_east, y_north)


def WcsAstropy_from_wcs(wcs: WCS) -> WcsAstropy:
    """Rebuild a :class:`WcsAstropy` from a plain astropy WCS.

    The pixel grid comes from :meth:`SpatialGeometry.from_astropy_wcs`, which
    reads astropy's numpy-ordered ``array_shape`` and the row norms of the
    pixel scale matrix, so rectangles, anisotropic pixels, and CD-matrix
    headers are all handled. The position angle is read from the Dec row of
    the pixel scale matrix, whose ``CDELT2`` is positive.

    Parameters
    ----------
    wcs : astropy.wcs.WCS
        The WCS object to analyze. Must know its array shape.

    Returns
    -------
    WcsAstropy
    """
    geometry = SpatialGeometry.from_astropy_wcs(wcs)

    is_galactic = wcs.wcs.ctype[0].upper().startswith("GLON")
    frame_name = "galactic" if is_galactic else (wcs.wcs.radesys or "ICRS").lower()
    coordinate_system = getattr(CoordinateSystems, frame_name).value

    center = SkyCoord(wcs.wcs.crval[0], wcs.wcs.crval[1], unit="deg", frame=frame_name)

    cd = wcs.pixel_scale_matrix
    position_angle = np.arctan2(cd[1, 0], cd[1, 1]) * u.rad

    return WcsAstropy.from_geometry(geometry, center, position_angle, coordinate_system)
