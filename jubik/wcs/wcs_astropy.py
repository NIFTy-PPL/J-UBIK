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
from .wcs_base import WcsMixin


class WcsAstropy(WCS, WcsMixin):
    """
    A wrapper around the astropy.wcs.WCS, in order to define a common interface
    with the gwcs.

    Public geometry is Cartesian ``(x, y)``: ``shape = (nx, ny)`` and
    ``fov = (fov_x, fov_y)``.  Celestial offsets use ``x = East`` and
    ``y = North``.  Numerical arrays remain NumPy-native ``(..., y, x)``;
    increasing columns therefore move West on a zero-position-angle sky.
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
        equinox : float, optional
            Equinox for FK4/FK5 systems (e.g., 2000.0 for J2000)
        """

        if isinstance(shape, (int, np.integer)):
            shape = (int(shape), int(shape))
        if (
            len(shape) != 2
            or any(not isinstance(v, (int, np.integer)) for v in shape)
            or any(int(v) <= 0 for v in shape)
        ):
            raise ValueError(f"shape must contain two positive dimensions, got {shape}")
        self.shape_xy = tuple(int(v) for v in shape)
        self.shape_yx = self.shape_xy[::-1]

        fov = u.Quantity(fov)
        if fov.isscalar:
            fov = u.Quantity((fov, fov))
        if fov.shape != (2,) or np.any(fov <= 0 * fov.unit):
            raise ValueError(f"fov must contain two positive angular sizes, got {fov}")
        if not fov.unit.is_equivalent(u.rad):
            raise u.UnitConversionError("fov must carry angular units")
        self.fov_xy = fov
        self.fov_yx = fov[::-1]
        self.pixel_scales_xy = self.fov_xy / np.asarray(self.shape_xy)
        self.pixel_scales_yx = self.pixel_scales_xy[::-1]
        self.center = center
        self.position_angle = u.Quantity(position_angle)
        if not self.position_angle.isscalar:
            raise ValueError("position_angle must be a scalar angle")
        if not self.position_angle.unit.is_equivalent(u.rad):
            raise u.UnitConversionError("position_angle must carry angular units")

        # Calculate rotation matrix
        rotation_value = self.position_angle.to(u.rad).value
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
            "CRPIX1": self.shape_xy[0] / 2 + 0.5,
            "CRPIX2": self.shape_xy[1] / 2 + 0.5,
            "CRVAL1": lon,
            "CRVAL2": lat,
            "CDELT1": -self.pixel_scales_xy[0].to_value(u.deg),
            "CDELT2": self.pixel_scales_xy[1].to_value(u.deg),
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
            spatial_model.shape_xy,
            spatial_model.fov_xy,
            spatial_model.wcs_model.position_angle,
            spatial_model.wcs_model.coordinate_system,
        )

    @property
    def dvol(self) -> u.Quantity:
        """Pixel area in square degrees, preserving the historical unit contract."""
        scales_deg = self.pixel_scales_xy.to(u.deg)
        return scales_deg[0] * scales_deg[1]

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
        # NOTE : renamed ext -> extension_value
        # ext0 extends array dim 0 (Dec/y); ext1 extends array dim 1 (RA/x).
        if extension_value is None:
            ext0, ext1 = [
                int(shp * extension_factor - shp) // 2 for shp in self.shape_yx
            ]
        else:
            ext0, ext1 = extension_value

        # x = FITS axis 1 (RA) spans array dim 1 (shape[1]);
        # y = FITS axis 2 (Dec) spans array dim 0 (shape[0]).
        xmin = -ext1 + 0.5
        xmax = self.shape_yx[1] + ext1 - 1 + 0.5
        ymin = -ext0 + 0.5
        ymax = self.shape_yx[0] + ext0 - 1 + 0.5

        points = np.array(((xmin, ymin), (xmin, ymax), (xmax, ymin), (xmax, ymax)))
        return self.pixel_to_world(*points.T)

    def extent(self, unit=u.Unit("arcsec")):
        """Matplotlib extent for a North-up, East-left zero-PA image."""
        pa = self.position_angle.to_value(u.deg) % 360.0
        if not np.isclose(pa, 0.0):
            raise ValueError("extent() is only valid at position_angle=0; use WCSAxes")
        half_x, half_y = (self.fov_xy / 2).to_value(unit)
        return half_x, -half_x, -half_y, half_y

    def world_to_offsets_xy(self, world: SkyCoord):
        """Return unit-bearing ``(East, North)`` offsets from the grid center."""
        return self.center.spherical_offsets_to(world)

    def offsets_xy_to_world(self, x_east: u.Quantity, y_north: u.Quantity):
        """Convert unit-bearing ``(East, North)`` offsets to world coordinates."""
        return self.center.spherical_offsets_by(x_east, y_north)

    def world_to_indices_yx(self, world: SkyCoord):
        """Return floating NumPy ``(row, column)`` indices for world coordinates."""
        column, row = self.world_to_pixel(world)
        return row, column

    def indices_yx_to_world(self, row, column):
        """Convert NumPy ``(row, column)`` indices to world coordinates."""
        return self.pixel_to_world(column, row)

    def coordinate_grid_yx(self):
        """Internal ``(North, East)`` offset grids, each shaped ``shape_yx``."""
        column, row = np.meshgrid(
            np.arange(self.shape_xy[0]),
            np.arange(self.shape_xy[1]),
            indexing="xy",
        )
        east, north = self.world_to_offsets_xy(self.pixel_to_world(column, row))
        return north, east


def WcsAstropy_from_wcs(wcs: WCS) -> WcsAstropy:
    """Rebuild a :class:`WcsAstropy` from a plain astropy WCS.

    Astropy's ``array_shape`` is NumPy ``(ny, nx)``. It is reversed once into
    the public ``shape_xy = (nx, ny)``; the reconstructed field of view is
    likewise public ``(width_x, height_y)``.

    Parameters
    ----------
    wcs : astropy.wcs.WCS
        The WCS object to analyze.

    Returns
    -------
    WcsAstropy
        A WcsAstropy with public ``shape_xy`` and ``fov_xy`` metadata.
    """
    # astropy array_shape is (ny, nx) = (nDec, nRA) for an RA/Dec WCS.
    n_dec, n_ra = wcs.array_shape

    is_galactic = wcs.wcs.ctype[0].upper().startswith("GLON")
    frame_name = "galactic" if is_galactic else (wcs.wcs.radesys or "ICRS").lower()
    coordinate_system = getattr(CoordinateSystems, frame_name).value

    # Get center coordinate
    center = SkyCoord(wcs.wcs.crval[0], wcs.wcs.crval[1], unit="deg", frame=frame_name)

    # Full-extent corners in astropy pixel order (x = axis 1 = RA over n_ra,
    # y = axis 2 = Dec over n_dec), sampled at the pixel edges (-0.5 .. n-0.5)
    # so the recovered separations span the full field of view n * CDELT.
    corners_pix = np.array(
        [
            [-0.5, -0.5],
            [n_ra - 0.5, -0.5],
            [n_ra - 0.5, n_dec - 0.5],
            [-0.5, n_dec - 0.5],
        ]
    )
    corners_world = wcs.wcs_pix2world(corners_pix, 0)
    corners = SkyCoord(corners_world, unit="deg", frame=frame_name)

    # width_ra: x (RA) varies along corner0->corner1 and corner3->corner2
    width = (corners[0].separation(corners[1]) + corners[3].separation(corners[2])) / 2

    # height_dec: y (Dec) varies along corner0->corner3 and corner1->corner2
    height = (corners[0].separation(corners[3]) + corners[1].separation(corners[2])) / 2

    pc = wcs.wcs.get_pc()
    position_angle = np.arctan2(pc[1, 0], pc[0, 0]) * u.rad
    return WcsAstropy(
        center,
        (n_ra, n_dec),
        (width, height),
        position_angle,
        coordinate_system,
    )
