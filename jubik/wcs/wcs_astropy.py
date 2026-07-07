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

    ``shape`` and ``fov`` are given in NUMPY/CANONICAL order — ``shape =
    (nDec, nRA)`` and ``fov = (fov_dec, fov_ra)`` — matching the layout of the
    sky array they describe (``sky[i, j]``: dim 0 = +Dec/North, dim 1 = -RA/West;
    see ``probes/README.md``).  The FITS header accordingly has axis 1 = RA
    (``CDELT1 < 0``, sized by ``shape[1]``/``fov[1]``) and axis 2 = Dec (sized by
    ``shape[0]``/``fov[0]``), and ``distances[k]`` describes array dim ``k``.
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
            The shape of the grid in numpy/canonical order ``(nDec, nRA)``.
        fov : tuple
            The field of view of the grid in numpy/canonical order
            ``(fov_dec, fov_ra)``. Typically given in degrees.
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
        # distances[k] = fov[k] / shape[k] is index-matched to array dim k
        # (dim 0 = Dec, dim 1 = RA); this is what charm's space_from_grid consumes.
        self.distances = u.Quantity([f.to(u.deg) / s for f, s in zip(fov, shape)])
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
            "CRPIX1": shape[1] / 2 + 0.5,
            "CRPIX2": shape[0] / 2 + 0.5,
            "CRVAL1": lon,
            "CRVAL2": lat,
            "CDELT1": -fov[1].to(u.deg).value / shape[1],
            "CDELT2": fov[0].to(u.deg).value / shape[0],
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
        ``shape``/``extension_value`` are numpy/canonical-ordered
        ``(dim0 = Dec, dim1 = RA)``.  ``pixel_to_world`` takes astropy pixel
        order ``(x, y)`` where the first pixel coordinate ``x`` is FITS axis 1
        (RA, columns of the sky array, ``shape[1]``) and the second ``y`` is
        axis 2 (Dec, rows, ``shape[0]``).
        """
        # NOTE : renamed ext -> extension_value
        # ext0 extends array dim 0 (Dec/y); ext1 extends array dim 1 (RA/x).
        if extension_value is None:
            ext0, ext1 = [int(shp * extension_factor - shp) // 2 for shp in self.shape]
        else:
            ext0, ext1 = extension_value

        # x = FITS axis 1 (RA) spans array dim 1 (shape[1]);
        # y = FITS axis 2 (Dec) spans array dim 0 (shape[0]).
        xmin = -ext1 + 0.5
        xmax = self.shape[1] + ext1 - 1 + 0.5
        ymin = -ext0 + 0.5
        ymax = self.shape[0] + ext0 - 1 + 0.5

        points = np.array(((xmin, ymin), (xmin, ymax), (xmax, ymin), (xmax, ymax)))
        return self.pixel_to_world(*points.T)

    def extent(self, unit=u.Unit("arcsec")):
        """The imshow extent 4-tuple ``(left, right, bottom, top)``.

        For a canonical sky ``sky[i, j]`` (dim 0 = Dec/rows, dim 1 = RA/columns)
        the imshow horizontal axis is dim 1 and the vertical axis is dim 0, so
        this returns ``(-h1, +h1, -h0, +h0)`` with
        ``h_k = shape[k] / 2 * distances[k]``.
        """
        distances = [d.to(unit).value for d in self.distances]
        halfside = np.array(self.shape) / 2 * np.array(distances)
        return -halfside[1], halfside[1], -halfside[0], halfside[0]

    def get_xycoords(self, centered: bool = True, unit: u.Unit = u.Unit("arcsec")):
        """Cartesian ``(x, y)`` coordinate meshgrid over the canonical grid.

        The x direction is FITS axis 1 (RA, array dim 1 -> ``shape[1]``/
        ``fov[1]``) and the y direction is axis 2 (Dec, array dim 0 ->
        ``shape[0]``/``fov[0]``); with ``indexing="xy"`` the returned arrays
        have the canonical sky layout ``(shape[0], shape[1])``.  Consumed by the
        black-body sky model to place Gaussians on the sky plane.
        """
        shape = self.shape
        distances = (u.Quantity(self.fov) / np.array(self.shape)).to(unit).value
        x_direction = coords(shape[1], distances[1])
        y_direction = coords(shape[0], distances[0])
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
    """Rebuild a :class:`WcsAstropy` from a plain astropy WCS.

    For an RA/Dec WCS astropy's ``wcs.array_shape`` is ``(ny, nx) = (nDec, nRA)``
    — numpy/canonical order — so it maps DIRECTLY onto the canonical
    ``shape = (nDec, nRA)``.  The field of view is returned index-matched as
    ``fov = (height_dec, width_ra)`` (dim 0 = Dec, dim 1 = RA), so the rebuilt
    object round-trips shape and fov on rectangular grids.

    Parameters
    ----------
    wcs : astropy.wcs.WCS
        The WCS object to analyze.

    Returns
    -------
    WcsAstropy
        A WcsAstropy with canonical ``shape = (nDec, nRA)`` and
        ``fov = (fov_dec, fov_ra)``.
    """
    # astropy array_shape is (ny, nx) = (nDec, nRA) for an RA/Dec WCS.
    n_dec, n_ra = wcs.array_shape

    # Get center coordinate
    center = SkyCoord(
        wcs.wcs.crval[0], wcs.wcs.crval[1], unit="deg", frame=wcs.wcs.radesys.lower()
    )

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
    corners = SkyCoord(corners_world, unit="deg", frame=wcs.wcs.radesys.lower())

    # width_ra: x (RA) varies along corner0->corner1 and corner3->corner2
    width = (corners[0].separation(corners[1]) + corners[3].separation(corners[2])) / 2

    # height_dec: y (Dec) varies along corner0->corner3 and corner1->corner2
    height = (corners[0].separation(corners[3]) + corners[1].separation(corners[2])) / 2

    return WcsAstropy(center, [n_dec, n_ra], (height, width))
