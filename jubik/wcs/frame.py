# SPDX-License-Identifier: BSD-2-Clause
# Copyright(C) 2026 Max-Planck-Society
# Author: Julian Rüstig

"""The one owner of spatial conventions.

Every spatial convention in jubik is implemented here and nowhere else:

* the canonical array layout ``sky[i, j]`` with ``i`` (dim 0) running along
  +Dec (North) and ``j`` (dim 1) along -RA (West), so that
  ``imshow(sky, origin="lower")`` renders North-up and East-left;
* the mapping between public ``(x, y)`` tuples and array ``(y, x)`` tuples;
* the FITS header rule (``CRPIX``, ``CDELT`` with ``CDELT1 < 0``, ``PC`` from a
  position angle measured from North through East);
* the matplotlib ``extent`` tuple;
* the relation between the canonical layout and every other array layout an
  instrument backend speaks (see :class:`Layout` and :meth:`SpatialGeometry.switch_layout`);
* padding, subsampling and cropping of a pixel grid.

Consumers receive a :class:`SpatialGeometry` and ask it questions such as
``n_ra`` or ``d_dec``. No module outside this one indexes a spatial tuple with
an integer literal, reverses one with ``[::-1]``, or transposes a sky array.
``test/conventions/test_frame_rule.py`` enforces that by grep.

The geometry is a pure pixel grid. It does not know the sky center, the
position angle or the coordinate system; those are astrometry and live on
:class:`jubik.wcs.WcsAstropy`, which owns one geometry.

The entries of the layout table are measured facts, not derivations. Each
one is pinned by a probe in ``probes/`` against an external anchor (astropy,
CASA, upstream resolve). Do not add an entry without one.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS

from ..parse.wcs.coordinate_system import CoordinateSystemModel, CoordinateSystems


class Layout(Enum):
    """Array layouts a sky can be stored in.

    ``CANONICAL_YX``
        ``sky[dec, -ra]``. The layout of every authored sky in jubik and the
        layout every ``Grid.array_shape`` describes. This is what the public
        xy / array yx documentation calls the array order; the word
        "internal" is not used here because the gridder layout below is
        internal too. ``geometry.shape_yx`` is its trailing shape.
    ``FITS_IMAGE``
        Same memory order as canonical. FITS axis 1 is the last numpy axis.
        Kept as a separate member so that the FITS export seam is a named
        boundary in the table, even though the entry is the identity.
    ``GRIDDER_LM``
        Native layout of the ducc0 wgridder and jax-finufft backends, ``(l, m)``.
        A pure transpose of canonical, no conjugation and no sign flip.
        Measured: p3, p4, p8 and w7 in ``probes/``.

    Public ``(x, y)`` tuples are not a layout. They are an argument order for
    constructors and config, and they exist only at :meth:`SpatialGeometry.from_xy`.
    """

    CANONICAL_YX = "canonical_yx"
    FITS_IMAGE = "fits_image"
    GRIDDER_LM = "gridder_lm"


def _swap_trailing(arr):
    return arr.swapaxes(-1, -2)


def _identity(arr):
    return arr


# Every array layout switch in the package. Keyed by (source, destination).
_LAYOUT_SWITCHES: dict[tuple[Layout, Layout], Callable] = {
    (Layout.CANONICAL_YX, Layout.CANONICAL_YX): _identity,
    (Layout.CANONICAL_YX, Layout.FITS_IMAGE): _identity,
    (Layout.FITS_IMAGE, Layout.CANONICAL_YX): _identity,
    (Layout.CANONICAL_YX, Layout.GRIDDER_LM): _swap_trailing,
    (Layout.GRIDDER_LM, Layout.CANONICAL_YX): _swap_trailing,
}


def _pair_int(value, name: str) -> tuple[int, int]:
    if isinstance(value, (int, np.integer)):
        value = (value, value)
    try:
        pair = tuple(int(v) for v in value)
    except TypeError:
        raise ValueError(f"{name} must be an int or a pair of ints, got {value!r}")
    if len(pair) != 2 or any(v <= 0 for v in pair):
        raise ValueError(f"{name} must contain two positive integers, got {value!r}")
    if any(int(v) != v for v in value):
        raise ValueError(f"{name} must contain integers, got {value!r}")
    return pair


def _pair_angle(value, name: str) -> u.Quantity:
    quantity = u.Quantity(value)
    if quantity.isscalar:
        quantity = u.Quantity((quantity, quantity))
    if quantity.shape != (2,):
        raise ValueError(f"{name} must contain two angular sizes, got {value!r}")
    if not quantity.unit.is_equivalent(u.rad):
        raise u.UnitConversionError(f"{name} must carry angular units, got {value!r}")
    if np.any(quantity <= 0 * quantity.unit):
        raise ValueError(f"{name} must contain two positive angular sizes, got {value!r}")
    return quantity


@dataclass(frozen=True, eq=False)
class SpatialGeometry:
    """A rectangular pixel grid on the sky, stored in canonical ``(y, x)`` order.

    Construct through :meth:`from_xy`, :meth:`from_yx` or :meth:`from_astropy_wcs`.
    Read through the named accessors. The stored fields are in array order;
    the constructors are the only two places a bare spatial tuple is accepted.
    """

    shape_yx: tuple[int, int]
    fov_yx: u.Quantity

    def __post_init__(self):
        object.__setattr__(self, "shape_yx", _pair_int(self.shape_yx, "shape_yx"))
        object.__setattr__(self, "fov_yx", _pair_angle(self.fov_yx, "fov_yx"))

    # ------------------------------------------------------------------ constructors
    @classmethod
    def from_xy(cls, shape_xy, fov_xy) -> "SpatialGeometry":
        """Build from public ``(nx, ny)`` and ``(fov_x, fov_y)``.

        This is the only reversal of a spatial tuple in jubik.
        """
        shape_xy = _pair_int(shape_xy, "shape_xy")
        fov_xy = _pair_angle(fov_xy, "fov_xy")
        return cls(shape_yx=shape_xy[::-1], fov_yx=fov_xy[::-1])

    @classmethod
    def from_yx(cls, shape_yx, fov_yx) -> "SpatialGeometry":
        """Build from array-order ``(ny, nx)`` and ``(fov_y, fov_x)``."""
        return cls(shape_yx=shape_yx, fov_yx=fov_yx)

    @classmethod
    def from_astropy_wcs(cls, wcs: WCS) -> "SpatialGeometry":
        """Recover the pixel grid of a celestial two-axis astropy WCS.

        Uses ``array_shape`` (already numpy ordered) and the row norms of the
        pixel scale matrix. FITS defines ``CDi_j = CDELTi * PCi_j`` with ``PC``
        a rotation, so the norm of row ``i`` is ``|CDELTi|`` for any position
        angle and any anisotropy. (``astropy.wcs.utils.proj_plane_pixel_scales``
        takes column norms and mixes the two scales once the grid is rotated.)
        PC and CD headers are both handled. The WCS must know its array shape
        (``NAXISn`` present or ``pixel_shape`` set).
        """
        if wcs.array_shape is None:
            raise ValueError("wcs has no array shape; set NAXIS1/NAXIS2 or pixel_shape")
        if wcs.naxis != 2:
            raise ValueError(f"expected a two-axis celestial WCS, got naxis={wcs.naxis}")
        n_dec, n_ra = wcs.array_shape
        scale_ra, scale_dec = np.sqrt((wcs.pixel_scale_matrix**2).sum(axis=1)) * u.Unit(wcs.wcs.cunit[0])
        return cls(shape_yx=(n_dec, n_ra), fov_yx=u.Quantity((scale_dec * n_dec, scale_ra * n_ra)))

    @classmethod
    def from_fits_header(cls, header) -> "SpatialGeometry":
        return cls.from_astropy_wcs(WCS(header))

    # ------------------------------------------------------------------ named scalars
    @property
    def n_dec(self) -> int:
        """Pixel count along Dec (array dim 0)."""
        return self.shape_yx[0]

    @property
    def n_ra(self) -> int:
        """Pixel count along RA (array dim 1)."""
        return self.shape_yx[1]

    @property
    def d_dec(self) -> u.Quantity:
        """Pixel size along Dec."""
        return self.fov_yx[0] / self.n_dec

    @property
    def d_ra(self) -> u.Quantity:
        """Pixel size along RA."""
        return self.fov_yx[1] / self.n_ra

    @property
    def dvol(self) -> u.Quantity:
        return self.d_ra * self.d_dec

    # ------------------------------------------------------------------ ordered views
    @property
    def shape_xy(self) -> tuple[int, int]:
        return (self.n_ra, self.n_dec)

    @property
    def fov_xy(self) -> u.Quantity:
        return u.Quantity((self.fov_yx[1], self.fov_yx[0]))

    @property
    def pixel_scales_yx(self) -> u.Quantity:
        return u.Quantity((self.d_dec, self.d_ra))

    @property
    def pixel_scales_xy(self) -> u.Quantity:
        return u.Quantity((self.d_ra, self.d_dec))

    def trailing_shape(self, layout: Layout) -> tuple[int, int]:
        """Trailing array shape of a sky stored in ``layout``."""
        if layout is Layout.GRIDDER_LM:
            return self.shape_xy
        return self.shape_yx

    # ------------------------------------------------------------------ derived grids
    def padded(self, ratio: float, fft_friendly: Optional[Callable[[int], int]] = None) -> "SpatialGeometry":
        """Grid enlarged by ``ratio`` per axis at constant pixel size.

        ``fft_friendly`` rounds each padded length; defaults to ducc0's
        ``good_size``. This is the rule that used to live in ``SkyModel``.
        """
        if ratio < 1:
            raise ValueError(f"padding ratio must be >= 1, got {ratio}")
        if fft_friendly is None:
            from ducc0.fft import good_size as fft_friendly
        new_shape = tuple(int(fft_friendly(int(n * ratio))) for n in self.shape_yx)
        scale = np.asarray(new_shape) / np.asarray(self.shape_yx)
        return SpatialGeometry(shape_yx=new_shape, fov_yx=self.fov_yx * scale)

    def crop_to(self, arr):
        """Cut a padded canonical array back to this geometry's trailing shape."""
        trailing = tuple(arr.shape[-2:])
        if trailing[0] < self.n_dec or trailing[1] < self.n_ra:
            raise ValueError(f"array trailing shape {trailing} is smaller than {self.shape_yx}")
        return arr[..., : self.n_dec, : self.n_ra]

    def subsampled(self, factor: int) -> "SpatialGeometry":
        """Grid with ``factor`` times more pixels per axis over the same field of view."""
        factor = int(factor)
        if factor < 1:
            raise ValueError(f"subsample factor must be >= 1, got {factor}")
        return SpatialGeometry(shape_yx=tuple(n * factor for n in self.shape_yx), fov_yx=self.fov_yx)

    def index_grid_yx(self) -> np.ndarray:
        """Integer ``(row, column)`` index grid, shape ``(2, n_dec, n_ra)``."""
        return np.indices(self.shape_yx)

    def index_grid_xy(self) -> tuple[np.ndarray, np.ndarray]:
        """``(column, row)`` index arrays, each shaped ``(n_dec, n_ra)``.

        For APIs that take separate x and y arguments (astropy ``pixel_to_world``,
        gwcs). The arrays keep canonical shape; only the argument order is xy.
        """
        row, column = self.index_grid_yx()
        return column, row

    def require_square(self, consumer: str) -> int:
        if self.n_ra != self.n_dec:
            raise ValueError(f"{consumer} requires a square grid; got shape_xy={self.shape_xy}")
        return self.n_ra

    # ------------------------------------------------------------------ orientation encodings
    def fits_header(
        self,
        center: SkyCoord,
        position_angle: u.Quantity = 0.0 * u.deg,
        coordinate_system: CoordinateSystemModel | CoordinateSystems = CoordinateSystems.icrs,
    ) -> dict:
        """The one CRPIX/CRVAL/CDELT/PC rule.

        FITS axis 1 is RA with ``CDELT1 < 0`` (East to the left), axis 2 is Dec.
        ``position_angle`` is measured from North through East. Byte-compatible
        with the header ``WcsAstropy`` has always written.

        Note that FITS applies ``CDELT`` after the ``PC`` rotation. For
        anisotropic pixels and a non-zero position angle the pixel grid on the
        sky is therefore sheared, not rigidly rotated; ``d_ra`` and ``d_dec``
        are the scales of the intermediate axes, not the edge lengths of a
        rotated pixel. Square pixels are unaffected.
        """
        if isinstance(coordinate_system, CoordinateSystems):
            coordinate_system = coordinate_system.value

        position_angle = u.Quantity(position_angle)
        if not position_angle.isscalar:
            raise ValueError("position_angle must be a scalar angle")
        if not position_angle.unit.is_equivalent(u.rad):
            raise u.UnitConversionError("position_angle must carry angular units")
        pa = position_angle.to_value(u.rad)

        if coordinate_system.radesys == CoordinateSystems.galactic.value.radesys:
            lon, lat = center.galactic.l.deg, center.galactic.b.deg
        else:
            lon, lat = center.ra.deg, center.dec.deg
        if np.isnan(lon) or np.isnan(lat):
            lon = lat = None

        header = {
            "WCSAXES": 2,
            "CTYPE1": coordinate_system.ctypes[0],
            "CTYPE2": coordinate_system.ctypes[1],
            "CRPIX1": self.n_ra / 2 + 0.5,
            "CRPIX2": self.n_dec / 2 + 0.5,
            "CRVAL1": lon,
            "CRVAL2": lat,
            "CDELT1": -self.d_ra.to_value(u.deg),
            "CDELT2": self.d_dec.to_value(u.deg),
            "PC1_1": np.cos(pa),
            "PC1_2": -np.sin(pa),
            "PC2_1": np.sin(pa),
            "PC2_2": np.cos(pa),
            "RADESYS": coordinate_system.radesys,
            "CUNIT1": "deg",
            "CUNIT2": "deg",
        }
        if coordinate_system.radesys in (
            CoordinateSystems.fk4.value.radesys,
            CoordinateSystems.fk5.value.radesys,
        ):
            header["EQUINOX"] = coordinate_system.equinox
        return header

    def imshow_extent(self, unit=u.arcsec) -> tuple[float, float, float, float]:
        """Matplotlib ``extent`` for a canonical sky at position angle zero.

        Returns ``(east_edge, west_edge, south_edge, north_edge)`` as
        ``(+half_ra, -half_ra, -half_dec, +half_dec)``: East on the left.
        """
        half_ra = (self.fov_yx[1] / 2).to_value(unit)
        half_dec = (self.fov_yx[0] / 2).to_value(unit)
        return half_ra, -half_ra, -half_dec, half_dec

    # ------------------------------------------------------------------ layouts
    def switch_layout(self, arr, src: Layout, dst: Layout):
        """Move a sky array between layouts, checking its trailing shape first.

        Raises if ``arr`` does not have the trailing shape this geometry expects
        for ``src``. On a rectangle that catches a transposed array at the call
        site instead of three modules downstream.
        """
        expected = self.trailing_shape(src)
        trailing = tuple(int(n) for n in arr.shape[-2:])
        if trailing != expected:
            raise ValueError(
                f"array trailing shape {trailing} does not match {src.name} layout "
                f"{expected} for geometry {self}"
            )
        try:
            convert = _LAYOUT_SWITCHES[(src, dst)]
        except KeyError:
            raise ValueError(f"no layout switch from {src.name} to {dst.name}; add a measured entry to frame._LAYOUT_SWITCHES")
        return convert(arr)

    # ------------------------------------------------------------------ dunder
    def __eq__(self, other) -> bool:
        if not isinstance(other, SpatialGeometry):
            return NotImplemented
        return self.shape_yx == other.shape_yx and bool(
            u.allclose(self.fov_yx, other.fov_yx, rtol=1e-12, atol=0 * u.arcsec)
        )

    def __hash__(self) -> int:
        return hash((self.shape_yx, tuple(np.round(self.fov_yx.to_value(u.arcsec), 9))))

    def __repr__(self) -> str:
        return (
            f"SpatialGeometry(n_ra={self.n_ra}, n_dec={self.n_dec}, "
            f"d_ra={self.d_ra.to(u.arcsec):.4g}, d_dec={self.d_dec.to(u.arcsec):.4g})"
        )
