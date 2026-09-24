# SPDX-License-Identifier: BSD-2-Clause
# Copyright(C) 2026 Max-Planck-Society
# Author: Julian Rüstig

"""SpatialGeometry: the pixel grid of a sky, with axis order owned in one place.

A sky array in jubik is stored ``sky[..., dec, ra]``: dim 0 of the trailing
pair increases toward North, dim 1 toward West. Public constructor arguments
and config files use Cartesian ``(x, y)`` order. The mapping between the two
orders happens in :meth:`SpatialGeometry.from_xy` and nowhere else in this
package; consumers read ``n_ra``, ``n_dec``, ``d_ra``, ``d_dec`` instead of
indexing a shape tuple.

The geometry is a pure pixel grid. It depends on numpy and astropy units
only. Sky center, position angle, coordinate system and the FITS/WCS
projection are astrometry and live on :class:`jubik.wcs.WcsAstropy`, which
owns one geometry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from astropy import units as u

from .._spatial_validation import normalize_fov, normalize_shape


#: Relative tolerance of :meth:`SpatialGeometry.__eq__` on the field of view.
#: Far below any physical distinction, far above unit-conversion round-off.
FOV_RTOL = 1e-10


@dataclass(frozen=True, eq=False)
class SpatialGeometry:
    """A rectangular pixel grid on the sky, stored in array ``(y, x)`` order.

    Construct through :meth:`from_xy` or :meth:`from_yx`. Read through the
    named accessors. Equality is exact on shape and within ``FOV_RTOL`` on
    the field of view, whatever unit it was given in; instances are
    deliberately unhashable.
    """

    shape_yx: tuple[int, int]
    fov_yx: u.Quantity

    def __post_init__(self):
        object.__setattr__(self, "shape_yx", normalize_shape(self.shape_yx, "shape_yx"))
        object.__setattr__(self, "fov_yx", normalize_fov(self.fov_yx, "fov_yx"))

    # ------------------------------------------------------------------ constructors
    @classmethod
    def from_xy(cls, shape_xy, fov_xy) -> "SpatialGeometry":
        """Build from public ``(nx, ny)`` and ``(fov_x, fov_y)``.

        This is the only reversal of a spatial tuple in jubik.
        """
        shape_xy = normalize_shape(shape_xy, "shape_xy")
        fov_xy = normalize_fov(fov_xy, "fov_xy")
        return cls(shape_yx=shape_xy[::-1], fov_yx=fov_xy[::-1])

    @classmethod
    def from_yx(cls, shape_yx, fov_yx) -> "SpatialGeometry":
        """Build from array-order ``(ny, nx)`` and ``(fov_y, fov_x)``."""
        return cls(shape_yx=shape_yx, fov_yx=fov_yx)

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

    # ------------------------------------------------------------------ derived grids
    def padded(self, ratio: float, fft_friendly: Optional[Callable[[int], int]] = None) -> "SpatialGeometry":
        """Grid enlarged by ``ratio`` per axis at constant pixel size.

        ``fft_friendly`` rounds each padded length; defaults to ducc0's
        ``good_size``. Used by ``SkyModel`` for the correlated-field padding.
        """
        if ratio < 1:
            raise ValueError(f"padding ratio must be >= 1, got {ratio}")
        if fft_friendly is None:
            from ducc0.fft import good_size as fft_friendly
        new_shape = tuple(int(fft_friendly(int(n * ratio))) for n in self.shape_yx)
        scale = np.asarray(new_shape) / np.asarray(self.shape_yx)
        return SpatialGeometry(shape_yx=new_shape, fov_yx=self.fov_yx * scale)

    # ------------------------------------------------------------------ orientation
    def imshow_extent(self, unit=u.arcsec) -> tuple[float, float, float, float]:
        """Matplotlib ``extent`` for a sky array at position angle zero.

        Returns ``(east_edge, west_edge, south_edge, north_edge)`` as
        ``(+half_ra, -half_ra, -half_dec, +half_dec)``: East on the left.
        """
        half_ra = (self.fov_yx[1] / 2).to_value(unit)
        half_dec = (self.fov_yx[0] / 2).to_value(unit)
        return half_ra, -half_ra, -half_dec, half_dec

    # ------------------------------------------------------------------ dunder
    def __eq__(self, other) -> bool:
        if not isinstance(other, SpatialGeometry):
            return NotImplemented
        # tolerant on the field of view: unit conversion (rad, mas, ...) leaves
        # float noise well below FOV_RTOL; no hash has to agree with this
        return self.shape_yx == other.shape_yx and bool(
            u.allclose(self.fov_yx, other.fov_yx, rtol=FOV_RTOL)
        )

    __hash__ = None

    def __repr__(self) -> str:
        return (
            f"SpatialGeometry(n_ra={self.n_ra}, n_dec={self.n_dec}, "
            f"d_ra={self.d_ra.to(u.arcsec):.4g}, d_dec={self.d_dec.to(u.arcsec):.4g})"
        )
