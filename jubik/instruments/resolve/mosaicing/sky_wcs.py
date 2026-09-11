# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright(C) 2025 Max-Planck-Society
# Author: Julian Rüstig


from typing import Tuple

from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.units import Unit
from astropy import units

import numpy as np


def build_astropy_wcs(
    center: SkyCoord,
    shape: Tuple[int, int],
    fov: Tuple[Unit, Unit],
    position_angle: Unit = 0.0 * units.deg,
) -> WCS:
    """
    Specify the Astropy wcs.

    Public ``shape`` and ``fov`` use ``(x, y)`` order. The returned WCS has
    FITS axis 1 = RA/x with negative CDELT and axis 2 = Dec/y.

    Parameters
    ----------
    center : SkyCoord
        The value of the center of the coordinate system (crval).

    shape : tuple
        Public shape ``(nx, ny)``.

    fov : tuple
        Public field of view ``(fov_x, fov_y)``.

    position_angle : units.Quantity
        Astronomical position angle measured from North through East.
    """

    # Create a WCS object
    w = WCS(naxis=2)

    # Rotation
    rotation_value = position_angle.to(units.rad).value
    pc11 = np.cos(rotation_value)
    pc12 = -np.sin(rotation_value)
    pc21 = np.sin(rotation_value)
    pc22 = np.cos(rotation_value)

    # Set up ICRS system
    w.wcs.crpix = [shape[0] / 2 + 0.5, shape[1] / 2 + 0.5]
    w.wcs.cdelt = [
        -fov[0].to(units.deg).value / shape[0],
        fov[1].to(units.deg).value / shape[1],
    ]
    w.wcs.crval = [center.ra.deg, center.dec.deg]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.pc = np.array([[pc11, pc12], [pc21, pc22]])

    return w
