# SPDX-License-Identifier: BSD-2-Clause
# Authors: Matteo Guardiani

# Copyright(C) 2026 Max-Planck-Society

# %

"""NIRC2 narrow-camera distortion and the sky coordinates of its pixels.

The published solutions are pairs of lookup maps, Δx and Δy in pixels, one
value per detector pixel centre, "the values which should be added to raw
NIRC2 positions to shift them to a distortion free frame" (Service et al.
2016, PASP 128, 095004, Sect. 5; Yelda et al. 2010, ApJ 725, 331, Sect. 3.1).
Map element [row, column] is the shift of detector pixel x = column, y = row,
0-based, which is how the KAI pipeline feeds them to astropy
(``DistortionLookupTable(map, crpix=(0, 0), crval=(0, 0), cdelt=(1, 1))``).

The distortion-free frame has the solution's plate scale and orientation
offset beta. The true position angle of the NIRC2 y axis, East of North, is
``ROTPOSN - INSTANGL - beta`` (https://github.com/jluastro/nirc2_distortion/wiki,
figure "gamma = alpha - beta"; KOA applies "a correction of -0.252 deg").
Yelda et al. 2010 (Sect. 4.1) word it as "0.252 deg must be added to the
presumed P.A."; Gaia DR3 settles the sign for the subtraction
(``demos/nirc2_gaia_astrometry_demo.py``, raw frames at ROTPOSN - INSTANGL
0 and 90 deg): adding beta leaves -0.49 to -0.61 deg against Gaia.
Subtracting the published values leaves (Gaia minus loader), for Yelda,
+0.02 +- 0.03 deg at 0 deg and +0.01 +- 0.02 deg at 90 deg on M92 in 2007
July (Yelda's field and season) but -0.073 +- 0.006 deg on M53 in 2015
April; a rotator counting from the north of date would explain 0.03 of that
change by precession. For
Service, checked on one night only (M53 in 2015 May, Service's own field and
month), it leaves -0.056 +- 0.008 deg at 0 deg and -0.082 +- 0.011 deg at 90
deg. Frames of one cluster share Gaia stars. The published values are kept:
the cause (the reference each was measured against, precession, the AO mode:
LGS in 2015, NGS in 2007) is not settled. The CD matrix in the raw NIRC2
headers adds 0.252 deg, also after 2015.

The solutions define no reference pixel: the fitted maps carry a constant
term. `nirc2_wcs` therefore pins the raw reference pixel to the reference sky
position, so a position measured on the raw frame stays where it was given.
"""

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS, DistortionLookupTable

from ...wcs.wcs_base import WcsMixin

__all__ = [
    "REALIGNMENT",
    "DistortionSolution",
    "Nirc2Wcs",
    "nirc2_wcs",
    "select_distortion",
    "service2016",
    "yelda2010",
]

# NIRC2/AO re-alignment: Yelda et al. 2010 up to this date, Service et al.
# 2016 after it (https://www2.keck.hawaii.edu/inst/nirc2/dewarp.html). Against
# Gaia, M53 on 2015-04-02 fits the first, on 2015-05-05 the second. KAI keeps
# the first through 2015-04-30. KOA's on-sky narrow-camera frames from 04-14
# to 04-30 (all from 04-28 on: Io, HD 84800, LGS test fields, at most three
# Gaia stars each) cannot test the day.
REALIGNMENT = date(2015, 4, 13)


@dataclass(frozen=True)
class DistortionSolution:
    """One NIRC2 narrow-camera distortion solution.

    Parameters
    ----------
    name : str
        Shown in the diagnostics.
    dx, dy : np.ndarray
        Pixels added to the raw x (column) and y (row) of every detector pixel
        centre, in numpy (row, column) order.
    pixel_scale : u.Quantity
        Plate scale of the distortion-free frame, angle per pixel.
    angle_offset : u.Quantity
        beta in ``true PA = ROTPOSN - INSTANGL - beta``.
    """

    name: str
    dx: np.ndarray
    dy: np.ndarray
    pixel_scale: u.Quantity
    angle_offset: u.Quantity

    def __post_init__(self) -> None:
        # astropy's lookup tables take float32 only
        dx = np.asarray(self.dx, np.float32)
        dy = np.asarray(self.dy, np.float32)
        if dx.ndim != 2 or dx.shape != dy.shape:
            raise ValueError(f"dx {dx.shape} and dy {dy.shape}: two equal 2-D maps")
        if not (np.isfinite(dx).all() and np.isfinite(dy).all()):
            raise ValueError(f"{self.name}: distortion maps must be finite")
        object.__setattr__(self, "dx", dx)
        object.__setattr__(self, "dy", dy)


def yelda2010(x_map: str | Path, y_map: str | Path) -> DistortionSolution:
    """Pre-realignment solution, data up to 2015-04-13.

    9.952 mas/pixel and beta = 0.252 deg (Yelda et al. 2010, Table 4; repeated
    in Service et al. 2016, Table 2). Maps: ``nirc2_X_distortion.fits`` and
    ``nirc2_Y_distortion.fits`` (UCLA Galactic Center group), or
    ``nirc2_distort_{X,Y}_pre20150413_v1.fits`` (jluastro/nirc2_distortion).
    """
    return _from_fits("Yelda et al. 2010", x_map, y_map, 9.952 * u.mas, 0.252 * u.deg)


def service2016(x_map: str | Path, y_map: str | Path) -> DistortionSolution:
    """Post-realignment solution, data after 2015-04-13.

    9.971 mas/pixel and beta = 0.262 deg (Service et al. 2016, Table 2, M53
    May). Maps: ``nirc2_distort_{X,Y}_post20150413_v1.fits``
    (jluastro/nirc2_distortion).
    """
    return _from_fits("Service et al. 2016", x_map, y_map, 9.971 * u.mas, 0.262 * u.deg)


def _from_fits(name, x_map, y_map, pixel_scale, angle_offset) -> DistortionSolution:
    return DistortionSolution(
        f"{name} ({Path(x_map).name}, {Path(y_map).name})",
        fits.getdata(x_map),
        fits.getdata(y_map),
        pixel_scale,
        angle_offset,
    )


def select_distortion(
    date_obs: str,
    pre_2015: DistortionSolution | None,
    post_2015: DistortionSolution | None,
) -> DistortionSolution:
    """The solution for an observation date.

    Parameters
    ----------
    date_obs : str
        ISO date (``DATE-OBS``, UT); anything after the date part is ignored.
    pre_2015, post_2015 : DistortionSolution | None
        Solutions for dates up to and including 2015-04-13, and after it.

    Raises
    ------
    ValueError
        The date is not ISO, or the solution for it was not given.
    """
    post = date.fromisoformat(date_obs[:10]) > REALIGNMENT
    solution = post_2015 if post else pre_2015
    if solution is None:
        which = (
            "post_2015 (Service et al. 2016)"
            if post
            else "pre_2015 (Yelda et al. 2010)"
        )
        raise ValueError(f"DATE-OBS {date_obs} needs the {which} distortion solution")
    return solution


class Nirc2Wcs(WCS, WcsMixin):
    """An astropy WCS whose ``pixel_to_world`` applies the distortion maps,
    with the pixel-grid helpers jubik's subsampling expects."""


def nirc2_wcs(
    solution: DistortionSolution,
    crval: SkyCoord,
    crpix: tuple[float, float],
    position_angle: u.Quantity,
) -> Nirc2Wcs:
    """Tangent-plane WCS of one exposure, raw pixels to sky.

    A raw pixel position p goes to ``p + (dx, dy)(p)``, the maps interpolated
    bilinearly (held constant beyond the outermost pixel centres), and then
    through a linear TAN projection with the solution's plate scale. The
    science pixels are not resampled.

    Parameters
    ----------
    solution : DistortionSolution
        Its maps must have the science image's shape.
    crval : SkyCoord
        Sky position of the raw reference pixel.
    crpix : tuple[float, float]
        FITS (1-based) raw x, y of the reference pixel.
    position_angle : u.Quantity
        True position angle of the y axis, East of North (beta included).
    """
    w = Nirc2Wcs(naxis=2)
    w.cpdis1 = DistortionLookupTable(solution.dx, (0, 0), (0, 0), (1, 1))
    w.cpdis2 = DistortionLookupTable(solution.dy, (0, 0), (0, 0), (1, 1))
    raw = np.asarray(crpix, float)[None] - 1
    scale = solution.pixel_scale.to_value(u.deg)
    pa = position_angle.to_value(u.rad)
    crval = crval.icrs

    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.cunit = ["deg", "deg"]
    w.wcs.radesys = "ICRS"
    w.wcs.crval = [crval.ra.deg, crval.dec.deg]
    # the corrected position of the raw reference pixel lands on crval
    w.wcs.crpix = w.p4_pix2foc(raw, 0)[0] + 1
    # East to the left at PA 0, the jubik.wcs rule (_set_wcs_cards)
    w.wcs.cdelt = [-scale, scale]
    w.wcs.pc = [[np.cos(pa), -np.sin(pa)], [np.sin(pa), np.cos(pa)]]
    w.wcs.set()
    return w
