"""Sky-beamer sweep — beam peak placement across quadrants, anisotropy, shape.

Breadth twin of ``probes/p5_sky_beamer_frame.py``.  p5 pins ONE off-center
pointing (4" N, 4" E) on a 33x33 anisotropic grid.  This module sweeps all four
(East, North) quadrants and adds a rectangular grid (beams on rectangles are new
post-Batch-D coverage).

Under the canonical frame (dim 0 = +Dec/North, dim 1 = -RA/West) a pointing
``dE`` arcsec East and ``dN`` arcsec North of the grid center must peak at

    i = c_dec + dN / d_dec   (North -> +dim0, at the Dec pixel size)
    j = c_ra  - dE / d_ra    (East  -> -dim1, at the RA  pixel size)

``sky_fov`` and the sky's trailing shape are NUMPY/CANONICAL-ordered
``(nDec, nRA)`` / ``(fov_dec, fov_ra)``.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from types import SimpleNamespace

import astropy.units as u
import nifty.re as jft
import numpy as np
import pytest
from astropy.coordinates import SkyCoord

from jubik.instruments.resolve.mosaicing.sky_beamer import build_sky_beamer

CENTER = SkyCoord(ra=10.0 * u.deg, dec=20.0 * u.deg)


def _pointing_observation(name, direction):
    phase_center = (direction.ra.rad, direction.dec.rad)
    d = SimpleNamespace(phase_center=phase_center, name=name)
    return SimpleNamespace(direction_from_key=lambda key, d=d: d, direction=d)


def _build_beams(observations, shape, fov):
    beamer = build_sky_beamer(
        sky_shape_with_dtype=jft.ShapeWithDtype(
            (1, 1, 1, shape[0], shape[1]), np.float64
        ),
        sky_fov=fov,
        sky_center=CENTER,
        sky_frequency_means=u.Quantity([100.0e9] * u.Hz),
        observations=observations,
        beam_func=lambda freq, x: np.exp(-((x / 3.0e-5) ** 2)),
    )
    return {k: np.asarray(v.beam)[0, 0, 0] for k, v in beamer.beam_directions.items()}


# 33x33 odd square, anisotropic pixels: 1"/px Dec, 2"/px RA (p5 geometry)
SQUARE_N = 33
SQUARE_FOV = (66.0 * u.arcsec, 33.0 * u.arcsec)
# four (East, North) arcsec quadrants; magnitudes divide the pixel sizes cleanly
QUADRANTS = [(4, 4), (4, -4), (-4, 4), (-4, -4)]


@pytest.mark.parametrize("dE,dN", QUADRANTS)
def test_square_quadrant_peak(dE, dN):
    c = SQUARE_N // 2
    direction = CENTER.spherical_offsets_by(dE * u.arcsec, dN * u.arcsec)
    beams = _build_beams(
        [_pointing_observation("centered", CENTER),
         _pointing_observation("off", direction)],
        (SQUARE_N, SQUARE_N), SQUARE_FOV,
    )
    peak = np.unravel_index(int(np.argmax(beams["off"])), beams["off"].shape)
    expected = (c + dN // 1, c - dE // 2)  # 1"/px Dec rows, 2"/px RA cols
    assert peak == expected, (
        f"pointing (E={dE}\", N={dN}\") peaks at {peak}, expected {expected} "
        "— beam does not pair with the canonical sky frame"
    )


def test_square_centered_control():
    c = SQUARE_N // 2
    beams = _build_beams(
        [_pointing_observation("centered", CENTER)], (SQUARE_N, SQUARE_N), SQUARE_FOV
    )
    peak = np.unravel_index(int(np.argmax(beams["centered"])), (SQUARE_N, SQUARE_N))
    assert peak == (c, c), f"centered pointing peaks at {peak}, expected {(c, c)}"


# RECTANGLE, odd dims, anisotropic pixels: nDec=25 (1"/px), nRA=41 (2"/px)
RECT_SHAPE = (25, 41)  # internal (ny, nx)
RECT_FOV = (82.0 * u.arcsec, 25.0 * u.arcsec)  # public (x, y)


@pytest.mark.parametrize("dE,dN", [(4, 4), (-4, -4)])
def test_rectangle_quadrant_peak(dE, dN):
    c_dec, c_ra = RECT_SHAPE[0] // 2, RECT_SHAPE[1] // 2
    direction = CENTER.spherical_offsets_by(dE * u.arcsec, dN * u.arcsec)
    beams = _build_beams(
        [_pointing_observation("centered", CENTER),
         _pointing_observation("off", direction)],
        RECT_SHAPE, RECT_FOV,
    )
    peak = np.unravel_index(int(np.argmax(beams["off"])), beams["off"].shape)
    expected = (c_dec + dN // 1, c_ra - dE // 2)
    assert peak == expected, (
        f"rectangle pointing (E={dE}\", N={dN}\") peaks at {peak}, "
        f"expected {expected} — rectangle beam frame broken"
    )
