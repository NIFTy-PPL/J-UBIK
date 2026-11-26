"""Claims: primary beams pair index for index with canonical skies.

Design page row pinned here (docs/source/user/canonical-sky-design.md):
"Sky beamer: beams are built on the canonical grid and multiply the sky index
for index", checked by "off-center pointing tests in all quadrants".

A pointing ``dE`` arcsec East and ``dN`` arcsec North of the grid center must
peak at

    i = c_dec + dN / d_dec   (North is +dim 0, at the Dec pixel size)
    j = c_ra  - dE / d_ra    (East  is -dim 1, at the RA  pixel size)

Pixels are anisotropic (1" Dec, 2" RA) so the fov <-> axis pairing is
observable, and a rectangle is included alongside the square.
"""

from types import SimpleNamespace

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord

import jubik as ju
from jubik.instruments.resolve.telescopes.primary_beam import build_beam
from jubik.wcs.wcs_astropy import WcsAstropy

CENTER = SkyCoord(ra=10.0 * u.deg, dec=20.0 * u.deg)


def _pointing(name, direction):
    return SimpleNamespace(phase_center=(direction.ra.rad, direction.dec.rad), name=name)


def _build_beams(pointings, shape_yx, fov_xy):
    grid = ju.Grid(
        spatial=WcsAstropy(center=CENTER, shape=shape_yx[::-1], fov=fov_xy),
        spectral=ju.Color.from_central_frequencies(np.array([100.0e9])),
    )
    beam_func = lambda freq, x: np.exp(-((x / 3.0e-5) ** 2))
    return {p.name: np.asarray(build_beam(p, grid, beam_func))[0, 0, 0] for p in pointings}


def _peak(beam):
    return tuple(int(k) for k in np.unravel_index(int(np.argmax(beam)), beam.shape))


# odd dims so the center pixel is unique; 1"/px Dec rows, 2"/px RA cols
SQUARE = ((33, 33), (66.0 * u.arcsec, 33.0 * u.arcsec))
RECT = ((25, 41), (82.0 * u.arcsec, 25.0 * u.arcsec))
GRIDS = [pytest.param(*SQUARE, id="square"), pytest.param(*RECT, id="rect")]
QUADRANTS = [(4, 4), (4, -4), (-4, 4), (-4, -4)]  # (East, North) arcsec


@pytest.mark.parametrize("shape_yx,fov_xy", GRIDS)
def test_centered_pointing_peaks_at_center(shape_yx, fov_xy):
    beams = _build_beams([_pointing("centered", CENTER)], shape_yx, fov_xy)
    c = (shape_yx[0] // 2, shape_yx[1] // 2)
    assert _peak(beams["centered"]) == c


@pytest.mark.parametrize("shape_yx,fov_xy", GRIDS)
@pytest.mark.parametrize("dE,dN", QUADRANTS)
def test_offset_pointing_peaks_north_up_east_left(shape_yx, fov_xy, dE, dN):
    c_dec, c_ra = shape_yx[0] // 2, shape_yx[1] // 2
    direction = CENTER.spherical_offsets_by(dE * u.arcsec, dN * u.arcsec)
    beams = _build_beams(
        [_pointing("centered", CENTER), _pointing("off", direction)],
        shape_yx, fov_xy,
    )
    expected = (c_dec + dN // 1, c_ra - dE // 2)  # 1"/px Dec, 2"/px RA
    assert _peak(beams["off"]) == expected, (
        f"pointing (E={dE}\", N={dN}\") peaks at {_peak(beams['off'])}, "
        f"expected {expected}: beam does not pair with the canonical frame"
    )
