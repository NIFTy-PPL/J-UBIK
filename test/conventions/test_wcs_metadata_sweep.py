"""WcsAstropy metadata sweep — the canonical frame across shapes and anisotropy.

Breadth twin of ``probes/p1_metadata_vs_response.py``.  p1 pins the Batch-D
metadata alignment on ONE anisotropic rectangle; this module sweeps square and
both-orientation rectangular grids crossed with isotropic and 2:1 anisotropic
fields of view, so nothing can pass by a square-grid or isotropic cancellation.

For a canonical sky ``sky[i, j]`` (dim 0 = +Dec/North, dim 1 = -RA/West):

    shape = (nDec, nRA) = (dim0, dim1),  fov = (fov_dec, fov_ra)

so FITS axis 1 (RA, CDELT1 < 0) is sized by ``shape[1]``/``fov[1]`` and axis 2
(Dec) by ``shape[0]``/``fov[0]``; ``distances[k]`` describes array dim ``k``.

KNOWN-UNTESTED BOUNDARY: every assertion here (and everywhere in the canonical
statement) assumes grid ``rotation == 0``.  A non-zero WCS rotation mixes the
RA/Dec pixel axes and is OUTSIDE the canonical statement — it is deliberately
not exercised.  ``WcsAstropy`` accepts a ``rotation`` argument, but its frame
semantics under rotation are not part of this pinned convention.
"""

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS

from jubik.wcs import world_coordinates_to_index_grid
from jubik.wcs.wcs_astropy import WcsAstropy, WcsAstropy_from_wcs

CENTER = SkyCoord(ra=10.0 * u.deg, dec=20.0 * u.deg)

SHAPES = [(16, 16), (8, 20), (20, 8)]
# (name, ra_over_dec pixel-size ratio): iso = square pixels, 2:1 = anisotropic
ANISO = [("iso", 1.0), ("2to1", 2.0)]


def _wcs(shape, ratio):
    n_dec, n_ra = shape
    # dec pixels are 1", ra pixels are `ratio`" -> fov = (dec*1", ra*ratio")
    fov = (1.0 * n_dec * u.arcsec, ratio * n_ra * u.arcsec)
    return WcsAstropy(center=CENTER, shape=shape, fov=fov), fov


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("aniso_name,ratio", ANISO)
def test_header_pairing(shape, aniso_name, ratio):
    """(a) axis 1 (RA) <- shape[1]/fov[1] with CDELT1<0; axis 2 (Dec) <- [0]."""
    n_dec, n_ra = shape
    wcs, fov = _wcs(shape, ratio)
    h = wcs.to_header()
    assert h["CRPIX1"] == n_ra / 2 + 0.5, "RA axis not sized by shape[1]"
    assert h["CRPIX2"] == n_dec / 2 + 0.5, "Dec axis not sized by shape[0]"
    assert h["CDELT1"] < 0, "axis 1 must be RA (negative CDELT)"
    assert np.isclose(-h["CDELT1"] * 3600, fov[1].to(u.arcsec).value / n_ra)
    assert np.isclose(h["CDELT2"] * 3600, fov[0].to(u.arcsec).value / n_dec)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("aniso_name,ratio", ANISO)
def test_distances_index_matched(shape, aniso_name, ratio):
    """(b) distances[k] == fov[k]/shape[k] (space_from_grid-safe)."""
    n_dec, n_ra = shape
    wcs, fov = _wcs(shape, ratio)
    d = wcs.distances.to(u.arcsec).value
    assert np.allclose(
        d, [fov[0].to(u.arcsec).value / n_dec, fov[1].to(u.arcsec).value / n_ra]
    )


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("aniso_name,ratio", ANISO)
def test_extent_house_recipe(shape, aniso_name, ratio):
    """(c) extent() == (-h1, +h1, -h0, +h0) with h_k = shape[k]/2 * distances[k]."""
    wcs, _ = _wcs(shape, ratio)
    d = wcs.distances.to(u.arcsec).value
    h0 = shape[0] / 2 * d[0]
    h1 = shape[1] / 2 * d[1]
    assert np.allclose(wcs.extent(u.arcsec), (-h1, h1, -h0, h0))


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("aniso_name,ratio", ANISO)
def test_world_span_coherence(shape, aniso_name, ratio):
    """(d) the world corners span exactly [0, shape[k]-1] on each array dim."""
    n_dec, n_ra = shape
    wcs, _ = _wcs(shape, ratio)
    corner_lo = wcs.pixel_to_world(0.0, 0.0)
    corner_hi = wcs.pixel_to_world(float(n_ra - 1), float(n_dec - 1))
    span = SkyCoord(
        ra=np.array([[corner_lo.ra.deg, corner_hi.ra.deg]]) * u.deg,
        dec=np.array([[corner_lo.dec.deg, corner_hi.dec.deg]]) * u.deg,
    )
    r = np.array(
        world_coordinates_to_index_grid([span], wcs, "ij")[0]
    ).reshape(2, 2)
    assert np.allclose(sorted(r[0]), [0, n_dec - 1], atol=1e-6), r[0]
    assert np.allclose(sorted(r[1]), [0, n_ra - 1], atol=1e-6), r[1]


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("aniso_name,ratio", ANISO)
def test_from_wcs_roundtrip(shape, aniso_name, ratio):
    """(e) WcsAstropy_from_wcs reads array_shape as (ny, nx) and reproduces."""
    n_dec, n_ra = shape
    wcs, fov = _wcs(shape, ratio)
    plain = WCS(wcs.to_header())
    plain.pixel_shape = (n_ra, n_dec)  # astropy pixel order (nx, ny)
    rebuilt = WcsAstropy_from_wcs(plain)
    assert tuple(rebuilt.shape) == (n_dec, n_ra), rebuilt.shape
    fov_rb = u.Quantity(rebuilt.fov).to(u.arcsec).value
    assert np.allclose(
        fov_rb, [fov[0].to(u.arcsec).value, fov[1].to(u.arcsec).value], rtol=1e-3
    ), fov_rb


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("aniso_name,ratio", ANISO)
def test_channel_truth_anisotropic(shape, aniso_name, ratio):
    """(f) North/East offsets track dim0/+ and dim1/- at the RIGHT pixel counts.

    Offsetting by exactly 3 Dec-pixels North moves +3 on dim0; offsetting by
    exactly 3 RA-pixels East moves -3 on dim1.  Because the offsets are measured
    in per-axis pixel sizes, this pins which pixel size divides which direction
    (the anisotropic case is what makes that observable).
    """
    wcs, _ = _wcs(shape, ratio)
    d_dec_as, d_ra_as = wcs.distances.to(u.arcsec).value
    north = CENTER.spherical_offsets_by(0 * u.arcsec, 3 * d_dec_as * u.arcsec)
    east = CENTER.spherical_offsets_by(3 * d_ra_as * u.arcsec, 0 * u.arcsec)
    pts = SkyCoord(
        ra=np.array([[CENTER.ra.deg, north.ra.deg, east.ra.deg]]) * u.deg,
        dec=np.array([[CENTER.dec.deg, north.dec.deg, east.dec.deg]]) * u.deg,
    )
    idx = world_coordinates_to_index_grid([pts], wcs, "ij")[0]
    c0, c1 = idx[0].ravel(), idx[1].ravel()  # order: center, north, east
    assert np.isclose(c0[1] - c0[0], 3.0), f"dim0 does not track +Dec: {c0}"
    assert np.isclose(c1[2] - c1[0], -3.0), f"dim1 does not track -RA: {c1}"
