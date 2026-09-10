"""WcsAstropy metadata sweep — the canonical frame across shapes and anisotropy.

Breadth twin of ``probes/p1_metadata_vs_response.py``.  p1 pins the Batch-D
metadata alignment on ONE anisotropic rectangle; this module sweeps square and
both-orientation rectangular grids crossed with isotropic and 2:1 anisotropic
fields of view, so nothing can pass by a square-grid or isotropic cancellation.

Public geometry is ``shape=(nx, ny)``, ``fov=(fov_x, fov_y)``. For a
canonical sky ``sky[i, j]`` (dim 0 = +Dec/North, dim 1 = -RA/West):

    sky.shape = shape_yx = (ny, nx)

FITS axis 1 is x/RA and FITS axis 2 is y/Dec. The final test pins the
astronomical position-angle sign; plotting rotated grids requires WCSAxes.
"""

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS

from jubik.wcs.wcs_astropy import WcsAstropy, WcsAstropy_from_wcs

CENTER = SkyCoord(ra=10.0 * u.deg, dec=20.0 * u.deg)

SHAPES = [(16, 16), (8, 20), (20, 8)]
# (name, ra_over_dec pixel-size ratio): iso = square pixels, 2:1 = anisotropic
ANISO = [("iso", 1.0), ("2to1", 2.0)]


def _wcs(shape, ratio):
    n_x, n_y = shape
    fov = (ratio * n_x * u.arcsec, 1.0 * n_y * u.arcsec)
    return WcsAstropy(center=CENTER, shape=shape, fov=fov), fov


def test_scalar_geometry_broadcasts_and_legacy_properties_are_absent():
    wcs = WcsAstropy(center=CENTER, shape=12, fov=6 * u.arcsec)
    assert wcs.shape_xy == (12, 12)
    assert wcs.shape_yx == (12, 12)
    assert np.all(wcs.geometry.fov_yx == [6, 6] * u.arcsec)
    assert np.all(wcs.pixel_scales_xy == [0.5, 0.5] * u.arcsec)
    for legacy_name in ("shape", "fov", "distances"):
        assert not hasattr(wcs, legacy_name)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("aniso_name,ratio", ANISO)
def test_header_pairing(shape, aniso_name, ratio):
    """(a) FITS axis 1 <- public x and FITS axis 2 <- public y."""
    n_x, n_y = shape
    wcs, fov = _wcs(shape, ratio)
    h = wcs.to_header()
    assert h["CRPIX1"] == n_x / 2 + 0.5
    assert h["CRPIX2"] == n_y / 2 + 0.5
    assert h["CDELT1"] < 0, "axis 1 must be RA (negative CDELT)"
    assert np.isclose(-h["CDELT1"] * 3600, fov[0].to(u.arcsec).value / n_x)
    assert np.isclose(h["CDELT2"] * 3600, fov[1].to(u.arcsec).value / n_y)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("aniso_name,ratio", ANISO)
def test_pixel_scales_index_matched(shape, aniso_name, ratio):
    """(b) explicit XY pixel scales are index matched."""
    n_x, n_y = shape
    wcs, fov = _wcs(shape, ratio)
    d = wcs.pixel_scales_xy.to(u.arcsec).value
    assert np.allclose(
        d, [fov[0].to(u.arcsec).value / n_x, fov[1].to(u.arcsec).value / n_y]
    )


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("aniso_name,ratio", ANISO)
def test_extent_house_recipe(shape, aniso_name, ratio):
    """(c) extent() is plot-ready East-left without a transpose."""
    wcs, _ = _wcs(shape, ratio)
    g = wcs.geometry
    half_x = (g.n_ra * g.d_ra / 2).to_value(u.arcsec)
    half_y = (g.n_dec * g.d_dec / 2).to_value(u.arcsec)
    assert np.allclose(wcs.extent(u.arcsec), (half_x, -half_x, -half_y, half_y))


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("aniso_name,ratio", ANISO)
def test_world_span_coherence(shape, aniso_name, ratio):
    """(d) the world corners span exactly [0, shape[k]-1] on each array dim."""
    n_x, n_y = shape
    wcs, _ = _wcs(shape, ratio)
    corner_lo = wcs.pixel_to_world(0.0, 0.0)
    corner_hi = wcs.pixel_to_world(float(n_x - 1), float(n_y - 1))
    span = SkyCoord(
        ra=np.array([[corner_lo.ra.deg, corner_hi.ra.deg]]) * u.deg,
        dec=np.array([[corner_lo.dec.deg, corner_hi.dec.deg]]) * u.deg,
    )
    r = np.array(wcs.world_to_indices_yx(span)).reshape(2, 2)
    assert np.allclose(sorted(r[0]), [0, n_y - 1], atol=1e-6), r[0]
    assert np.allclose(sorted(r[1]), [0, n_x - 1], atol=1e-6), r[1]


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("aniso_name,ratio", ANISO)
def test_from_wcs_roundtrip(shape, aniso_name, ratio):
    """(e) WcsAstropy_from_wcs reads array_shape as (ny, nx) and reproduces."""
    n_x, n_y = shape
    wcs, fov = _wcs(shape, ratio)
    plain = WCS(wcs.to_header())
    plain.pixel_shape = (n_x, n_y)
    rebuilt = WcsAstropy_from_wcs(plain)
    assert rebuilt.shape_xy == (n_x, n_y)
    assert rebuilt.shape_yx == (n_y, n_x)
    fov_rb = rebuilt.geometry.fov_yx.to(u.arcsec).value
    assert np.allclose(
        fov_rb, [fov[1].to(u.arcsec).value, fov[0].to(u.arcsec).value], rtol=1e-3
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
    d_ra_as, d_dec_as = wcs.pixel_scales_xy.to(u.arcsec).value
    north = CENTER.spherical_offsets_by(0 * u.arcsec, 3 * d_dec_as * u.arcsec)
    east = CENTER.spherical_offsets_by(3 * d_ra_as * u.arcsec, 0 * u.arcsec)
    pts = SkyCoord(
        ra=np.array([[CENTER.ra.deg, north.ra.deg, east.ra.deg]]) * u.deg,
        dec=np.array([[CENTER.dec.deg, north.dec.deg, east.dec.deg]]) * u.deg,
    )
    idx = np.array(wcs.world_to_indices_yx(pts))
    c0, c1 = idx[0].ravel(), idx[1].ravel()  # order: center, north, east
    assert np.isclose(c0[1] - c0[0], 3.0), f"dim0 does not track +Dec: {c0}"
    assert np.isclose(c1[2] - c1[0], -3.0), f"dim1 does not track -RA: {c1}"


def test_positive_position_angle_is_north_through_east():
    wcs = WcsAstropy(
        CENTER, shape=(21, 21), fov=(21, 21) * u.arcsec,
        position_angle=30 * u.deg,
    )
    center = wcs.pixel_to_world(10, 10)
    row_up = wcs.pixel_to_world(10, 11)
    east, north = center.spherical_offsets_to(row_up)
    assert east > 0 * u.arcsec
    assert north > 0 * u.arcsec
    assert np.isclose(np.arctan2(east, north).to_value(u.deg), 30, atol=1e-3)
    with pytest.raises(ValueError, match="WCSAxes"):
        wcs.extent()
