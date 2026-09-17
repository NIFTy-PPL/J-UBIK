"""Seam: ``SpatialGeometry`` and ``WcsAstropy`` own the XY <-> YX conversion.

Design page claims pinned here (docs/source/user/canonical-sky-design.md):

- "The public API speaks Cartesian order. shape=(nx, ny), fov=(fov_x, fov_y)."
- "A field is stored as sky[..., dec, ra]": ``shape_yx == (ny, nx)``.
- "CRPIX1 is the RA axis and has size nx."  (FITS output seam)
- "imshow(sky, origin='lower', extent=wcs.extent()) gives North up and East
  left with no transpose."
- ``world_to_indices_yx``: dim 0 tracks +Dec, dim 1 tracks -RA, each at its
  own pixel size.
- ``WcsAstropy_from_wcs`` reads astropy's ``array_shape`` as ``(ny, nx)``.

Every claim is checked on square and both-orientation rectangular grids
crossed with isotropic and 2:1 anisotropic pixels, so nothing can pass by a
square-grid or isotropic cancellation.
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
def test_public_xy_maps_to_storage_yx(shape, aniso_name, ratio):
    """shape=(nx, ny) in, shape_yx=(ny, nx) and pixel_scales in both orders out."""
    n_x, n_y = shape
    wcs, fov = _wcs(shape, ratio)
    assert wcs.shape_xy == (n_x, n_y)
    assert wcs.shape_yx == (n_y, n_x)
    d_xy = wcs.pixel_scales_xy.to(u.arcsec).value
    d_yx = wcs.pixel_scales_yx.to(u.arcsec).value
    fx, fy = fov[0].to(u.arcsec).value, fov[1].to(u.arcsec).value
    assert np.allclose(d_xy, [fx / n_x, fy / n_y])
    assert np.allclose(d_yx, [fy / n_y, fx / n_x])
    g = wcs.geometry
    assert (g.n_ra, g.n_dec) == (n_x, n_y)
    assert np.isclose(g.d_ra.to_value(u.arcsec), fx / n_x)
    assert np.isclose(g.d_dec.to_value(u.arcsec), fy / n_y)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("aniso_name,ratio", ANISO)
def test_fits_axis1_is_ra_sized_by_nx(shape, aniso_name, ratio):
    """FITS output seam: axis 1 is RA with size nx, axis 2 is Dec with size ny."""
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
def test_extent_is_north_up_east_left(shape, aniso_name, ratio):
    """extent() = (+half_x, -half_x, -half_y, +half_y): East left, no transpose."""
    wcs, _ = _wcs(shape, ratio)
    g = wcs.geometry
    half_x = (g.n_ra * g.d_ra / 2).to_value(u.arcsec)
    half_y = (g.n_dec * g.d_dec / 2).to_value(u.arcsec)
    assert np.allclose(wcs.extent(u.arcsec), (half_x, -half_x, -half_y, half_y))


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("aniso_name,ratio", ANISO)
def test_world_corners_span_the_array_dims(shape, aniso_name, ratio):
    """The world corners land on [0, ny-1] along dim 0 and [0, nx-1] along dim 1."""
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
def test_indices_yx_track_north_on_dim0_and_east_on_negative_dim1(
    shape, aniso_name, ratio
):
    """Three Dec pixels North is +3 on dim 0; three RA pixels East is -3 on dim 1.

    The offsets are measured in per-axis pixel sizes, so this also pins which
    pixel size divides which direction.  Only the anisotropic case can see
    that.
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


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("aniso_name,ratio", ANISO)
def test_from_wcs_reads_array_shape_as_ny_nx(shape, aniso_name, ratio):
    """WcsAstropy_from_wcs reproduces shape and fov from a plain astropy WCS."""
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
