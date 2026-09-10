"""WcsAstropy owns one SpatialGeometry and adds astrometry to it."""

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS

from jubik.wcs.frame import SpatialGeometry
from jubik.parse.wcs.coordinate_system import CoordinateSystems
from jubik.wcs.wcs_astropy import WcsAstropy, WcsAstropy_from_wcs, _apply_header, fits_header, geometry_from_wcs

SHAPE_XY = (32, 24)
FOV_XY = (32.0, 12.0) * u.arcsec
CENTER = SkyCoord(ra=10.0 * u.deg, dec=-30.0 * u.deg)


@pytest.fixture
def wcs():
    return WcsAstropy(center=CENTER, shape=SHAPE_XY, fov=FOV_XY, position_angle=30 * u.deg)


def test_owns_a_geometry(wcs):
    assert isinstance(wcs.geometry, SpatialGeometry)
    assert wcs.geometry == SpatialGeometry.from_xy(SHAPE_XY, FOV_XY)
    assert wcs.shape_yx == wcs.geometry.shape_yx == (24, 32)
    assert wcs.shape_xy == (32, 24)
    assert u.allclose(wcs.pixel_scales_yx, (0.5, 1.0) * u.arcsec)
    assert wcs.array_shape == (24, 32), "astropy knows the array shape"
    assert u.isclose(wcs.dvol, (0.5 * u.arcsec**2).to(u.deg**2))
    assert wcs.dvol.unit == u.deg**2


def test_geometry_is_read_only(wcs):
    with pytest.raises(AttributeError):
        wcs.shape_yx = (1, 1)


def test_from_geometry_matches_constructor(wcs):
    other = WcsAstropy.from_geometry(wcs.geometry, CENTER, 30 * u.deg)
    assert other.to_header() == wcs.to_header()


def test_extent_forwards_to_geometry():
    flat = WcsAstropy(center=CENTER, shape=SHAPE_XY, fov=FOV_XY)
    assert flat.extent() == flat.geometry.imshow_extent()
    rotated = WcsAstropy(center=CENTER, shape=SHAPE_XY, fov=FOV_XY, position_angle=30 * u.deg)
    with pytest.raises(ValueError, match="position_angle=0"):
        rotated.extent()


def test_world_corners_span_the_rectangle():
    # Position angle zero: with anisotropic pixels the FITS PC convention applies
    # CDELT after the rotation, so a rotated grid is sheared and edge lengths are
    # no longer n * d. That is the existing header rule, pinned by the goldens.
    wcs = WcsAstropy(center=CENTER, shape=SHAPE_XY, fov=FOV_XY)
    corners = wcs.world_corners()
    assert len(corners) == 4
    g = wcs.geometry
    # points are ((xmin, ymin), (xmin, ymax), (xmax, ymin), (xmax, ymax))
    assert u.isclose(corners[0].separation(corners[2]), (g.n_ra - 1) * g.d_ra, rtol=1e-3)
    assert u.isclose(corners[0].separation(corners[1]), (g.n_dec - 1) * g.d_dec, rtol=1e-3)
    assert corners[2].ra < corners[0].ra, "xmax corner is West"
    assert corners[1].dec > corners[0].dec, "ymax corner is North"


@pytest.mark.parametrize("pa", [0.0, 30.0, -117.5] * u.deg)
def test_from_wcs_round_trips_pc_header(pa):
    original = WcsAstropy(center=CENTER, shape=SHAPE_XY, fov=FOV_XY, position_angle=pa)
    rebuilt = WcsAstropy_from_wcs(original)
    assert rebuilt.geometry.shape_yx == original.geometry.shape_yx
    assert u.allclose(rebuilt.geometry.fov_yx, original.geometry.fov_yx, rtol=1e-12)
    assert u.isclose(rebuilt.position_angle, pa, atol=1e-9 * u.deg)
    assert rebuilt.to_header() == original.to_header()


def test_from_wcs_round_trips_cd_header():
    original = WcsAstropy(center=CENTER, shape=SHAPE_XY, fov=FOV_XY, position_angle=30 * u.deg)
    header = original.to_header()
    cd = original.pixel_scale_matrix
    for key in [k for k in header if k.startswith(("PC", "CDELT"))]:
        del header[key]
    header.update(CD1_1=cd[0, 0], CD1_2=cd[0, 1], CD2_1=cd[1, 0], CD2_2=cd[1, 1])
    header.update(NAXIS=2, NAXIS1=SHAPE_XY[0], NAXIS2=SHAPE_XY[1])
    rebuilt = WcsAstropy_from_wcs(WCS(header))
    assert rebuilt.geometry.shape_yx == original.geometry.shape_yx
    assert u.allclose(rebuilt.geometry.fov_yx, original.geometry.fov_yx, rtol=1e-12)
    assert u.isclose(rebuilt.position_angle, 30 * u.deg, atol=1e-9 * u.deg)


# ---------------------------------------------------------------- the header rule
@pytest.fixture
def geometry():
    return SpatialGeometry.from_xy(SHAPE_XY, FOV_XY)


@pytest.mark.parametrize("pa", [0.0, 30.0, -117.5] * u.deg)
@pytest.mark.parametrize("cs", [CoordinateSystems.icrs, CoordinateSystems.fk5, CoordinateSystems.galactic])
def test_fits_header_is_what_wcs_astropy_writes(geometry, pa, cs):
    """The header rule applied to a plain WCS matches what WcsAstropy writes,
    compared after astropy's own normalisation (identity PC cards dropped)."""
    reference = WcsAstropy(
        center=CENTER, shape=SHAPE_XY, fov=FOV_XY, position_angle=pa, coordinate_system=cs.value
    ).to_header()
    plain = WCS(naxis=2)
    _apply_header(plain, fits_header(geometry, CENTER, pa, cs))
    header = plain.to_header()
    assert set(header.keys()) == set(reference.keys())
    for key in reference:
        if isinstance(reference[key], float):
            assert header[key] == pytest.approx(reference[key], abs=0, rel=1e-13), key
        else:
            assert header[key] == reference[key], key


def test_header_orients_east_left_north_up(geometry):
    """External anchor: astropy resolves the header, not our own arithmetic."""
    header = fits_header(geometry, CENTER)
    assert header["CDELT1"] < 0
    assert header["CDELT2"] > 0
    wcs = WCS(header)
    column, row = geometry.n_ra // 2, geometry.n_dec // 2
    here = wcs.pixel_to_world(column, row)
    west = wcs.pixel_to_world(column + 1, row)
    north = wcs.pixel_to_world(column, row + 1)
    assert west.ra < here.ra
    assert north.dec > here.dec
    # CDELT is a projection-plane size: one pixel step is one pixel scale on the sky
    assert u.isclose(here.separation(north), geometry.d_dec, rtol=1e-6)
    assert u.isclose(here.separation(west), geometry.d_ra, rtol=1e-6)


def test_geometry_from_wcs_round_trips_rectangle(geometry):
    wcs = WcsAstropy(center=CENTER, shape=SHAPE_XY, fov=FOV_XY, position_angle=30 * u.deg)
    recovered = geometry_from_wcs(wcs)
    assert recovered.shape_yx == geometry.shape_yx
    assert u.allclose(recovered.fov_yx, geometry.fov_yx, rtol=1e-12)


def test_geometry_from_wcs_handles_cd_matrix(geometry):
    header = fits_header(geometry, CENTER, 30 * u.deg)
    wcs = WCS(header)
    cd = wcs.wcs.cdelt[:, None] * wcs.wcs.get_pc()
    header_cd = {k: v for k, v in header.items() if not k.startswith(("PC", "CDELT"))}
    header_cd.update(CD1_1=cd[0, 0], CD1_2=cd[0, 1], CD2_1=cd[1, 0], CD2_2=cd[1, 1])
    header_cd.update(NAXIS=2, NAXIS1=geometry.n_ra, NAXIS2=geometry.n_dec)
    recovered = geometry_from_wcs(WCS(header_cd))
    assert recovered.shape_yx == geometry.shape_yx
    assert u.allclose(recovered.fov_yx, geometry.fov_yx, rtol=1e-12)


def test_geometry_from_wcs_requires_shape(geometry):
    with pytest.raises(ValueError):
        geometry_from_wcs(WCS(fits_header(geometry, CENTER)))


def test_equinox_is_honoured_as_float():
    fk4 = WcsAstropy(center=CENTER, shape=SHAPE_XY, fov=FOV_XY, coordinate_system=CoordinateSystems.fk4.value)
    assert fk4.wcs.equinox == 1950.0
    fk5 = WcsAstropy(center=CENTER, shape=SHAPE_XY, fov=FOV_XY, coordinate_system=CoordinateSystems.fk5.value)
    assert fk5.wcs.equinox == 2000.0
    assert fk5.to_header()["EQUINOX"] == 2000.0
