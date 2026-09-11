"""jubik.wcs.frame: SpatialGeometry, the one owner of spatial conventions.

Every test uses a rectangle with anisotropic pixels. Square grids hide every
transpose, so none appear here except in the square-only guard test.
"""

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord

from jubik.parse.wcs.coordinate_system import CoordinateSystems
from jubik.wcs.frame import SpatialGeometry
from jubik.wcs.wcs_astropy import WcsAstropy

SHAPE_XY = (32, 24)
FOV_XY = (32.0, 12.0) * u.arcsec  # d_ra = 1", d_dec = 0.5"
CENTER = SkyCoord(ra=10.0 * u.deg, dec=-30.0 * u.deg)


@pytest.fixture
def geometry():
    return SpatialGeometry.from_xy(SHAPE_XY, FOV_XY)


# ---------------------------------------------------------------- construction
def test_from_xy_and_from_yx_agree(geometry):
    other = SpatialGeometry.from_yx((24, 32), (12.0, 32.0) * u.arcsec)
    assert geometry == other
    assert geometry != SpatialGeometry.from_yx((24, 32), (12.0, 32.0 + 1e-9) * u.arcsec), "equality is exact"
    assert geometry != SpatialGeometry.from_xy(SHAPE_XY, (32.0, 12.5) * u.arcsec)


def test_is_unhashable(geometry):
    # tolerant equality cannot honour the hash contract, so there is no hash
    with pytest.raises(TypeError):
        hash(geometry)


def test_named_accessors(geometry):
    assert geometry.n_ra == 32
    assert geometry.n_dec == 24
    assert geometry.shape_yx == (24, 32)
    assert geometry.shape_xy == (32, 24)
    assert u.isclose(geometry.d_ra, 1.0 * u.arcsec)
    assert u.isclose(geometry.d_dec, 0.5 * u.arcsec)
    assert u.allclose(geometry.pixel_scales_yx, (0.5, 1.0) * u.arcsec)
    assert u.allclose(geometry.pixel_scales_xy, (1.0, 0.5) * u.arcsec)
    assert u.allclose(geometry.fov_xy, FOV_XY)


def test_scalar_inputs_broadcast():
    g = SpatialGeometry.from_xy(16, 8 * u.arcmin)
    assert g.shape_yx == (16, 16)
    assert u.allclose(g.fov_yx, (8, 8) * u.arcmin)


@pytest.mark.parametrize(
    "shape, fov",
    [
        ((32, 24, 3), FOV_XY),
        ((32, 0), FOV_XY),
        ((32, -24), FOV_XY),
        ((32.5, 24), FOV_XY),
        ((32, 24.0001), FOV_XY),
        (SHAPE_XY, (32.0, 12.0, 1.0) * u.arcsec),
        (SHAPE_XY, (32.0, 0.0) * u.arcsec),
    ],
)
def test_rejects_malformed_input(shape, fov):
    with pytest.raises(ValueError):
        SpatialGeometry.from_xy(shape, fov)


def test_rejects_non_angular_fov():
    with pytest.raises(u.UnitConversionError):
        SpatialGeometry.from_xy(SHAPE_XY, (32.0, 12.0) * u.m)


def test_is_frozen(geometry):
    with pytest.raises(AttributeError):
        geometry.shape_yx = (1, 1)
    with pytest.raises(ValueError):
        geometry.fov_yx[0] = 1 * u.arcsec


def test_validates_before_coercing():
    with pytest.raises(ValueError, match="integers"):
        SpatialGeometry.from_xy((v for v in (2.5, 3)), 1 * u.arcsec)
    with pytest.raises(ValueError, match="integers"):
        SpatialGeometry.from_xy((2.5, 3), 1 * u.arcsec)
    with pytest.raises(ValueError):
        SpatialGeometry.from_xy(True, 1 * u.arcsec)
    with pytest.raises(ValueError, match="integers"):
        SpatialGeometry.from_xy((2.0, 3), 1 * u.arcsec)
    assert SpatialGeometry.from_xy((2, np.int64(3)), 1 * u.arcsec).shape_yx == (3, 2)
    assert SpatialGeometry.from_xy(iter((4, 6)), 1 * u.arcsec).shape_yx == (6, 4)


# ---------------------------------------------------------------- imshow extent
def test_imshow_extent_matches_wcs_astropy(geometry):
    wcs = WcsAstropy(center=CENTER, shape=SHAPE_XY, fov=FOV_XY)
    assert geometry.imshow_extent() == pytest.approx(wcs.extent())
    east, west, south, north = geometry.imshow_extent()
    assert east > west, "East is on the left"
    assert north > south
    assert east == pytest.approx(16.0)
    assert north == pytest.approx(6.0)


# ---------------------------------------------------------------- derived grids
def test_padded_keeps_pixel_size(geometry):
    padded = geometry.padded(1.5, fft_friendly=lambda n: n)
    assert padded.shape_yx == (36, 48)
    assert u.allclose(padded.pixel_scales_yx, geometry.pixel_scales_yx)
    assert u.allclose(padded.fov_yx, (18.0, 48.0) * u.arcsec)


def test_padded_default_is_fft_friendly(geometry):
    padded = geometry.padded(1.3)
    assert padded.n_dec >= int(24 * 1.3)
    assert padded.n_ra >= int(32 * 1.3)
    assert u.allclose(padded.pixel_scales_yx, geometry.pixel_scales_yx)


def test_padded_rejects_shrinking(geometry):
    with pytest.raises(ValueError):
        geometry.padded(0.9)


def test_index_grids(geometry):
    grid = geometry.index_grid_yx()
    assert grid.shape == (2, 24, 32)
    assert grid[0, 5, 0] == 5 and grid[0, 5, 31] == 5, "grid[0] is the row index"
    assert grid[1, 0, 7] == 7 and grid[1, 23, 7] == 7, "grid[1] is the column index"
    column, row = geometry.index_grid_xy()
    assert column.shape == row.shape == (24, 32)
    np.testing.assert_array_equal(column, grid[1])
    np.testing.assert_array_equal(row, grid[0])


def test_index_grid_xy_feeds_pixel_to_world(geometry):
    wcs = WcsAstropy(center=CENTER, shape=SHAPE_XY, fov=FOV_XY)
    world = wcs.pixel_to_world(*geometry.index_grid_xy())
    assert world.shape == geometry.shape_yx
    # column increases West: RA decreases along axis 1
    assert np.all(np.diff(world.ra.deg, axis=1) < 0)
    # row increases North: Dec increases along axis 0
    assert np.all(np.diff(world.dec.deg, axis=0) > 0)
