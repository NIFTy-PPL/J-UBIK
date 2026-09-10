"""jubik.wcs.frame: SpatialGeometry, the one owner of spatial conventions.

Every test uses a rectangle with anisotropic pixels. Square grids hide every
transpose, so none appear here except in the square-only guard test.
"""

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS

from jubik.parse.wcs.coordinate_system import CoordinateSystems
from jubik.wcs.frame import Layout, SpatialGeometry
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
    assert hash(geometry) == hash(other)


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
    assert u.isclose(geometry.dvol, 0.5 * u.arcsec**2)


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


# ---------------------------------------------------------------- FITS header rule
@pytest.mark.parametrize("pa", [0.0, 30.0, -117.5] * u.deg)
@pytest.mark.parametrize("cs", [CoordinateSystems.icrs, CoordinateSystems.fk5, CoordinateSystems.galactic])
def test_fits_header_matches_wcs_astropy(geometry, pa, cs):
    """Same header WcsAstropy has always written, compared after astropy's own
    normalisation (identity PC cards dropped, EQUINOX coerced to float)."""
    reference = WcsAstropy(
        center=CENTER, shape=SHAPE_XY, fov=FOV_XY, position_angle=pa, coordinate_system=cs.value
    ).to_header()
    header = WCS(geometry.fits_header(CENTER, pa, cs)).to_header()
    assert set(header.keys()) == set(reference.keys())
    for key in reference:
        if isinstance(reference[key], float):
            assert header[key] == pytest.approx(reference[key], abs=0, rel=1e-13), key
        else:
            assert header[key] == reference[key], key


def test_header_orients_east_left_north_up(geometry):
    """External anchor: astropy resolves the header, not our own arithmetic."""
    header = geometry.fits_header(CENTER)
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


def test_from_astropy_wcs_round_trips_rectangle(geometry):
    wcs = WcsAstropy(center=CENTER, shape=SHAPE_XY, fov=FOV_XY, position_angle=30 * u.deg)
    wcs.pixel_shape = geometry.shape_xy
    recovered = SpatialGeometry.from_astropy_wcs(wcs)
    assert recovered.shape_yx == geometry.shape_yx
    assert u.allclose(recovered.fov_yx, geometry.fov_yx, rtol=1e-9)


def test_from_astropy_wcs_handles_cd_matrix(geometry):
    header = geometry.fits_header(CENTER, 30 * u.deg)
    wcs = WCS(header)
    cd = wcs.wcs.cdelt[:, None] * wcs.wcs.get_pc()
    header_cd = {k: v for k, v in header.items() if not k.startswith(("PC", "CDELT"))}
    header_cd.update(CD1_1=cd[0, 0], CD1_2=cd[0, 1], CD2_1=cd[1, 0], CD2_2=cd[1, 1])
    header_cd.update(NAXIS=2, NAXIS1=geometry.n_ra, NAXIS2=geometry.n_dec)
    recovered = SpatialGeometry.from_fits_header(header_cd)
    assert recovered == geometry


def test_from_astropy_wcs_requires_shape(geometry):
    with pytest.raises(ValueError):
        SpatialGeometry.from_astropy_wcs(WCS(geometry.fits_header(CENTER)))


# ---------------------------------------------------------------- imshow extent
def test_imshow_extent_matches_wcs_astropy(geometry):
    wcs = WcsAstropy(center=CENTER, shape=SHAPE_XY, fov=FOV_XY)
    assert geometry.imshow_extent() == pytest.approx(wcs.extent())
    east, west, south, north = geometry.imshow_extent()
    assert east > west, "East is on the left"
    assert north > south
    assert east == pytest.approx(16.0)
    assert north == pytest.approx(6.0)


# ---------------------------------------------------------------- layouts
def test_switch_layout_canonical_to_gridder_is_transpose(geometry):
    sky = np.random.default_rng(0).normal(size=(3, 24, 32))
    lm = geometry.switch_layout(sky, Layout.CANONICAL_YX, Layout.GRIDDER_LM)
    assert lm.shape == (3, 32, 24)
    np.testing.assert_array_equal(lm, np.swapaxes(sky, -1, -2))
    back = geometry.switch_layout(lm, Layout.GRIDDER_LM, Layout.CANONICAL_YX)
    np.testing.assert_array_equal(back, sky)


def test_switch_layout_fits_is_identity(geometry):
    sky = np.zeros((24, 32))
    assert geometry.switch_layout(sky, Layout.CANONICAL_YX, Layout.FITS_IMAGE) is sky


def test_switch_layout_rejects_wrong_trailing_shape(geometry):
    transposed_by_mistake = np.zeros((32, 24))
    with pytest.raises(ValueError, match="does not match CANONICAL_YX"):
        geometry.switch_layout(transposed_by_mistake, Layout.CANONICAL_YX, Layout.GRIDDER_LM)


def test_switch_layout_rejects_unmeasured_pair(geometry):
    with pytest.raises(ValueError, match="no layout switch"):
        geometry.switch_layout(np.zeros((24, 32)), Layout.FITS_IMAGE, Layout.GRIDDER_LM)


def test_switch_layout_works_on_jax_arrays(geometry):
    jnp = pytest.importorskip("jax.numpy")
    sky = jnp.arange(24 * 32, dtype=float).reshape(24, 32)
    lm = geometry.switch_layout(sky, Layout.CANONICAL_YX, Layout.GRIDDER_LM)
    assert lm.shape == (32, 24)
    assert float(lm[5, 7]) == float(sky[7, 5])


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


def test_crop_to_recovers_shape(geometry):
    padded = geometry.padded(2.0, fft_friendly=lambda n: n)
    field = np.arange(np.prod((5,) + padded.shape_yx)).reshape((5,) + padded.shape_yx)
    cropped = geometry.crop_to(field)
    assert cropped.shape == (5, 24, 32)
    np.testing.assert_array_equal(cropped, field[:, :24, :32])
    with pytest.raises(ValueError):
        geometry.crop_to(np.zeros((10, 10)))


def test_subsampled(geometry):
    sub = geometry.subsampled(3)
    assert sub.shape_yx == (72, 96)
    assert u.allclose(sub.fov_yx, geometry.fov_yx)
    assert u.isclose(sub.d_ra, geometry.d_ra / 3)


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


# ---------------------------------------------------------------- square guard
def test_require_square(geometry):
    with pytest.raises(ValueError, match="Chandra requires a square grid"):
        geometry.require_square("Chandra")
    assert SpatialGeometry.from_xy(64, 1 * u.arcmin).require_square("Chandra") == 64
