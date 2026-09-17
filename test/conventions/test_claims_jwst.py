"""Claims: the JWST path lands data in the canonical frame.

Design page claims pinned here (docs/source/user/canonical-sky-design.md):

- JWST loader seam: "gwcs pixel centers go through world_to_indices_yx onto
  the reconstruction grid; interpolators take yx indices only."  A bump one
  row up is read by a point North of center, a bump one column left by a
  point East.
- External witness: a synthetic ImageModel with the F painted through its own
  gwcs "comes back through the real loader as an F, not a mirrored or
  rotated F."  The loader chain is JwstData -> bounding indices ->
  subsample_pixel_centers -> world_to_indices_yx, then a scatter onto the
  canonical grid.
"""

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord

from glyph import dihedral_verdict, rasterize_canonical

from jubik.instruments.jwst.rotation_and_shift.linear_rotation_and_shift import (
    build_linear_rotation_and_shift,
)
from jubik.wcs import subsample_pixel_centers
from jubik.wcs.wcs_astropy import WcsAstropy

CENTER = SkyCoord(ra=10.0 * u.deg, dec=20.0 * u.deg)

# (name, shape_xy, fov_xy): a square with 1" pixels and an anisotropic
# rectangle with 1" Dec pixels and 2" RA pixels
GRIDS = [
    ("square", (9, 9), (9 * u.arcsec, 9 * u.arcsec)),
    ("rect_aniso", (11, 7), (22 * u.arcsec, 7 * u.arcsec)),
]


def _reader(wcs):
    """Interpolate a canonical sky at (center, 2 Dec-px North, 2 RA-px East)."""
    d_ra, d_dec = wcs.pixel_scales_xy.to(u.arcsec).value
    north = CENTER.spherical_offsets_by(0 * u.arcsec, 2 * d_dec * u.arcsec)
    east = CENTER.spherical_offsets_by(2 * d_ra * u.arcsec, 0 * u.arcsec)
    pts = SkyCoord(
        ra=np.array([[CENTER.ra.deg, north.ra.deg, east.ra.deg]]) * u.deg,
        dec=np.array([[CENTER.dec.deg, north.dec.deg, east.dec.deg]]) * u.deg,
    )
    idx = np.array(wcs.world_to_indices_yx(pts))
    interp = build_linear_rotation_and_shift(out_shape=idx.shape[-2:], mode="constant")
    return lambda sky: np.asarray(interp(sky, idx)).ravel()


@pytest.mark.parametrize("name,shape_xy,fov", GRIDS)
def test_row_up_is_north(name, shape_xy, fov):
    wcs = WcsAstropy(center=CENTER, shape=shape_xy, fov=fov)
    n_dec, n_ra = wcs.shape_yx
    c0, c1 = n_dec // 2, n_ra // 2
    sky = np.zeros((n_dec, n_ra))
    sky[c0 + 2, c1] = 1.0
    reads = _reader(wcs)(sky)  # order: center, north, east
    assert np.allclose(reads, [0, 1, 0]), f"dim 0 is not +Dec: {reads}"


@pytest.mark.parametrize("name,shape_xy,fov", GRIDS)
def test_column_left_is_east(name, shape_xy, fov):
    wcs = WcsAstropy(center=CENTER, shape=shape_xy, fov=fov)
    n_dec, n_ra = wcs.shape_yx
    c0, c1 = n_dec // 2, n_ra // 2
    sky = np.zeros((n_dec, n_ra))
    sky[c0, c1 - 2] = 1.0
    reads = _reader(wcs)(sky)
    assert np.allclose(reads, [0, 0, 1]), f"dim 1 is not -RA: {reads}"


# --- external witness: synthetic datamodel through the production loader -----
RECON_SHAPE = (128, 128)
RECON_FOV = 16.0 * u.arcsec
RECON_PIXSIZE_ARCSEC = 0.125  # RECON_FOV / RECON_SHAPE


def _roundtrip(path):
    import jwst_fixture
    from jubik.instruments.jwst.data.jwst_data import JwstData

    center = jwst_fixture.field_center()
    jd = JwstData(str(path))
    recon = WcsAstropy(center=center, shape=RECON_SHAPE, fov=(RECON_FOV, RECON_FOV))

    bounds = jd.wcs.bounding_indices_from_world_extrema(recon.world_corners())
    min_row, max_row, min_col, max_col = bounds
    cutout = jd.dm.data[min_row:max_row, min_col:max_col]

    centers = subsample_pixel_centers(bounds, jd.wcs, subsample=1)
    idx = np.array(recon.world_to_indices_yx(centers))

    scattered = np.zeros(RECON_SHAPE)
    ii = np.round(idx[0]).astype(int)
    jj = np.round(idx[1]).astype(int)
    inb = (ii >= 0) & (ii < RECON_SHAPE[0]) & (jj >= 0) & (jj < RECON_SHAPE[1])
    np.add.at(scattered, (ii[inb], jj[inb]), cutout[inb])

    x, y = recon.world_to_pixel(center)
    anchor = (int(round(float(y))), int(round(float(x))))  # canonical (i, j)
    truth = rasterize_canonical(
        RECON_SHAPE, anchor, (RECON_PIXSIZE_ARCSEC, RECON_PIXSIZE_ARCSEC),
        scale=jwst_fixture.GLYPH_SCALE,
    )
    return scattered, truth


def test_datamodel_comes_back_as_an_f(jwst_datamodel_path):
    """Dihedral verdict ``identity``: no transpose or flip in the loader chain."""
    scattered, truth = _roundtrip(jwst_datamodel_path)
    assert (scattered > 0).sum() > 0, "nothing landed on the reconstruction grid"
    verdict, scores = dihedral_verdict(scattered, truth)
    runner_up = max(v for k, v in scores.items() if k != verdict)
    assert verdict == "identity", (
        f"JWST loader chain roundtrip is {verdict!r}, not identity; "
        f"scores={ {k: round(v, 3) for k, v in scores.items()} }"
    )
    assert scores[verdict] - runner_up > 0.05, f"weak margin: {scores}"

