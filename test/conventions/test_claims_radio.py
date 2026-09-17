"""Claims: the resolve radio path lands in the canonical frame.

Design page claims pinned here (docs/source/user/canonical-sky-design.md):

- "A unit point source at a known pixel must produce visibilities matching
  the measurement equation, on anisotropic pixels": both backends, iso and
  2:1 pixels, a spread of offsets, and the full ``interferometry_response``
  on a genuine rectangle.
- "beams are built on the canonical grid" is in test_claims_sky_beamer; the
  dirty image of physical visibilities of the glyph comes out ``identity``
  on a rectangle here.
- External witness: the CASA-simulated observation "comes back as an F"
  through ``dirty_image``, and "the radio roundtrip now also correlates the
  forward model with the CASA visibilities directly, where no adjoint can
  cancel anything."

The CASA fixture uses gridder epsilon 1e-5 because it carries single
precision weights (ducc has no 1e-9 kernel for float32).
"""

import importlib.util
from types import SimpleNamespace

import astropy.units as u
import numpy as np
import pytest

import radio_fixture as rf
from glyph import dihedral_verdict, rasterize_canonical
from radio_anchor import (
    DUCC,
    OFFSETS,
    PIXSIZES,
    RECT_SHAPE,
    canonical_point_sky,
    disk_uvw,
    make_obs,
    point_cube,
    predicted_vis,
    rect_grid,
    stub_backend,
)

from jubik.grid import Grid
from jubik.instruments.resolve.dirty_image import dirty_image
from jubik.instruments.resolve.parse.response import Ducc0Settings
from jubik.instruments.resolve.response import (
    canonical_sky_to_visibilities,
    interferometry_response,
    interferometry_response_ducc,
)

finufft_required = pytest.mark.skipif(
    importlib.util.find_spec("jax_finufft") is None, reason="jax_finufft not installed"
)
BACKENDS = ["ducc", pytest.param("finufft", marks=finufft_required)]


# --- analytic anchor: point sources ------------------------------------------
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("pix_name,d_ra,d_dec", PIXSIZES)
@pytest.mark.parametrize("di,dj", OFFSETS)
def test_point_source_matches_measurement_equation(backend, pix_name, d_ra, d_dec, di, dj):
    apply = stub_backend(backend, d_ra, d_dec)
    vis = canonical_sky_to_visibilities(apply, canonical_point_sky(di, dj))
    np.testing.assert_allclose(
        vis, predicted_vis(di, dj, d_ra, d_dec), rtol=1e-4, atol=1e-13,
        err_msg=f"{backend}: canonical contract violated at offset {(di, dj)}",
    )


_RECT_UVW = np.array(
    [(3000.0, 0.0, 0.0), (7000.0, 0.0, 0.0), (0.0, 3000.0, 0.0),
     (0.0, 7000.0, 0.0), (2000.0, 4000.0, 0.0), (-4000.0, 2500.0, 0.0)]
)


def _rect_response():
    grid = rect_grid()
    nrow = _RECT_UVW.shape[0]
    R = interferometry_response(
        make_obs(np.zeros((1, nrow, 1)), _RECT_UVW), grid, DUCC
    )
    return grid, R


def test_rectangle_center_pixel_is_shape_over_two():
    """The phase-free pixel of the full response is (n_dec//2, n_ra//2).

    On a rectangle this pins that ``npix_y = shape_yx[0]`` and
    ``npix_x = shape_yx[1]`` are read the right way round.
    """
    _, R = _rect_response()
    n_dec, n_ra = RECT_SHAPE
    cands = [
        (n_dec // 2, n_ra // 2), (n_dec // 2 - 1, n_ra // 2 - 1),
        (n_dec // 2, n_ra // 2 - 1), (n_dec // 2 - 1, n_ra // 2),
    ]
    phases = {
        c: float(np.max(np.abs(np.angle(np.asarray(R(point_cube(RECT_SHAPE, *c)))))))
        for c in cands
    }
    center = min(phases, key=phases.get)
    assert phases[center] < 1e-6, f"no phase-free center pixel: {phases}"
    assert center == (n_dec // 2, n_ra // 2), center


@pytest.mark.parametrize("di,dj", OFFSETS)
def test_rectangle_response_matches_measurement_equation(di, dj):
    """Full ``interferometry_response`` on an anisotropic rectangle."""
    grid, R = _rect_response()
    n_dec, n_ra = RECT_SHAPE
    d_dec, d_ra = grid.spatial.pixel_scales_yx.to(u.rad).value
    vis = np.asarray(R(point_cube(RECT_SHAPE, n_dec // 2 + di, n_ra // 2 + dj))).ravel()
    np.testing.assert_allclose(
        vis, predicted_vis(di, dj, d_ra, d_dec, uvw=_RECT_UVW), rtol=1e-4, atol=1e-13,
        err_msg=f"rectangle response != measurement equation at offset {(di, dj)}",
    )


# --- glyph through dirty_image on a rectangle --------------------------------
def _pad_square(a):
    """Embed a rectangle in a centered square so every D4 element is defined.

    The shift-invariant correlation, and so the verdict, is unchanged.
    """
    n = max(a.shape)
    out = np.zeros((n, n))
    i0 = (n - a.shape[0]) // 2
    j0 = (n - a.shape[1]) // 2
    out[i0: i0 + a.shape[0], j0: j0 + a.shape[1]] = a
    return out


def _rect_glyph(grid, scale=1.5):
    d_dec, d_ra = grid.spatial.pixel_scales_yx.to(u.arcsec).value
    n_dec, n_ra = RECT_SHAPE
    return rasterize_canonical(
        RECT_SHAPE, (n_dec // 2, n_ra // 2), (d_dec, d_ra), scale=scale
    )


def _analytic_glyph_vis(grid, glyph, uvw):
    """Visibilities of the glyph in the external (CASA) convention."""
    n_dec, n_ra = RECT_SHAPE
    d_dec, d_ra = grid.spatial.pixel_scales_yx.to(u.rad).value
    V = np.zeros(uvw.shape[0], np.complex128)
    for i, j in np.argwhere(glyph > 0):
        V += glyph[i, j] * predicted_vis(i - n_dec // 2, j - n_ra // 2, d_ra, d_dec, uvw=uvw)
    return V


def test_dirty_image_of_physical_visibilities_is_identity_on_rectangle():
    grid = rect_grid()
    glyph = _rect_glyph(grid)
    uvw = disk_uvw()
    V = _analytic_glyph_vis(grid, glyph, uvw).reshape(1, uvw.shape[0], 1)

    dirty = dirty_image(make_obs(V, uvw), grid, DUCC, weighting="natural")
    img = np.real(np.asarray(dirty.value))[0, 0, 0]

    verdict, scores = dihedral_verdict(_pad_square(img), _pad_square(glyph))
    runner_up = max(v for k, v in scores.items() if k != verdict)
    assert verdict == "identity", (
        f"rectangle dirty image verdict {verdict!r}; "
        f"scores={ {k: round(v, 3) for k, v in scores.items()} }"
    )
    assert scores[verdict] - runner_up > 0.05, f"weak margin: {scores}"


def test_forward_then_dirty_is_identity():
    """R^H R has its autocorrelation peak at the origin.

    With the bilinear transpose instead of the Hermitian adjoint this came
    out ``rot180``.  It is the in-repo witness of the dirty_image seam.
    """
    grid = rect_grid()
    glyph = _rect_glyph(grid)
    uvw = disk_uvw()
    nrow = uvw.shape[0]
    R = interferometry_response(make_obs(np.zeros((1, nrow, 1)), uvw), grid, DUCC)
    vis = np.asarray(R(glyph.reshape((1, 1, 1) + RECT_SHAPE))).astype(np.complex128)

    dirty = dirty_image(make_obs(vis, uvw), grid, DUCC, weighting="natural")
    img = np.real(np.asarray(dirty.value))[0, 0, 0]

    verdict, _ = dihedral_verdict(_pad_square(img), _pad_square(glyph))
    assert verdict == "identity", verdict


# --- external witness: the CASA-simulated observation ------------------------
CASA_DUCC = Ducc0Settings(epsilon=1e-5, do_wgridding=False, nthreads=1, verbosity=0)


def _casa_dirty(obs):
    fov = [rf.RECON_NPIX * rf.RECON_PIX_ARCSEC] * 2 * u.arcsec
    grid = Grid.from_shape_and_fov(
        (rf.RECON_NPIX, rf.RECON_NPIX), fov, frequencies=None,
        sky_center=rf.phase_center(),
    )
    dirty = dirty_image(obs, grid, CASA_DUCC, weighting="natural")
    return np.real(np.asarray(dirty.value))[0, 0, 0]


def test_casa_observation_comes_back_as_an_f(casa_observation):
    """CASA sky in, jubik dirty image out, dihedral verdict ``identity``.

    ``rot180`` would mean a flipped sign layer (uv or visibility
    conjugation), a transpose-family verdict an axis swap.  This is a
    convention finding: measure, do not patch production code from here.
    """
    img = _casa_dirty(casa_observation)
    truth = rasterize_canonical(
        (rf.RECON_NPIX, rf.RECON_NPIX),
        (rf.RECON_NPIX // 2, rf.RECON_NPIX // 2),
        (rf.RECON_PIX_ARCSEC, rf.RECON_PIX_ARCSEC),
        scale=rf.GLYPH_SCALE,
    )
    verdict, scores = dihedral_verdict(img, truth)
    runner_up = max(v for k, v in scores.items() if k != verdict)
    assert verdict == "identity", (
        f"CASA roundtrip verdict {verdict!r}; "
        f"scores={ {k: round(v, 3) for k, v in scores.items()} }"
    )
    assert scores[verdict] - runner_up > 0.05, f"weak margin: {scores}"


def test_forward_model_matches_casa_visibilities_directly(casa_observation):
    """corr(R(truth), data) > 0.99 and corr(conj(R(truth)), data) < 0.5.

    The dirty image cannot see a conjugation in the forward model, because
    the adjoint conjugates once more and the two cancel.  Correlating in the
    visibility domain can.
    """
    obs = casa_observation
    hdu = rf.truth_hdu()
    truth = np.squeeze(hdu.data).astype(np.float64)
    dpix_rad = abs(hdu.header["CDELT1"]) * np.pi / 180.0

    stub = SimpleNamespace(uvw=np.asarray(obs.uvw), freq=np.asarray(obs.freq))
    backend = interferometry_response_ducc(
        stub, npix_x=truth.shape[1], npix_y=truth.shape[0],
        pixsize_x=dpix_rad, pixsize_y=dpix_rad,
        do_wgridding=False, epsilon=1e-5, nthreads=1, verbosity=0,
    )
    v_model = np.asarray(canonical_sky_to_visibilities(backend, truth)).ravel()
    d = np.asarray(obs.vis_val[0]).ravel()

    def corr(a, b):
        return float(np.abs(np.vdot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b)))

    direct, conjugated = corr(v_model, d), corr(np.conj(v_model), d)
    assert direct > 0.99, (
        f"forward model does not match CASA visibilities (corr {direct:.3f}); "
        f"conjugated corr {conjugated:.3f}. A high conjugated value means a "
        "spurious conjugation in the response."
    )
    assert conjugated < 0.5, f"conjugated model also correlates ({conjugated:.3f})"
