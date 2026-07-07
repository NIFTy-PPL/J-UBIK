"""Radio canonical-frame sweep — adapter contract, rectangle response, adjoint.

Breadth twin of ``probes/p3``/``p4``/``p7`` for the resolve radio path.  Where
the probes pin the adapter seam and the square roundtrip, this module adds:

  (a) the canonical_sky_to_visibilities adapter contract across BOTH backends
      (ducc, finufft), isotropic and anisotropic pixels, and a spread of
      source offsets (the p4 anchor, widened);
  (b) a RESPONSE-LEVEL pin on a genuine RECTANGLE (nDec != nRA) — the
      interferometry_response reading npix_x = shape[1] / pixsize_x =
      distances[1] on a non-square grid, which NO probe covers;
  (c) the dirty_image adjoint identity for the conj-containing response;
  (d) dirty-image orientation on a rectangle (glyph roundtrip, golden-free).

The measurement-equation anchor (same as p3/p4): a unit point source ``di``
pixels North and ``dj`` pixels West of center (canonical ``sky[c+di, c+dj]=1``)
produces

    V(u, v) = d_ra * d_dec * exp(+2*pi*i * (u*l + v*m)),
    m = +di * d_dec  (North),   l = -dj * d_ra  (dj increases West).

The ``+2*pi*i`` sign is the effective measurement equation the shipped
response EMITS directly (Batch E, 2026-07-07): the adapter is a pure axis
transpose, no conjugation, matching upstream ``resolve`` and the CASA/M51
external witnesses (see ``probes/p4``/``p7``).
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import importlib.util
from types import SimpleNamespace

import astropy.units as u
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from astropy.coordinates import SkyCoord

from glyph import dihedral_verdict, rasterize_canonical  # via conftest sys.path

from jubik.grid import Grid
from jubik.instruments.resolve.data import Observation
from jubik.instruments.resolve.data.antenna_positions import AntennaPositions
from jubik.instruments.resolve.dirty_image import dirty_image
from jubik.instruments.resolve.parse.response import Ducc0Settings
from jubik.instruments.resolve.response import (
    canonical_sky_to_visibilities,
    interferometry_response,
    interferometry_response_ducc,
    interferometry_response_finufft,
)
from jubik.polarization import Polarization

C_LIGHT = 299792458.0
_HAS_FINUFFT = importlib.util.find_spec("jax_finufft") is not None
finufft_required = pytest.mark.skipif(
    not _HAS_FINUFFT, reason="jax_finufft not installed"
)


# ==========================================================================
# (a) adapter contract sweep — the p4 anchor, widened across backends/pixels
# ==========================================================================
NPIX = 32
# freq = c => u_meters == u_lambda; explicit baselines incl. mixed and neg-u
UVW = np.array(
    [
        (3000.0, 0.0, 0.0),
        (7000.0, 0.0, 0.0),
        (0.0, 3000.0, 0.0),
        (0.0, 7000.0, 0.0),
        (2000.0, 4000.0, 0.0),
        (-4000.0, 2500.0, 0.0),
        (-3500.0, -1500.0, 0.0),
    ]
)
STUB_OBS = SimpleNamespace(uvw=UVW, freq=np.array([C_LIGHT]))
OFFSETS = [(0, 0), (6, 0), (0, 4), (5, -3), (-4, -2)]
PIXSIZES = [("iso", 1.0e-5, 1.0e-5), ("aniso", 1.0e-5, 1.5e-5)]


def _canonical_point_sky(di, dj):
    c = NPIX // 2
    sky = np.zeros((NPIX, NPIX))
    sky[c + di, c + dj] = 1.0
    return sky


def _predicted(di, dj, d_ra, d_dec):
    l, m = -dj * d_ra, +di * d_dec
    uu, vv = UVW[:, 0], UVW[:, 1]
    return d_ra * d_dec * np.exp(+2j * np.pi * (uu * l + vv * m))


def _stub_backend(backend, d_ra, d_dec):
    if backend == "ducc":
        op = interferometry_response_ducc(
            STUB_OBS, npix_x=NPIX, npix_y=NPIX, pixsize_x=d_ra, pixsize_y=d_dec,
            do_wgridding=False, epsilon=1e-9, nthreads=1, verbosity=0,
        )
    elif backend == "finufft":
        op = interferometry_response_finufft(
            STUB_OBS, pixsize_x=d_ra, pixsize_y=d_dec, epsilon=1e-9,
            center_x=0.0, center_y=0.0,
        )
    else:
        raise ValueError(backend)
    return lambda s: np.asarray(op(s)).ravel()


@pytest.mark.parametrize("backend", ["ducc", pytest.param("finufft", marks=finufft_required)])
@pytest.mark.parametrize("pix_name,d_ra,d_dec", PIXSIZES)
@pytest.mark.parametrize("di,dj", OFFSETS)
def test_adapter_contract(backend, pix_name, d_ra, d_dec, di, dj):
    apply = _stub_backend(backend, d_ra, d_dec)
    vis = canonical_sky_to_visibilities(apply, _canonical_point_sky(di, dj))
    np.testing.assert_allclose(
        vis, _predicted(di, dj, d_ra, d_dec), rtol=1e-4, atol=1e-13,
        err_msg=f"{backend} adapter contract violated at offset {(di, dj)}",
    )


def test_adapter_does_real_work():
    """Control: the RAW backend alone must NOT satisfy the canonical contract."""
    d_ra, d_dec = 1.0e-5, 1.5e-5
    apply = _stub_backend("ducc", d_ra, d_dec)
    raw = apply(_canonical_point_sky(6, 4))
    assert not np.allclose(
        raw, _predicted(6, 4, d_ra, d_dec), rtol=1e-4, atol=1e-13
    ), "raw backend already satisfies the contract — adapter is a no-op?"


# ==========================================================================
# shared helpers for the response-level (b)/(c)/(d) tests
# ==========================================================================
CENTER = SkyCoord(ra=10.0 * u.deg, dec=20.0 * u.deg)
# RECTANGLE, anisotropic: nDec=24 rows of 1", nRA=32 cols of 1.5"
RECT_SHAPE = (24, 32)
RECT_FOV = (RECT_SHAPE[0] * 1.0 * u.arcsec, RECT_SHAPE[1] * 1.5 * u.arcsec)
DUCC = Ducc0Settings(epsilon=1e-9, do_wgridding=False, nthreads=1, verbosity=0)


def _rect_grid():
    return Grid.from_shape_and_fov(
        RECT_SHAPE, RECT_FOV, frequencies=None, sky_center=CENTER
    )


def _make_obs(vis, uvw):
    """Minimal single-channel, Stokes-I, imaging-only Observation."""
    return Observation(
        antenna_positions=AntennaPositions(uvw=np.asarray(uvw, np.float64).copy()),
        vis=np.asarray(vis, np.complex128),
        weight=np.ones(np.shape(vis), np.float64),
        polarization=Polarization.trivial(),
        freq=np.array([C_LIGHT]),
        auxiliary_tables=None,
    )


def _point_cube(shape, i, j):
    s = np.zeros((1, 1, 1) + tuple(shape))
    s[0, 0, 0, i, j] = 1.0
    return s


# ==========================================================================
# (b) response-level rectangle pin — new coverage (no probe hits this)
# ==========================================================================
def test_response_rectangle_center_pixel():
    """The response's phase-free center pixel is (shape[0]//2, shape[1]//2).

    Determined empirically, as p3 does: the source position with no residual
    phase across all baselines is the center pixel.  On a rectangle this pins
    that npix_y = shape[0] and npix_x = shape[1] are read the right way round.
    """
    grid = _rect_grid()
    nrow = 6
    uvw = np.array(
        [(3000.0, 0.0, 0.0), (7000.0, 0.0, 0.0), (0.0, 3000.0, 0.0),
         (0.0, 7000.0, 0.0), (2000.0, 4000.0, 0.0), (-4000.0, 2500.0, 0.0)]
    )
    R = interferometry_response(
        _make_obs(np.zeros((1, nrow, 1)), uvw), grid, DUCC
    )
    n_dec, n_ra = RECT_SHAPE
    cands = [
        (n_dec // 2, n_ra // 2), (n_dec // 2 - 1, n_ra // 2 - 1),
        (n_dec // 2, n_ra // 2 - 1), (n_dec // 2 - 1, n_ra // 2),
    ]
    phases = {
        c: float(np.max(np.abs(np.angle(np.asarray(R(_point_cube(RECT_SHAPE, *c)))))))
        for c in cands
    }
    center = min(phases, key=phases.get)
    assert phases[center] < 1e-6, f"no phase-free center pixel: {phases}"
    assert center == (n_dec // 2, n_ra // 2), center


@pytest.mark.parametrize("di,dj", [(0, 0), (6, 0), (0, 4), (5, -3), (-4, -2)])
def test_response_rectangle_measurement_equation(di, dj):
    """interferometry_response on a RECTANGLE == the analytic measurement eq.

    Pins response.py's ``npix_x = shape[1]`` / ``pixsize_x = distances[1]``
    (RA/dim1) reads on a non-square grid: a unit point ``di`` North / ``dj``
    West of the center pixel yields d_ra*d_dec*exp(+2πi(u*l + v*m)) with
    m = +di*d_dec, l = -dj*d_ra.
    """
    grid = _rect_grid()
    n_dec, n_ra = RECT_SHAPE
    c0, c1 = n_dec // 2, n_ra // 2
    d_dec, d_ra = grid.spatial.distances.to(u.rad).value  # index-matched [dim0, dim1]

    nrow = 6
    uvw = np.array(
        [(3000.0, 0.0, 0.0), (7000.0, 0.0, 0.0), (0.0, 3000.0, 0.0),
         (0.0, 7000.0, 0.0), (2000.0, 4000.0, 0.0), (-4000.0, 2500.0, 0.0)]
    )
    R = interferometry_response(
        _make_obs(np.zeros((1, nrow, 1)), uvw), grid, DUCC
    )
    vis = np.asarray(R(_point_cube(RECT_SHAPE, c0 + di, c1 + dj))).ravel()

    uu, vv = uvw[:, 0], uvw[:, 1]
    l, m = -dj * d_ra, +di * d_dec
    predicted = d_ra * d_dec * np.exp(+2j * np.pi * (uu * l + vv * m))
    np.testing.assert_allclose(
        vis, predicted, rtol=1e-4, atol=1e-13,
        err_msg=f"rectangle response != measurement eq at offset {(di, dj)}",
    )


# ==========================================================================
# (c) adjoint consistency of the C-linear response
# ==========================================================================
def test_response_adjoint_identity():
    """jax.linear_transpose adjoint pairing for the C-linear response.

    Since Batch E the adapter is a pure axis transpose (no ``jnp.conj``), so
    the response ``R`` is C-linear (holomorphic).  ``jax.linear_transpose``
    returns the transpose w.r.t. the BILINEAR pairing ``sum(a * b)`` (no
    conjugation).  R's INPUT domain is nonetheless REAL — the sky is a float
    array (``dtype_float2complex`` rejects a complex sky) — so the transpose,
    built at the real example primal, lands back in the real image domain:
    ``R^T(v)`` is real.  A real ``R^T(v)`` can only carry the REAL part of the
    complex bilinear pairing, so the identity that actually holds is

        Re( sum( R(s) * v ) )  ==  sum( s * R^T(v) ),    R^T(v) real.

    (Measured: dropping the ``Re`` leaves a residual imaginary part from
    ``sum(R(s)*v)`` that ``sum(s*R^T(v))`` — being real — cannot match.  The
    clean complex bilinear identity would need a COMPLEX sky domain, which the
    response does not have.)  The naive conjugated pairing
    ``Re(sum(R(s) * conj(v)))`` does NOT coincide either, documenting that the
    non-conjugating transpose is the right one.  This is exactly the pairing
    ``dirty_image`` relies on (it conjugates the visibility cotangent to form
    the Hermitian adjoint ``R^H``).
    """
    grid = _rect_grid()
    rng = np.random.default_rng(3)
    nrow = 40
    uvw = rng.normal(size=(nrow, 3)) * 3000.0
    uvw[:, 2] = 0.0
    R = interferometry_response(
        _make_obs(np.zeros((1, nrow, 1)), uvw), grid, DUCC
    )

    full = (1, 1, 1) + RECT_SHAPE
    s = rng.normal(size=full)
    v = rng.normal(size=(1, nrow, 1)) + 1j * rng.normal(size=(1, nrow, 1))

    Rs = np.asarray(R(jnp.array(s)))
    RT = jax.linear_transpose(R, jnp.ones(full))
    RTv = np.asarray(RT(jnp.array(v))[0])

    assert np.isrealobj(RTv) or np.allclose(RTv.imag, 0.0)
    lhs = float(np.real(np.sum(Rs * v)))            # bilinear pairing, no conj
    rhs = float(np.sum(s * np.real(RTv)))
    np.testing.assert_allclose(lhs, rhs, rtol=1e-6, atol=1e-12)

    # And the naive conjugated pairing must NOT coincide (documents the choice).
    # Compare relatively: the inner products are ~1e-9 in magnitude (R carries a
    # d_ra*d_dec ~ 1e-11 volume factor), so np.isclose's default atol would
    # swamp the check — require a real relative gap instead.
    lhs_conj = float(np.real(np.sum(Rs * np.conj(v))))
    assert abs(lhs_conj - rhs) > 0.1 * abs(rhs), (
        "conjugated pairing unexpectedly matched — the adapter conj analysis "
        f"in this test's docstring would be wrong (lhs_conj={lhs_conj}, rhs={rhs})"
    )


# ==========================================================================
# (d) dirty-image orientation on a RECTANGLE (golden-free, glyph roundtrip)
# ==========================================================================
def _pad_square(a):
    """Embed a rectangular array in a centered square canvas.

    ``dihedral_verdict`` correlates against all of D4, and transpose/rot90
    change a rectangle's shape (they only close on a square).  Padding both the
    image and the truth to a common square makes every D4 element well-defined
    while leaving the shift-invariant correlation — and hence the winning
    verdict — unchanged.
    """
    n = max(a.shape)
    out = np.zeros((n, n))
    i0 = (n - a.shape[0]) // 2
    j0 = (n - a.shape[1]) // 2
    out[i0 : i0 + a.shape[0], j0 : j0 + a.shape[1]] = a
    return out


def _rect_glyph(grid, scale=1.5):
    d_dec, d_ra = grid.spatial.distances.to(u.arcsec).value
    n_dec, n_ra = RECT_SHAPE
    return rasterize_canonical(
        RECT_SHAPE, (n_dec // 2, n_ra // 2), (d_dec, d_ra), scale=scale
    )


def _analytic_glyph_vis(grid, glyph, uvw):
    """Physical (external-convention) visibilities of the glyph.

    V = d_ra*d_dec*exp(+2πi(u*l + v*m)) — the standard radio / CASA
    convention (the same external anchor p7 uses).  Since Batch E the forward
    response EMITS this convention directly (the adapter is a pure transpose,
    no conjugation), so these analytic visibilities are exactly what the
    response would produce for the glyph; ``dirty_image`` (the Hermitian
    adjoint) then closes the loop to an *identity* orientation.  See the
    module docstring and test_dirty_of_response_forward_is_identity below.
    """
    n_dec, n_ra = RECT_SHAPE
    c0, c1 = n_dec // 2, n_ra // 2
    d_dec, d_ra = grid.spatial.distances.to(u.rad).value
    uu, vv = uvw[:, 0], uvw[:, 1]
    V = np.zeros(uvw.shape[0], np.complex128)
    for i, j in np.argwhere(glyph > 0):
        di, dj = i - c0, j - c1
        l, m = -dj * d_ra, +di * d_dec
        V += glyph[i, j] * d_ra * d_dec * np.exp(+2j * np.pi * (uu * l + vv * m))
    return V


def _disk_uvw(seed=7, nrow=3000, umax=8.0e4):
    rng = np.random.default_rng(seed)
    r = umax * np.sqrt(rng.uniform(size=nrow))
    th = rng.uniform(0.0, 2 * np.pi, nrow)
    uvw = np.zeros((nrow, 3))
    uvw[:, 0] = r * np.cos(th)
    uvw[:, 1] = r * np.sin(th)
    return uvw


def test_dirty_image_rectangle_orientation():
    """Physical vis -> dirty_image -> canonical identity, on a RECTANGLE.

    The honest rectangle analogue of p7 (which is square-only): feed dirty_image
    *physical* visibilities of the orientation glyph (the CASA/external sign
    convention) on an anisotropic rectangular grid and require the dihedral
    verdict against the canonical glyph to be ``identity``.  Golden-free.
    """
    grid = _rect_grid()
    glyph = _rect_glyph(grid)
    uvw = _disk_uvw()
    V = _analytic_glyph_vis(grid, glyph, uvw).reshape(1, uvw.shape[0], 1)

    dirty = dirty_image(_make_obs(V, uvw), grid, DUCC, weighting="natural")
    img = np.real(np.asarray(dirty.value))[0, 0, 0]

    verdict, scores = dihedral_verdict(_pad_square(img), _pad_square(glyph))
    runner_up = max(v for k, v in scores.items() if k != verdict)
    assert verdict == "identity", (
        f"rectangle dirty image verdict {verdict!r} (not identity); "
        f"scores={ {k: round(v, 3) for k, v in scores.items()} }"
    )
    assert scores[verdict] - runner_up > 0.05, f"weak margin: {scores}"


def test_dirty_of_response_forward_is_identity():
    """Forward through R, then dirty -> identity (the Batch-E fix, witnessed).

    This test was the honest witness of the Batch-A defect.  Back then the
    adapter carried a spurious ``jnp.conj`` and ``dirty_image`` used the plain
    ``jax.linear_transpose`` (the bilinear transpose R^T), so routing the
    response's own output back through the dirty image formed R^T R — whose
    autocorrelation peak is point-reflected, giving a deterministic ``rot180``.
    It was an xfail recording that finding.

    Batch E (2026-07-07) removed the conj (R is now C-linear) and made
    ``dirty_image`` HERMITIAN (it conjugates the visibility cotangent, forming
    R^H).  Forward-then-dirty is now the Hermitian normal equations R^H R,
    whose autocorrelation peak is at the origin -> the orientation is
    ``identity``.  The xfail is gone; this now passes as a regular test.
    """
    grid = _rect_grid()
    glyph = _rect_glyph(grid)
    uvw = _disk_uvw()
    nrow = uvw.shape[0]
    R = interferometry_response(
        _make_obs(np.zeros((1, nrow, 1)), uvw), grid, DUCC
    )
    vis = np.asarray(R(glyph.reshape((1, 1, 1) + RECT_SHAPE))).astype(np.complex128)

    dirty = dirty_image(_make_obs(vis, uvw), grid, DUCC, weighting="natural")
    img = np.real(np.asarray(dirty.value))[0, 0, 0]

    verdict, _ = dihedral_verdict(_pad_square(img), _pad_square(glyph))
    assert verdict == "identity", verdict
