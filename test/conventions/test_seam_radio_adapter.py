"""Seams: the two named conversions of the resolve radio path.

Design page rows pinned here (docs/source/user/canonical-sky-design.md):

- Resolve forward: "canonical_sky_to_visibilities transposes to the wgridder
  (l, m) layout. Pure transpose, no conjugation."
- Resolve dirty image: "dirty_image conjugates the visibility cotangent so
  the result is the Hermitian adjoint, not the bilinear transpose."

The raw gridders (ducc, finufft) read their array as ``(l, m)``, RA first.
That is their layout, not ours; the adapter is where the two meet.  The
controls here make sure the adapter is not a no-op and carries no hidden
conjugation, which is the defect that once made a likelihood fit converge to
the rotated sky.
"""

import importlib.util

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from radio_anchor import (
    DUCC,
    NPIX,
    OFFSETS,
    canonical_point_sky,
    disk_uvw,
    make_obs,
    predicted_vis,
    rect_grid,
    RECT_SHAPE,
    stub_backend,
)

from jubik.instruments.resolve.constants import RESOLVE_SKY_UNIT
from jubik.instruments.resolve.dirty_image import dirty_image
from jubik.instruments.resolve.response import (
    canonical_sky_to_visibilities,
    interferometry_response,
)

finufft_required = pytest.mark.skipif(
    importlib.util.find_spec("jax_finufft") is None, reason="jax_finufft not installed"
)
BACKENDS = ["ducc", pytest.param("finufft", marks=finufft_required)]
D_RA, D_DEC = 1.0e-5, 1.5e-5


@pytest.mark.parametrize("backend", BACKENDS)
def test_adapter_is_exactly_a_transpose(backend):
    """adapter(sky) == raw(sky.T) for an arbitrary sky: no flip, no conj, no scale."""
    apply = stub_backend(backend, D_RA, D_DEC)
    sky = np.random.default_rng(11).normal(size=(NPIX, NPIX)) ** 2
    via_adapter = canonical_sky_to_visibilities(apply, sky)
    np.testing.assert_allclose(via_adapter, apply(sky.T), rtol=1e-12, atol=0)


@pytest.mark.parametrize("backend", BACKENDS)
def test_raw_gridder_is_ra_first(backend):
    """Control: the raw gridder alone violates the canonical contract.

    Its dim 0 is l (RA).  If this ever passes, the gridder changed layout and
    the adapter's transpose became wrong.
    """
    apply = stub_backend(backend, D_RA, D_DEC)
    raw = apply(canonical_point_sky(6, 4))
    assert not np.allclose(raw, predicted_vis(6, 4, D_RA, D_DEC), rtol=1e-4, atol=1e-13)


def test_conjugated_adapter_violates_contract():
    """Control: conj(adapter(sky)) does not satisfy the contract either.

    Guards the sign of the exponent: a spurious conjugation cannot re-enter
    and pass.
    """
    apply = stub_backend("ducc", D_RA, D_DEC)
    vis = canonical_sky_to_visibilities(apply, canonical_point_sky(6, 4))
    assert not np.allclose(
        np.conj(vis), predicted_vis(6, 4, D_RA, D_DEC), rtol=1e-4, atol=1e-13
    )


def test_response_linear_transpose_is_bilinear():
    """jax.linear_transpose of the response is the BILINEAR transpose.

    Since the adapter is a pure transpose the response ``R`` is C-linear.
    Its input domain is real, so ``R^T(v)`` is real and the identity that
    holds is ``Re(sum(R(s) v)) == sum(s R^T(v))``.  The conjugated pairing
    must not coincide.  ``dirty_image`` builds on this: it has to conjugate
    the visibility cotangent itself to get the Hermitian adjoint.
    """
    grid = rect_grid()
    rng = np.random.default_rng(3)
    nrow = 40
    uvw = rng.normal(size=(nrow, 3)) * 3000.0
    uvw[:, 2] = 0.0
    R = interferometry_response(make_obs(np.zeros((1, nrow, 1)), uvw), grid, DUCC)

    full = (1, 1, 1) + RECT_SHAPE
    s = rng.normal(size=full)
    v = rng.normal(size=(1, nrow, 1)) + 1j * rng.normal(size=(1, nrow, 1))

    Rs = np.asarray(R(jnp.array(s)))
    RT = jax.linear_transpose(R, jnp.ones(full))
    RTv = np.asarray(RT(jnp.array(v))[0])

    assert np.isrealobj(RTv) or np.allclose(RTv.imag, 0.0)
    lhs = float(np.real(np.sum(Rs * v)))
    rhs = float(np.sum(s * np.real(RTv)))
    np.testing.assert_allclose(lhs, rhs, rtol=1e-6, atol=1e-12)

    # The inner products are ~1e-9 (R carries a d_ra*d_dec ~ 1e-11 volume
    # factor), so compare relatively rather than with np.isclose's atol.
    lhs_conj = float(np.real(np.sum(Rs * np.conj(v))))
    assert abs(lhs_conj - rhs) > 0.1 * abs(rhs)


def test_dirty_image_is_the_hermitian_adjoint():
    """<R s, p>_Hermitian == vol^2 * <s, dirty(p)> for the weighted data p.

    ``dirty_image`` returns ``R^H(d w / sum w) / vol^2``.  With ``R`` C-linear
    and the sky real, ``Re(sum(conj(R s) * p))`` is the Hermitian pairing.
    The bilinear pairing ``Re(sum(R s * p))`` must NOT match: that is the
    defect where forward-then-dirty came out rot180.
    """
    grid = rect_grid()
    rng = np.random.default_rng(5)
    uvw = disk_uvw(seed=5, nrow=400)
    nrow = uvw.shape[0]
    d = rng.normal(size=(1, nrow, 1)) + 1j * rng.normal(size=(1, nrow, 1))
    obs = make_obs(d, uvw)  # unit weights, so p = d / nrow

    R = interferometry_response(obs, grid, DUCC)
    full = (1, 1, 1) + RECT_SHAPE
    s = rng.normal(size=full)
    Rs = np.asarray(R(jnp.array(s)))

    dirty = dirty_image(obs, grid, DUCC, weighting="natural")
    img = np.asarray(dirty.to(RESOLVE_SKY_UNIT).value)
    vol = grid.spatial.dvol.value
    p = d / nrow

    hermitian = float(np.real(np.sum(np.conj(Rs) * p)))
    rhs = float(np.sum(s * img)) * vol**2
    np.testing.assert_allclose(hermitian, rhs, rtol=1e-6, atol=1e-15)

    bilinear = float(np.real(np.sum(Rs * p)))
    assert abs(bilinear - rhs) > 0.1 * abs(rhs), "dirty_image is not Hermitian"
