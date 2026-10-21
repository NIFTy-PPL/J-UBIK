# Copyright(C) 2026
# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause
# Authors: Matteo Guardiani

"""HEALPix synthesis conventions and public CPU interface."""

import jax
import numpy as np
import pytest
from nifty.re.correlated_field import get_sht, make_grid
from numpy.testing import assert_allclose

import jubik as ju

pmp = pytest.mark.parametrize


@pytest.mark.parametrize("nside, lmax, mmax", [(2, 4, 4), (4, 8, 3)])
def test_public_healpix_synthesis_and_derivatives(nside, lmax, mmax):
    with jax.enable_x64(True):
        sht = ju.get_healpix_sht(nside, lmax, mmax)
        reference = get_sht(nside, 0, lmax, mmax, 1)
        size = (lmax + 1) ** 2 - (lmax - mmax) * (lmax - mmax + 1)
        monopole = np.zeros(size)
        monopole[0] = 1.0
        assert_allclose(sht(monopole), np.ones(12 * nside**2), atol=1e-12)

        rng = np.random.default_rng(nside)
        x, tangent = rng.normal(size=(2, size))
        cotangent = rng.normal(size=12 * nside**2)
        assert_allclose(jax.jit(sht)(x), reference(x), rtol=1e-12, atol=1e-12)
        assert_allclose(
            jax.jvp(sht, (x,), (tangent,))[1],
            reference(tangent),
            rtol=1e-12,
            atol=1e-12,
        )
        assert_allclose(
            jax.vjp(sht, x)[1](cotangent)[0],
            jax.vjp(reference, x)[1](cotangent)[0],
            rtol=1e-12,
            atol=1e-12,
        )
        assert_allclose(
            jax.jit(jax.vmap(sht))(np.stack((x, tangent))),
            jax.vmap(reference)(np.stack((x, tangent))),
            rtol=1e-12,
            atol=1e-12,
        )


def test_healpix_sht_rejects_unknown_backend():
    with pytest.raises(ValueError, match="backend"):
        ju.get_healpix_sht(2, 4, 4, backend="cuda")


@pytest.mark.parametrize(
    "nside, lmax, mmax", [(0, 4, 4), (2, -1, 0), (2, 4, -1), (2, 4, 5)]
)
def test_healpix_sht_rejects_invalid_geometry(nside, lmax, mmax):
    with pytest.raises(ValueError, match="nside.*mmax.*lmax"):
        ju.get_healpix_sht(nside, lmax, mmax)


@pmp("nside", [4, 16])
def test_real_alm_packing_with_ducc0_core_matches_nifty(nside):
    """The cunusht backend's packing around a complex synthesis/adjoint pair,
    checked on CPU with ducc0 as that pair."""
    import ducc0
    from jubik.sht import _alm_from_real, _real_from_alm

    with jax.enable_x64(True):
        hg = make_grid((nside,), (1.0,), "spherical").harmonic_grid
        L = hg.lmax + 1
        info = ducc0.healpix.Healpix_Base(nside, "RING").sht_info()
        ducc = get_sht(nside, 0, hg.lmax, hg.mmax, 1)
        rng = np.random.default_rng(nside)
        x = rng.normal(size=hg.shape)
        cot = rng.normal(size=12 * nside**2)

        alm = np.empty((hg.shape[0] + L) // 2, dtype=np.complex128)
        _alm_from_real(x, alm, L)
        m = ducc0.sht.synthesis(alm=alm[None], lmax=hg.lmax, spin=0, **info)[0]
        assert_allclose(np.sqrt(4 * np.pi) * m, ducc(x), rtol=1e-12, atol=1e-12)

        adj = ducc0.sht.adjoint_synthesis(
            map=cot[None], lmax=hg.lmax, spin=0, **info
        )[0]
        xt = np.empty(hg.shape)
        _real_from_alm(np.sqrt(4 * np.pi) * adj, xt, L)
        assert_allclose(xt, jax.vjp(ducc, x)[1](cot)[0], rtol=1e-12, atol=1e-12)


def test_cunusht_rejects_mmax_and_missing_gpu():
    from jubik.sht import _require_float64

    with pytest.raises(ValueError, match="mmax"):
        ju.get_healpix_sht(16, 32, 31, backend="cunusht")
    with pytest.raises(ValueError, match="lmax >= 32"):
        ju.get_healpix_sht(16, 16, 16, backend="cunusht")
    with pytest.raises(TypeError, match="float64"):
        _require_float64(np.zeros(3, dtype=np.float32))
    try:
        jax.devices("gpu")
    except RuntimeError:
        with pytest.raises(RuntimeError, match="CUDA"):
            ju.get_healpix_sht(16, 32, 32, backend="cunusht")
    else:
        pytest.skip("a GPU is present; test/test_cunusht_sht.py covers it")
