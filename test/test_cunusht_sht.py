# Copyright(C) 2026
# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause
# Authors: Matteo Guardiani

"""cunuSHT HEALPix backend against ducc0. GPU only.

Skips unless cupy, cunuSHT, jaxbind-gpu and a CUDA device are present; see
'HEALPix SHT on GPU' in the README for the install. The ducc0 references run
on the CPU device, so JAX must see both platforms (leave JAX_PLATFORMS unset).
Thresholds are the acceptance criteria of the backend: do not loosen them to
make a failure pass.
"""

import gc
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# JAX's GPU backend first: it loads the CUDA libraries cupy and cuNuSHT reuse.
try:
    GPU = jax.devices("gpu")[0]
    CPU = jax.devices("cpu")[0]
except RuntimeError:
    pytest.skip("needs a CUDA device and JAX's CPU platform", allow_module_level=True)
pytest.importorskip("cupy")
pytest.importorskip("jaxbind_gpu")
pytest.importorskip("cunusht")

from nifty.re.correlated_field import get_sht, make_grid

import jubik as ju

jax.config.update("jax_enable_x64", True)
pmp = pytest.mark.parametrize

SMALL_TOL, LARGE_TOL, ADJOINT_TOL = 1e-10, 1e-9, 1e-10


def assert_close(actual, desired, tol):
    """Scaled L2 and maximum-norm errors, both relative to `desired`."""
    a, d = np.ravel(np.asarray(actual)), np.ravel(np.asarray(desired))
    l2 = np.linalg.norm(a - d) / np.linalg.norm(d)
    mx = np.max(np.abs(a - d)) / np.max(np.abs(d))
    assert l2 <= tol and mx <= tol, f"L2 {l2:.2e}, max {mx:.2e} > {tol:.0e}"


def on_cpu(f, *args):
    """Run a ducc0 reference on the CPU: nifty's jaxbind ducc0 callback is
    registered for the CPU only. Returns numpy arrays."""
    with jax.default_device(CPU):
        return jax.tree_util.tree_map(np.asarray, f(*jax.device_put(args, CPU)))


def assert_adjoint(f, x, y, tol=ADJOINT_TOL):
    """<f x, y> == <x, f^T y>, normalized by ||f x|| ||y||."""
    fx, vjp = jax.vjp(f, x)
    lhs = np.vdot(np.asarray(fx), y)
    rhs = np.vdot(x, np.asarray(vjp(y)[0]))
    defect = abs(lhs - rhs) / (np.linalg.norm(fx) * np.linalg.norm(y))
    assert defect <= tol, f"adjoint defect {defect:.2e} > {tol:.0e}"


def transforms(nside):
    hg = make_grid((nside,), (1.0,), "spherical").harmonic_grid
    ducc = get_sht(nside, 0, hg.lmax, hg.mmax, 0)
    cu = ju.get_healpix_sht(nside, hg.lmax, hg.mmax, backend="cunusht")
    return hg, ducc, jax.jit(cu)


def packed_index(l, m, lmax):
    """Real part of a_lm in nifty's packed vector; the imaginary part is +1."""
    L = lmax + 1
    return l if m == 0 else L + 2 * ((m - 1) * L - (m - 1) * m // 2 + l - m)


def inputs(hg, rng):
    yield "random", rng.normal(size=hg.shape)
    lmax = hg.lmax
    for name, i in [
        ("monopole", 0),
        ("l2m0", packed_index(2, 0, lmax)),
        ("l3m1_re", packed_index(3, 1, lmax)),
        ("l3m1_im", packed_index(3, 1, lmax) + 1),
        ("lmax_re", packed_index(lmax, lmax, lmax)),
        ("lmax_im", packed_index(lmax, lmax, lmax) + 1),
    ]:
        x = np.zeros(hg.shape)
        x[i] = 1.0
        yield name, x


@pmp("nside", [16, 32])
def test_synthesis_jvp_vjp_batch_match_ducc0(nside):
    hg, ducc, cu = transforms(nside)
    rng = np.random.default_rng(nside)
    npix = 12 * nside**2

    for name, x in inputs(hg, rng):
        assert_close(cu(x), on_cpu(ducc, x), SMALL_TOL)

    x, tx, z = rng.normal(size=(3,) + hg.shape)
    cot = rng.normal(size=npix)
    xs = rng.normal(size=(3,) + hg.shape)
    cots = rng.normal(size=(3, npix))

    def jvp(f, x, tx):
        return jax.jvp(f, (x,), (tx,))[1]

    def vjp(f, x, cot):
        return jax.vjp(f, x)[1](cot)[0]

    def batch(f, xs):
        return jax.vmap(f)(xs)

    def batched_vjp(f, x, cots):
        return jax.vmap(jax.vjp(f, x)[1])(cots)[0]

    def vjp_of_transpose(f, x, cot, z):
        # Differentiating the transpose gives the forward transform back.
        return vjp(lambda y: vjp(f, x, y), cot, z)

    for check, args in [
        (jvp, (x, tx)),
        (vjp, (x, cot)),
        (batch, (xs,)),
        (batched_vjp, (x, cots)),
        (vjp_of_transpose, (x, cot, z)),
    ]:
        assert_close(
            check(cu, *args), on_cpu(lambda *a: check(ducc, *a), *args), SMALL_TOL
        )
    assert_adjoint(cu, x, cot)


def test_repeat_calls_keep_inputs_and_no_stale_buffers():
    hg, ducc, cu = transforms(16)
    rng = np.random.default_rng(1)
    xs = rng.normal(size=(3,) + hg.shape)
    cots = rng.normal(size=(3, 12 * 16**2))
    pull = jax.jit(lambda c: jax.vjp(cu, xs[0])[1](c)[0])
    for i in (0, 1, 2, 0, 2):
        x = jax.device_put(xs[i], GPU)
        c = jax.device_put(cots[i], GPU)
        assert_close(cu(x), on_cpu(ducc, xs[i]), SMALL_TOL)
        desired = on_cpu(lambda x, c: jax.vjp(ducc, x)[1](c)[0], xs[0], cots[i])
        assert_close(pull(c), desired, SMALL_TOL)
        np.testing.assert_array_equal(np.asarray(x), xs[i])
        np.testing.assert_array_equal(np.asarray(c), cots[i])


def test_stays_on_gpu():
    hg, _, cu = transforms(16)
    rng = np.random.default_rng(2)
    x = jax.device_put(rng.normal(size=hg.shape), GPU)
    c = jax.device_put(rng.normal(size=12 * 16**2), GPU)
    grad = jax.jit(lambda x, c: jax.grad(lambda x: jnp.vdot(cu(x), c))(x))
    cu(x).block_until_ready(), grad(x, c).block_until_ready()  # compile first
    with jax.transfer_guard("disallow"):
        y, g = cu(x), grad(x, c)
    assert y.devices() == {GPU} and g.devices() == {GPU}


def test_cold_start_in_fresh_process():
    script = """
import jax, numpy as np
jax.config.update("jax_enable_x64", True)
from nifty.re.correlated_field import get_sht
import jubik as ju
x = np.random.default_rng(0).normal(size=33**2)
y = np.asarray(ju.get_healpix_sht(16, 32, 32, backend="cunusht")(x))
cpu = jax.devices("cpu")[0]
with jax.default_device(cpu):  # the ducc0 reference is CPU only
    d = np.asarray(get_sht(16, 0, 32, 32, 1)(jax.device_put(x, cpu)))
print("REL_ERR", np.linalg.norm(y - d) / np.linalg.norm(d))
"""
    out = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )
    # SHTns' own (C stdio) output is flushed at exit, after Python's print, so
    # read the marked line rather than the last one.
    lines = [l for l in out.stdout.splitlines() if l.startswith("REL_ERR ")]
    assert len(lines) == 1, out.stdout + out.stderr
    assert float(lines[0].split()[1]) <= SMALL_TOL, out.stderr


def test_plan_reuse_and_release():
    import jaxbind_gpu

    hg, ducc, cu = transforms(16)
    x = np.random.default_rng(3).normal(size=hg.shape)
    cu(x).block_until_ready()
    before = jaxbind_gpu.runtime_stats()
    cu(2 * x).block_until_ready()
    after = jaxbind_gpu.runtime_stats()
    assert after["cache_misses"] == before["cache_misses"]
    assert after["cache_hits"] > before["cache_hits"]

    del cu
    gc.collect()
    jaxbind_gpu.synchronize(clear_cache=True)
    assert jaxbind_gpu.runtime_stats()["cached"] == 0
    _, _, cu = transforms(16)
    assert_close(cu(x), on_cpu(ducc, x), SMALL_TOL)


@pmp("nside", [128, 512, 1024])
def test_large_nside(nside):
    hg, ducc, cu = transforms(nside)
    rng = np.random.default_rng(nside)
    x = rng.normal(size=hg.shape)
    cot = rng.normal(size=12 * nside**2)
    assert_close(cu(x), on_cpu(ducc, x), LARGE_TOL)
    desired = on_cpu(lambda x, c: jax.vjp(ducc, x)[1](c)[0], x, cot)
    assert_close(jax.vjp(cu, x)[1](cot)[0], desired, LARGE_TOL)
    assert_adjoint(cu, x, cot)
