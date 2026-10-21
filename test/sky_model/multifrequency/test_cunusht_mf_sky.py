# Copyright(C) 2026
# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause
# Authors: Matteo Guardiani

"""Multi-frequency sky integration with the cunuSHT backend. GPU only.

Skips unless cupy, cunuSHT, jaxbind-gpu and a CUDA device are present; see
'HEALPix SHT on GPU' in the README for the install. The ducc0 references run
on the CPU device, so JAX must see both platforms (leave JAX_PLATFORMS unset).
Thresholds are the acceptance criteria of the backend: do not loosen them to
make a failure pass.
"""

import jax
import jax.numpy as jnp
import jax.random as random
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

import nifty.re as jft

import jubik as ju

jax.config.update("jax_enable_x64", True)
pmp = pytest.mark.parametrize

SMALL_TOL = 1e-10


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


def build_sky(sht_backend, deviations_settings, nside=16):
    amplitude = dict(
        fluctuations=(0.1, 0.01),
        loglogavgslope=(-4.0, 0.1),
        flexibility=None,
        asperity=None,
    )
    return ju.build_simple_spectral_sky(
        prefix="cunusht_test",
        shape=(nside,),
        distances=(1.0,),
        log_frequencies=np.array((0.1, 0.2, 0.6)),
        reference_frequency_index=0,
        zero_mode_settings=(0.0, 0.1),
        spatial_amplitude_settings=amplitude,
        spectral_index_settings=dict(mean=(0.0, 1.0), fluctuations=(0.1, 0.01)),
        spectral_amplitude_settings=amplitude,
        deviations_settings=deviations_settings,
        harmonic_type="spherical",
        sht_backend=sht_backend,
    )


def ravel(tree):
    return np.concatenate([np.ravel(v) for v in jax.tree_util.tree_leaves(tree)])


@pmp("deviations_settings", [None, dict(process="wiener", sigma=(1.0, 0.1))])
def test_mf_model_value_gradient_and_metric(deviations_settings):
    ref, cu = (build_sky(b, deviations_settings) for b in ("ducc0", "cunusht"))
    # Host inputs, so each side places them on its own device.
    p, v = (
        jax.tree_util.tree_map(np.asarray, ref.init(random.PRNGKey(k))) for k in (0, 1)
    )
    w = np.exp(np.random.default_rng(2).normal(size=ref.target.shape))

    def value_grad_metric(model, p, v, w):
        tangent = jax.jvp(model, (p,), (v,))[1]
        metric = jax.vjp(model, p)[1](w * tangent)[0]
        grad = jax.grad(lambda p: jnp.sum(w * model(p)))(p)
        return model(p), grad, metric

    actual = jax.jit(lambda *a: value_grad_metric(cu, *a))(p, v, w)
    desired = on_cpu(lambda *a: value_grad_metric(ref, *a), p, v, w)
    for a, d in zip(actual, desired):
        assert_close(ravel(a), ravel(d), SMALL_TOL)


def test_reconstruction_smoke():
    def reconstruct(sht_backend):
        sky = build_sky(sht_backend, None)
        truth = sky.init(random.PRNGKey(10))
        noise = 0.1 * random.normal(random.PRNGKey(11), sky.target.shape)
        lh = jft.Gaussian(sky(truth) + noise, lambda x: x / 0.1**2).amend(sky)

        def energy(p):
            return lh(p) + 0.5 * jft.vdot(p, p)

        x0 = 0.1 * jft.Vector(sky.init(random.PRNGKey(12)))
        x = jft.newton_cg(
            fun_and_grad=jax.jit(jax.value_and_grad(energy)),
            x0=x0,
            hessp=jax.jit(lambda p, t: lh.metric(p, t) + t),
            maxiter=3,
        )
        return energy(x0), energy(x)

    e0, e_ref = on_cpu(lambda: reconstruct("ducc0"))
    # Newton-CG amplifies 1e-12 transform differences to ~1e-8 in position,
    # so only the energies are compared.
    _, e_cu = reconstruct("cunusht")
    assert e_cu < e0
    np.testing.assert_allclose(e_cu, e_ref, rtol=1e-8)
