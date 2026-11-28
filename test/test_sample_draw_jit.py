"""`linear_minimizer_jit`: the parser's sample-draw kwargs, and NIFTy's linear
sample draw run with them."""

import logging

import jax
import jax.numpy as jnp
import nifty.re as jft
import numpy as np
import pytest

from jubik.instruments.jwst.rotation_and_shift.nufft_rotation_and_shift import (
    build_nufft_rotation_and_shift,
)
from jubik.minimization_parser import MinimizationParser

jax.config.update("jax_enable_x64", True)


def _config(jit=None, cg_iterations=10):
    config = {
        "samples": {
            "switches": [0],
            "n_samples": [2],
            "mode": ["linear_resample"],
            # a fixed number of CG iterations: no stop before maxiter
            "lin_miniter": [cg_iterations],
            "lin_maxiter": [cg_iterations],
            "lin_absdelta": [1e-300],
        },
        "kl_minimization": {"switches": [0], "kl_maxiter": [1], "kl_absdelta": [1.0]},
        "n_total_iterations": 1,
        "delta": {"switches": [0], "values": [1e-6]},
    }
    if jit is not None:
        config["linear_minimizer_jit"] = jit
    return config


def test_off_gives_todays_kwargs():
    for config in (_config(), _config(jit=False)):
        parser = MinimizationParser(config, n_dof=10, verbose=False)
        assert parser.linear_minimizer_jit is False
        assert parser.draw_linear_kwargs(0) == dict(
            cg_name=None,
            cg_kwargs=dict(
                name="lin sampling_cg",
                absdelta=1e-300,
                tol=None,
                atol=None,
                miniter=10,
                maxiter=10,
            ),
        )


def test_on_takes_static_cg_without_the_name():
    off = MinimizationParser(_config(), n_dof=10, verbose=False).draw_linear_kwargs(0)
    parser = MinimizationParser(_config(jit=True), n_dof=10, verbose=False)
    on = parser.draw_linear_kwargs(0)
    assert parser.linear_minimizer_jit is True
    assert on["cg"] is jft.conjugate_gradient.static_cg
    assert on["cg_name"] is None
    without_name = {k: v for k, v in off["cg_kwargs"].items() if k != "name"}
    assert on["cg_kwargs"] == without_name


@pytest.mark.parametrize("value", ["yes", 1, 0.0])
def test_non_boolean_is_refused(value):
    with pytest.raises(ValueError, match="linear_minimizer_jit"):
        MinimizationParser(_config(jit=value), n_dof=10, verbose=False)


def _likelihood():
    """A Gaussian likelihood of a sky seen through the JWST NUFFT, with its
    capacity cond: rotated subsample grids, partly outside the sky."""
    pytest.importorskip("jax_finufft")
    sky_shape, out_shape = (24, 20), (13, 17)
    rng = np.random.default_rng(3)
    y, x = np.mgrid[: out_shape[0], : out_shape[1]] * 0.7 + 1.1
    centers = []
    for i in range(3):
        theta = rng.uniform(-0.3, 0.3)
        c, s = np.cos(theta), np.sin(theta)
        centers.append(np.stack([c * y - s * x, s * y + c * x]) + 4.0 * i)
    centers = np.stack(centers)
    upper = np.array(sky_shape)[:, None, None]
    inside = int(np.all((centers >= 0) & (centers <= upper), axis=-3).sum())
    assert 0 < inside < centers[:, 0].size  # the gather and the mask both work
    rs = build_nufft_rotation_and_shift(sky_shape, out_shape, capacity=inside + 20)
    domain = {"sky": jft.ShapeWithDtype(sky_shape, float)}
    model = jft.Model(lambda p: rs(jnp.exp(0.3 * p["sky"]), centers), domain=domain)
    truth = jft.random_like(jax.random.PRNGKey(1), model.domain)
    noise = 0.05 * jax.random.normal(jax.random.PRNGKey(2), model.target.shape)
    data = model(truth) + noise
    return jft.Gaussian(data, noise_cov_inv=lambda r: r / 0.05**2).amend(model)


def _draw(likelihood, parser, key, position):
    """Linear samples as optimize_kl draws them with this parser's settings."""
    vi = jft.OptimizeVI(likelihood, 1, linear_minimizer_jit=parser.linear_minimizer_jit)
    keys = jax.random.split(key, 2)
    kwargs = parser.draw_linear_kwargs(0)
    samples, _ = vi.draw_linear_samples(position, keys, **kwargs)
    return samples


def _flat(tree):
    return np.concatenate([np.ravel(v) for v in jax.tree.leaves(tree)])


def test_on_draws_todays_samples():
    likelihood = _likelihood()
    position = jft.Vector(jft.random_like(jax.random.PRNGKey(4), likelihood.domain))
    key = jax.random.PRNGKey(5)
    off = MinimizationParser(_config(), n_dof=10, verbose=False)
    on = MinimizationParser(_config(jit=True), n_dof=10, verbose=False)
    today = _flat(_draw(likelihood, off, key, position))
    jitted = _flat(_draw(likelihood, on, key, position))
    # 2e-9 of the largest value: the agreement measured at 10 CG iterations on a
    # V100 (GPU atomics); on the CPU it is close to rounding
    assert np.max(np.abs(jitted - today)) <= 2e-9 * np.max(np.abs(today))


def test_on_compiles_the_draw_once(caplog):
    # the NUFFT's capacity cond runs inside the one jitted draw: a second draw
    # (new key, new position, as in each KL iteration) compiles nothing
    likelihood = _likelihood()
    parser = MinimizationParser(_config(jit=True), n_dof=10, verbose=False)
    vi = jft.OptimizeVI(likelihood, 1, linear_minimizer_jit=True)

    def draw(seed):
        latent = jft.random_like(jax.random.PRNGKey(seed), likelihood.domain)
        keys = jax.random.split(jax.random.PRNGKey(seed + 100), 2)
        kwargs = parser.draw_linear_kwargs(0)  # anew each time, as optimize_kl
        samples = vi.draw_linear_samples(jft.Vector(latent), keys, **kwargs)
        return jax.block_until_ready(samples)

    draw(0)
    # the jax logger's own level, so a JAX_LOGGING_LEVEL above WARNING cannot
    # silence the compile records
    with caplog.at_level(logging.WARNING, logger="jax"), jax.log_compiles(True):
        draw(1)
    compiles = [r.getMessage() for r in caplog.records if "Compiling" in r.getMessage()]
    assert compiles == []
