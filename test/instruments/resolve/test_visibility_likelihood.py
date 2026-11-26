"""The Gaussian likelihood of one observation given predicted visibilities."""

import jax
import nifty.re as jft
import numpy as np
import pytest
from numpy.testing import assert_allclose

import jubik as ju
import jubik.instruments.resolve as rve
from jubik.instruments.resolve.likelihood import build_likelihood

from generate_test_obs import generate_random_obs

jax.config.update("jax_enable_x64", True)


def random_obs():
    pol = ju.polarization.PolarizationType(("LL", "RR"))
    return generate_random_obs(
        np.array([1e11, 1.1e11]), 30, [-1e3, 1e3], [-5, 5], pol,
        rng=np.random.default_rng(5),
    )


def test_weights_are_the_inverse_noise_covariance():
    obs = random_obs()
    d, w = obs.vis, obs.weight
    scaled = jft.Model(lambda s: s * d, domain=jft.ShapeWithDtype(()))  # bare-array input

    likelihood = build_likelihood(obs, scaled)

    assert_allclose(likelihood.energy(0.0), 0.5 * np.sum(w * np.abs(d) ** 2), rtol=1e-12)


def test_inverse_std_is_per_real_and_imaginary_part():
    obs = random_obs()
    d = obs.vis
    zero = jft.Model(lambda x: x["s"] * d, domain={"s": jft.ShapeWithDtype(())})
    inverse_std = jft.Model(lambda x: x["inv"] * np.ones(d.shape), domain={"inv": jft.ShapeWithDtype(())})

    likelihood = build_likelihood(obs, zero, inverse_std)

    energy = likelihood.energy(jft.Vector({"s": 0.0, "inv": 2.0}))
    assert_allclose(energy, 0.5 * 4 * np.sum(np.abs(d) ** 2) - 2 * d.size * np.log(2), rtol=1e-12)


def test_flagged_points_are_left_out():
    obs = random_obs()
    d, w = obs.vis.copy(), obs.weight.copy()
    d[0, 0, 0], w[0, 0, 0] = np.nan, 0.0  # a flagged point may hold NaN
    obs = rve.Observation(obs.antenna_positions, d, w, obs.legacy_polarization, obs.freq)
    good = w > 0
    vis = jft.Model(lambda x: x["s"] * np.nan_to_num(d), domain={"s": jft.ShapeWithDtype(())})
    sqrt_weight = jft.Model(lambda x: x["inv"] * np.sqrt(w), domain={"inv": jft.ShapeWithDtype(())})

    fixed = build_likelihood(obs, vis).energy(jft.Vector({"s": 0.0}))
    inferred = build_likelihood(obs, vis, sqrt_weight).energy(jft.Vector({"s": 0.0, "inv": 1.0}))

    expected = 0.5 * np.sum(w[good] * np.abs(d[good]) ** 2)
    assert_allclose(fixed, expected, rtol=1e-12)
    assert_allclose(inferred, expected - np.sum(np.log(w[good])), rtol=1e-12)


def test_likelihood_keeps_the_model_initializers():
    obs = random_obs()
    one = lambda key: np.ones(())
    # a callable and a structured initializer
    vis = jft.Model(lambda x: x["s"] * obs.vis, init=lambda key: {"s": np.ones(())})
    ones = np.ones(obs.vis.shape)
    inverse_std = jft.Model(lambda x: x["inv"] * ones, init=jft.Initializer({"inv": one}))

    for likelihood in (build_likelihood(obs, vis), build_likelihood(obs, vis, inverse_std)):
        start = likelihood.init(jax.random.PRNGKey(0))
        assert all(v == 1.0 for v in jax.tree.leaves(start))


def test_wrong_shape_raises():
    obs = random_obs()
    wrong = jft.Model(lambda x: x["v"], domain={"v": jft.ShapeWithDtype((1, 2, 3), complex)})
    with pytest.raises(ValueError, match="predicted visibilities"):
        build_likelihood(obs, wrong)

    vis = jft.Model(lambda x: x["v"], domain={"v": jft.ShapeWithDtype(obs.vis.shape, complex)})
    scalar = jft.Model(lambda x: x["inv"], domain={"inv": jft.ShapeWithDtype(())})
    with pytest.raises(ValueError, match="inverse_std"):
        build_likelihood(obs, vis, scalar)
