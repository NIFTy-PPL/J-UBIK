"""The Gaussian likelihood of one observation given predicted visibilities."""

import jax
import nifty.re as jft
import numpy as np
import pytest
from numpy.testing import assert_allclose

import jubik as ju
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
    d, w = obs.vis_val, obs.weight_val
    scaled = jft.Model(lambda s: s * d, domain=jft.ShapeWithDtype(()))  # bare-array input

    likelihood = build_likelihood(obs, scaled)

    assert_allclose(likelihood.energy(0.0), 0.5 * np.sum(w * np.abs(d) ** 2), rtol=1e-12)


def test_inverse_std_is_per_real_and_imaginary_part():
    obs = random_obs()
    d = obs.vis_val
    zero = jft.Model(lambda x: x["s"] * d, domain={"s": jft.ShapeWithDtype(())})
    inverse_std = jft.Model(lambda x: x["inv"] * np.ones(d.shape), domain={"inv": jft.ShapeWithDtype(())})

    likelihood = build_likelihood(obs, zero, inverse_std)

    energy = likelihood.energy(jft.Vector({"s": 0.0, "inv": 2.0}))
    assert_allclose(energy, 0.5 * 4 * np.sum(np.abs(d) ** 2) - 2 * d.size * np.log(2), rtol=1e-12)


def test_wrong_shape_raises():
    wrong = jft.Model(lambda x: x["v"], domain={"v": jft.ShapeWithDtype((1, 2, 3), complex)})
    with pytest.raises(ValueError, match="predicted visibilities"):
        build_likelihood(random_obs(), wrong)
