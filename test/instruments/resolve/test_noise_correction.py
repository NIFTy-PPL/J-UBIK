import jax
import numpy as np
import pytest
from numpy.testing import assert_allclose

import jubik as ju
import nifty.re as jft
from jubik.instruments.resolve.noise.factory_noise_correction import (
    factory_noise_correction_model,
)
from jubik.instruments.resolve.parse.noise import factory_noise_correction_parser

from generate_test_obs import generate_random_obs

jax.config.update("jax_enable_x64", True)

# Rows (0,1), (0,2), (1,2), (0,1); antenna 3 has no data.
ANT1 = np.array([0, 0, 1, 0])
ANT2 = np.array([1, 2, 2, 1])
CORRECTION = {"alpha": 2.0, "scale": 1.0}


def observation():
    return generate_random_obs(
        np.array([1e9, 1.1e9]),
        4,
        [-1e2, 1e2],
        [0, 0],
        ju.polarization.PolarizationType(("I",)),
        ant1=ANT1,
        ant2=ANT2,
        times=np.arange(4.0),
        rng=np.random.default_rng(1),
    )


def build(weight_correction):
    obs = observation()
    settings = factory_noise_correction_parser({"weight_correction": weight_correction})
    model = factory_noise_correction_model(settings, obs, prefix="f_noise_std")
    x = jft.random_like(jax.random.PRNGKey(3), model.domain)
    return obs, model, x


def test_antenna_correction_shares_factors_between_rows():
    obs, model, x = build({"antenna": CORRECTION})
    assert set(model.domain) == {"f_noise_std_antenna_sigma"}
    assert model.domain["f_noise_std_antenna_sigma"].shape == (3,)
    assert model.correction_model is None

    s = np.asarray(model.antenna_correction_model(x))
    factor = np.sqrt(obs.weight.asnumpy()) / np.asarray(model(x))
    # Rows (0,1) and (0,2) share antenna 0.
    assert_allclose(factor[:, 0] / factor[:, 1], s[1] / s[2])
    assert_allclose(factor[:, 2], s[1] * s[2])
    assert_allclose(factor[:, 0], factor[:, 3])


def test_both_corrections_multiply():
    obs, model, x = build({"baseline": CORRECTION, "antenna": CORRECTION})
    assert set(model.domain) == {"f_noise_std_sigma", "f_noise_std_antenna_sigma"}
    assert model.domain["f_noise_std_sigma"].shape == (3,)

    sb = np.asarray(model.correction_model(x))
    s = np.asarray(model.antenna_correction_model(x))
    baseline_of_row = np.array([0, 1, 2, 0])
    expected = (
        np.sqrt(obs.weight.asnumpy())
        / (sb[baseline_of_row] * s[ANT1] * s[ANT2])[None, :, None]
    )
    assert_allclose(np.asarray(model(x)), expected)


def test_baseline_only_keeps_latent_name():
    _, model, _ = build({"baseline": CORRECTION})
    assert set(model.domain) == {"f_noise_std_sigma"}


@pytest.mark.parametrize("block", [{"station": CORRECTION}, {}])
def test_unknown_or_empty_block_raises(block):
    with pytest.raises(ValueError, match="weight_correction"):
        factory_noise_correction_parser({"weight_correction": block})
