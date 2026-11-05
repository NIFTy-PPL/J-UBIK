"""_get_data_model_and_chi2: mean data model, σ and reduced chi² over samples,
for a fixed σ and for a model that returns its own σ."""

import jax.numpy as jnp
import nifty.re as jft
import numpy as np

from jubik.instruments.jwst.plotting.plotting_base import (
    _get_data_model_and_chi2,
    redchi2,
)

RNG = np.random.default_rng(0)
DATA = RNG.normal(size=(2, 5, 6))  # two exposures
MASK = RNG.random(DATA.shape) > 0.2
STD = 0.5 + RNG.random(DATA.shape)
W = RNG.normal(size=int(MASK.sum()))
DOMAIN = {"a": jft.ShapeWithDtype((), float), "k": jft.ShapeWithDtype((), float)}
SAMPLES = jft.Samples(
    pos=jft.Vector({"a": jnp.array(1.0)}),
    samples=jft.Vector({"a": jnp.array([0.3, -0.3])}),
)
SKIES = np.array([{"k": 0.1}, {"k": -0.2}])


def _prediction(x):
    return x["a"] * W + x["k"]


def _image(masked):
    image = np.zeros_like(DATA)
    image[MASK] = masked
    return image


def _expected(predictions, sigmas):
    """Mean model, mean σ and per-exposure redchi² mean and std, by hand."""
    models = [_image(p) for p in predictions]
    chis = [
        [
            redchi2(d[m], mo[m], s[m], d[m].size)
            for d, mo, m, s in zip(DATA, model, MASK, sigma)
        ]
        for model, sigma in zip(models, sigmas)
    ]
    return np.mean(models, 0), np.mean(sigmas, 0), jft.mean_and_std(chis)


def _inputs():
    return [{"a": 1.3, "k": 0.1}, {"a": 0.7, "k": -0.2}]  # pos ± sample, with sky


def test_fixed_std_over_samples():
    model = jft.Model(_prediction, domain=DOMAIN)
    predictions = [np.asarray(_prediction(x)) for x in _inputs()]
    mean, sigma, chi = _expected(predictions, [STD, STD])

    got_mean, got_std, got_chi = _get_data_model_and_chi2(
        SAMPLES, SKIES, model, DATA, MASK, STD
    )
    np.testing.assert_allclose(got_mean, mean, rtol=1e-12)
    assert got_std is STD
    np.testing.assert_allclose(got_chi, chi, rtol=1e-12)


def test_model_with_its_own_std_over_samples():
    model = jft.Model(
        lambda x: (_prediction(x), 1 / (1 + x["a"] ** 2 + 0 * W)), domain=DOMAIN
    )
    predictions = [np.asarray(_prediction(x)) for x in _inputs()]
    sigmas = [_image(np.full(W.size, 1 + x["a"] ** 2)) for x in _inputs()]
    mean, sigma, chi = _expected(predictions, sigmas)  # each sample its own σ

    got_mean, got_std, got_chi = _get_data_model_and_chi2(
        SAMPLES, SKIES, model, DATA, MASK, None
    )
    np.testing.assert_allclose(got_mean, mean, rtol=1e-12)
    np.testing.assert_allclose(got_std, sigma, rtol=1e-12)
    np.testing.assert_allclose(got_chi, chi, rtol=1e-12)


def test_single_position():
    model = jft.Model(_prediction, domain=DOMAIN)
    position = {"a": 1.3}
    got_mean, _, (chi, chi_std) = _get_data_model_and_chi2(
        position, {"k": 0.1}, model, DATA, MASK, STD
    )
    mean = _image(np.asarray(_prediction(_inputs()[0])))
    expected = [
        redchi2(d[m], mo[m], s[m], d[m].size)
        for d, mo, m, s in zip(DATA, mean, MASK, STD)
    ]
    np.testing.assert_allclose(got_mean, mean, rtol=1e-12)
    np.testing.assert_allclose(chi, expected, rtol=1e-12)
    np.testing.assert_array_equal(chi_std, 0)
