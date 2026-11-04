"""ShotNoiseGaussianLikelihoodBuilder: σ² = a + b · max(R + sky, 0), with R the
response output."""

import jax
import jax.numpy as jnp
import nifty.re as jft
import numpy as np
import pytest

from jubik.instruments.jwst.likelihood.likelihood import (
    ShotNoise,
    ShotNoiseGaussianLikelihoodBuilder,
)

SHAPE = (1, 3, 4)
MASK = np.ones(SHAPE, bool)
MASK[0, 0, 0] = False
A, B = 4.0, 0.5
SKY = np.linspace(0.0, 2.0, 12).reshape(SHAPE)
SIGNAL = np.linspace(-50.0, 50.0, 12).reshape(SHAPE)  # negative and positive
RESPONSE = jft.Model(
    lambda x: x["s"][MASK], domain={"s": jft.ShapeWithDtype(SHAPE, float)}
)


def _builder(data=0.0, **noise):
    noise = dict(a=A, b=B, sky=SKY) | noise
    return ShotNoiseGaussianLikelihoodBuilder(
        RESPONSE, np.broadcast_to(data, SHAPE), ShotNoise(**noise), MASK
    )


def test_inverse_std_follows_the_model_and_clips_negative_values():
    r, inverse_std = _builder().model()({"s": jnp.array(SIGNAL)})
    np.testing.assert_array_equal(r, SIGNAL[MASK])
    for s, sky, inv in zip(SIGNAL[MASK], SKY[MASK], np.asarray(inverse_std)):
        variance = A if s + sky <= 0 else A + B * (s + sky)
        assert inv == pytest.approx(variance**-0.5, rel=1e-6)


def test_no_term_depends_on_the_data():
    x = {"s": jnp.array(SIGNAL)}
    faint, bright = (_builder(data=d).model()(x) for d in (0.0, 1e6))
    np.testing.assert_array_equal(faint[1], bright[1])


def test_energy_jits_and_differentiates():
    data = np.full(SHAPE, 3.0)
    like = _builder(data=data).build()
    energy, grad = jax.jit(jax.value_and_grad(like))({"s": jnp.array(SIGNAL)})

    def by_hand(s):  # Gaussian negative log-likelihood without the constant
        variance = A + B * np.maximum(s + SKY, 0)
        return np.sum(((data - s) ** 2 / variance + np.log(variance))[MASK]) / 2

    assert energy == pytest.approx(by_hand(SIGNAL), rel=1e-5)
    step = np.zeros(SHAPE)
    step[0, 2, 3] = 1e-2  # a pixel with positive signal, where σ moves
    numeric = (by_hand(SIGNAL + step) - by_hand(SIGNAL - step)) / 2e-2
    assert grad["s"][0, 2, 3] == pytest.approx(numeric, rel=1e-3)
    assert grad["s"][0, 0, 0] == 0  # masked


@pytest.mark.parametrize("noise", [dict(a=0.0), dict(b=-1.0), dict(sky=np.nan)])
def test_rejects_unusable_noise_on_unmasked_pixels(noise):
    with pytest.raises(ValueError, match="a > 0"):
        _builder(**noise).build()


def test_masked_pixels_may_be_unusable():
    a = np.full(SHAPE, A)
    a[~MASK] = np.nan
    _builder(a=a).build()
