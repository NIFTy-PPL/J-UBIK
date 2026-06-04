import numpy as np
import pytest

import jubik as ju


@pytest.fixture
def size():
    return 100

@pytest.fixture
def shape(size):
    return (size,) * 2

@pytest.fixture
def exposure_cut():
    return 500

@pytest.fixture
def exposures(size):
    rng = np.random.default_rng(0)
    return rng.uniform(0.0, 3e3, size=3 * size**2).reshape((3, size, size))

@pytest.fixture
def x(shape):
    return np.linspace(0.25, 1.75, num=shape[0] * shape[1]).reshape(shape)


def test_build_exposure_applies_threshold_and_multiplies_input(exposures, x, exposure_cut):
    result = ju.build_exposure_function(exposures.copy(), exposure_cut)(x)

    expected_result = exposures.copy()
    expected_result[expected_result < exposure_cut] = 0
    expected_result = expected_result * x

    assert result.shape == exposures.shape
    assert np.isfinite(result).all()
    assert (result >= 0).all()
    np.testing.assert_allclose(result, expected_result)


def test_build_exposure_without_cut_only_multiplies(exposures, x):
    result = ju.build_exposure_function(exposures.copy(), None)(x)
    expected_result = exposures * x

    assert result.shape == exposures.shape
    np.testing.assert_allclose(result, expected_result)


def test_build_exposure_negative_cut_raises(exposures):
    with pytest.raises(ValueError):
        ju.build_exposure_function(exposures.copy(), -1)
