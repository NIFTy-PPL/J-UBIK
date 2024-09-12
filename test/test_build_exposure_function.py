import numpy as np
import pytest

import jubik0 as ju


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
    return np.random.uniform(0., 3e3, size=3 * size ** 2).reshape((3,
                                                                   size,
                                                                   size))

@pytest.fixture
def x(shape):
    return np.ones(shape)

@pytest.fixture
def build_exposure(exposures, exposure_cut):
    return ju.build_exposure_function(exposures, exposure_cut)

def test_build_exposure(build_exposure, exposures, x, exposure_cut):
    result = build_exposure(x)
    expected_result = exposures.copy()
    expected_result[exposures < exposure_cut] = 0

    np.testing.assert_array_equal(result, expected_result)

