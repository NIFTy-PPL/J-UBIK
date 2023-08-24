import numpy as np
import xubik0 as xu
import nifty8.re as jft
import pytest


@pytest.fixture
def exposures():
    size = 100
    return np.random.uniform(0., 3e3, size=2 * size ** 2).reshape((2, size, size))


@pytest.fixture
def exposure_cut():
    return 20


@pytest.fixture
def single_exposure(exposures):
    return np.expand_dims(exposures[0], axis=0)


@pytest.fixture
def keys():
    return ('tm1', 'tm2')


@pytest.fixture
def x():
    size = 100
    exposed_sky_1 = np.ones((size, size)) * 100
    exposed_sky_2 = np.ones_like(exposed_sky_1) * 40
    return np.stack((exposed_sky_1, exposed_sky_2))


@pytest.fixture
def single_exposured_sky():
    size = 100
    return np.expand_dims(np.ones((size, size)) * 100, axis=0)


@pytest.fixture
def expected_result(exposures, exposure_cut, x):
    mask = exposures < exposure_cut
    return jft.Vector({'tm1': x[0][~mask[0]], 'tm2': x[1][~mask[1]]})


def test_build_readout_function(exposures, exposure_cut, keys, x, expected_result):
    build_exposure_readout = xu.build_readout_function(exposures, exposure_cut, keys)
    result = build_exposure_readout(x)
    assert result.tree.keys() == expected_result.tree.keys()
    np.testing.assert_array_equal(list(result.tree.values())[0],
                                  list(expected_result.tree.values())[0])


def test_build_readout_function_wrong_input_shape(exposures, exposure_cut, keys, x):
    build_exposure_readout = xu.build_readout_function(exposures, exposure_cut, keys)
    with pytest.raises(IndexError):
        build_exposure_readout(x[0])


def test_build_readout_function_negative_exposure_cut(exposures, keys):
    with pytest.raises(ValueError):
        xu.build_readout_function(exposures, -1, keys)


def test_build_readout_function_with_None_keys(single_exposure, exposure_cut, x,
                                               single_exposured_sky):
    build_exposure_readout = xu.build_readout_function(single_exposure, exposure_cut, None)
    result = build_exposure_readout(x)
    exposure = single_exposure.copy()
    exposure[exposure < exposure_cut] = 0
    mask = exposure != 0
    expected_result = jft.Vector({'masked input': single_exposured_sky[mask]})
    assert result.tree.keys() == expected_result.tree.keys()
    np.testing.assert_array_equal(list(result.tree.values())[0],
                                  list(expected_result.tree.values())[0])
