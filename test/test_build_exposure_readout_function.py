import nifty.re as jft
import numpy as np
import pytest

import jubik as ju


@pytest.mark.parametrize("keys", [
    ('tm_1', 'tm_2')])
class TestBuildReadoutFunction:
    @pytest.fixture
    def exposures(self):
        rng = np.random.default_rng(0)
        s_size = 24
        e_size = 3
        return rng.uniform(0.0, 3e3, size=2 * s_size**2 * e_size).reshape(
            (2, e_size, s_size, s_size)
        )

    @pytest.fixture
    def exposure_cut(self):
        return 20

    @pytest.fixture
    def single_exposure(self, exposures):
        return np.expand_dims(exposures[0], axis=0)

    @pytest.fixture
    def x(self):
        s_size = 24
        e_size = 3
        return (
            np.arange(2 * e_size * s_size * s_size, dtype=float)
            .reshape((2, e_size, s_size, s_size))
            + 0.5
        )

    @pytest.fixture
    def single_exposured_sky(self):
        s_size = 24
        e_size = 3
        return (
            np.arange(e_size * s_size * s_size, dtype=float)
            .reshape((1, e_size, s_size, s_size))
            + 1.0
        )

    def test_build_readout_function(self, exposures, exposure_cut,
                                    keys, x):
        flags = exposures.copy()
        mask = flags < exposure_cut
        expected_result = jft.Vector({key: x[i][~mask[i]] for i, key in enumerate(keys)})

        build_exposure_readout = ju.build_readout_function(flags,
                                                           exposure_cut, keys)
        result = build_exposure_readout(x)

        assert tuple(result.tree.keys()) == keys
        for i, key in enumerate(keys):
            res = np.asarray(result.tree[key])
            exp = np.asarray(expected_result.tree[key])
            assert res.ndim == 1
            assert res.size == np.count_nonzero(~mask[i])
            assert np.isfinite(res).all()
            np.testing.assert_array_equal(res, exp)

    def test_build_readout_function_wrong_input_shape(self, exposures,
                                                      exposure_cut, keys, x):
        build_exposure_readout = ju.build_readout_function(exposures.copy(),
                                                           exposure_cut, keys)
        with pytest.raises(ValueError):
            build_exposure_readout(x[0])

    def test_build_readout_function_negative_exposure_cut(self, exposures, keys):
        with pytest.raises(ValueError):
            ju.build_readout_function(exposures.copy(), -1, keys)

    def test_build_readout_function_keys_length_mismatch(self, exposures, exposure_cut):
        with pytest.raises(ValueError):
            ju.build_readout_function(exposures.copy(), exposure_cut, ("tm_1",))

    def test_build_readout_function_with_none_threshold_reads_all_elements(self, exposures, keys, x):
        build_exposure_readout = ju.build_readout_function(exposures.copy(), None, keys)
        result = build_exposure_readout(x)

        for i, key in enumerate(keys):
            np.testing.assert_array_equal(np.asarray(result.tree[key]), x[i].ravel())

    def test_build_readout_function_with_none_keys(self, single_exposure, exposure_cut,
                                                   single_exposured_sky):
        flags = single_exposure.copy()
        mask = flags < exposure_cut
        build_exposure_readout = ju.build_readout_function(flags, exposure_cut, None)
        result = build_exposure_readout(single_exposured_sky)

        expected = single_exposured_sky[0][~mask[0]]
        assert tuple(result.tree.keys()) == ("masked input",)
        assert np.asarray(result.tree["masked input"]).size == np.count_nonzero(~mask[0])
        np.testing.assert_array_equal(np.asarray(result.tree["masked input"]), expected)
