import nifty8.re as jft
import numpy as np
import pytest

import jubik0 as ju


@pytest.mark.parametrize(
    "keys", [("tm_1", "tm_2")]
)  # TODO: test more cases (e.g. the single tm case which would now fail.)
class TestBuildReadoutFunction:
    @pytest.fixture
    def exposures(self):
        s_size = 100
        e_size = 3
        return np.random.uniform(0.0, 3e3, size=2 * s_size**2 * e_size).reshape(
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
        s_size = 100
        e_size = 3
        exposed_sky_1 = np.ones((e_size, s_size, s_size)) * 100
        exposed_sky_2 = np.ones_like(exposed_sky_1) * 40
        return np.stack((exposed_sky_1, exposed_sky_2))

    @pytest.fixture
    def single_exposured_sky(self):
        s_size = 100
        e_size = 3
        return np.expand_dims(np.ones((e_size, s_size, s_size)) * 100, axis=0)

    @pytest.fixture
    def expected_result(self, exposures, exposure_cut, x, keys):
        mask = exposures < exposure_cut
        return jft.Vector({key: x[i][~mask[i]] for i, key in enumerate(keys)})

    def test_build_readout_function(
        self, exposures, exposure_cut, keys, x, expected_result
    ):
        build_exposure_readout = ju.build_readout_function(
            exposures, exposure_cut, keys
        )
        result = build_exposure_readout(x)
        assert result.tree.keys() == expected_result.tree.keys()
        np.testing.assert_array_equal(
            list(result.tree.values())[0], list(expected_result.tree.values())[0]
        )

    def test_build_readout_function_wrong_input_shape(
        self, exposures, exposure_cut, keys, x
    ):
        build_exposure_readout = ju.build_readout_function(
            exposures, exposure_cut, keys
        )
        with pytest.raises(ValueError):
            build_exposure_readout(x[0])

    def test_build_readout_function_negative_exposure_cut(self, exposures, keys):
        with pytest.raises(ValueError):
            ju.build_readout_function(exposures, -1, keys)

    def test_build_readout_function_with_none_keys(
        self, single_exposure, exposure_cut, x, single_exposured_sky, keys
    ):
        build_exposure_readout = ju.build_readout_function(
            single_exposure, exposure_cut, None
        )
        result = build_exposure_readout(x)
        exposure = single_exposure.copy()
        exposure[exposure < exposure_cut] = 0
        mask = exposure != 0
        expected_result = jft.Vector({"masked input": single_exposured_sky[mask]})
        assert result.tree.keys() == expected_result.tree.keys()
        np.testing.assert_array_equal(
            list(result.tree.values())[0], list(expected_result.tree.values())[0]
        )
