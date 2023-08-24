import numpy as np
import pytest
import xubik0 as xu


class TestBuildCallableFromExposureFile:
    # TODO: add .fits tests
    @pytest.fixture
    def file(self):
        return "./misc/erosita_example_exposure.npy"

    @pytest.fixture
    def file_list(self):
        return ["./misc/erosita_example_exposure.npy"]

    def test_build_callable_from_exposure_file(self, file_list):
        R = xu.build_callable_from_exposure_file(xu.build_erosita_response, file_list,
                                                 exposure_cut=0)
        sky = np.ones((426, 426))
        result = R(sky)['masked input'][20:30]
        expected_result = np.array(
            [4.12181568, 4.13665915, 4.20947599, 4.33986044, 4.41858196, 4.51194668,
             4.602139, 4.64281225, 4.71162701, 4.77253675])
        np.testing.assert_allclose(result, expected_result)

    def test_build_callable_from_exposure_file_not_list(self, file):
        with pytest.raises(ValueError):
            xu.build_callable_from_exposure_file(xu.build_erosita_response, file)


if __name__ == '__main__':
    pytest.main()
