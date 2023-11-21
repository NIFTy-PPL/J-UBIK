import numpy as np
import pytest
import jubik0 as ju


def test_build_exposure():
    size = 100
    shape = (size,) * 2
    exposure_cut = 500
    exposures = np.random.uniform(0., 3e3, size=3 * size ** 2).reshape((3, size, size))
    x = np.ones(shape)

    build_exposure = ju.build_exposure_function(exposures, exposure_cut)
    result = build_exposure(x)
    expected_result = exposures
    expected_result[exposures < 500] = 0

    np.testing.assert_array_equal(result, expected_result)


if __name__ == '__main__':
    pytest.main()
