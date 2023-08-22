import unittest
import numpy as np
import xubik0 as xu


class TestApplyExposure(unittest.TestCase):
    def test_apply_exposure(self):
        size = 100
        shape = (size,)*2
        exposure_cut = 500
        exposures = np.random.uniform(0., 3e3, size=3*size**2).reshape((3, size, size))
        x = np.ones(shape)

        apply_exposure = xu.apply_exposure(exposures, exposure_cut)
        result = apply_exposure(x)
        expected_result = exposures
        expected_result[exposures < 500] = 0

        self.assertEqual(result.all(), expected_result.all())


if __name__ == '__main__':
    unittest.main()
