import unittest
import numpy as np
import xubik0 as xu


class TestApplyExposureReadout(unittest.TestCase):
    def setUp(self):
        size = 100
        self.exposures = np.random.uniform(0., 3e3, size=2 * size ** 2).reshape((2, size, size))
        self.exposure_cut = 10
        self.keys = ('tm1', 'tm2')
        exposed_sky_1 = np.ones((size, size)) * 512
        exposed_sky_2 = np.ones_like(exposed_sky_1) * 40

        self.x = np.stack((exposed_sky_1, exposed_sky_2))
        self.expected_result = {'tm1': exposed_sky_1[exposed_sky_1 != 0],
                                'tm2': exposed_sky_2[exposed_sky_2 != 0]}

    def test_apply_exposure_readout(self):
        apply_exposure_readout = xu.apply_exposure_readout(self.exposures, self.exposure_cut,
                                                           self.keys)
        result = apply_exposure_readout(self.x)
        self.assertEqual(result.keys(), self.expected_result.keys())
        self.assertEqual(list(result.values())[0].all(),
                         list(self.expected_result.values())[0].all())

    def test_apply_exposure_readout_wrong_input_shape(self):
        apply_exposure_readout = xu.apply_exposure_readout(self.exposures, self.exposure_cut,
                                                           self.keys)

        x = self.x[0]
        with self.assertRaises(ValueError):
            apply_exposure_readout(x)

    def test_apply_exposure_readout_negative_exposure_cut(self):
        with self.assertRaises(ValueError):
            xu.apply_exposure_readout(self.exposures, -1, self.keys)


if __name__ == '__main__':
    unittest.main()
