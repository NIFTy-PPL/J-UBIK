import unittest
import numpy as np
import xubik0 as xu
import nifty8.re as jft


class TestApplyExposureReadout(unittest.TestCase):
    def setUp(self):
        size = 100
        self.exposures = np.random.uniform(0., 3e3, size=2 * size ** 2).reshape((2, size, size))
        self.exposure_cut = 10
        self.single_exposure = self.exposures[0]
        self.keys = ('tm1', 'tm2')
        self.exposed_sky_1 = np.ones((size, size)) * 512
        exposed_sky_2 = np.ones_like(self.exposed_sky_1) * 40

        self.x = np.stack((self.exposed_sky_1, exposed_sky_2))
        self.expected_result = jft.Vector({'tm1':self.exposed_sky_1[self.exposed_sky_1 != 0],
                                'tm2': exposed_sky_2[exposed_sky_2 != 0]})

    def test_apply_exposure_readout(self):
        apply_exposure_readout = xu.apply_exposure_readout(self.exposures, self.exposure_cut,
                                                           self.keys)
        result = apply_exposure_readout(self.x)
        self.assertEqual(result.tree.keys(), self.expected_result.tree.keys())
        self.assertEqual(list(result.tree.values())[0].all(),
                         list(self.expected_result.tree.values())[0].all())

    def test_apply_exposure_readout_wrong_input_shape(self):
        apply_exposure_readout = xu.apply_exposure_readout(self.exposures, self.exposure_cut,
                                                           self.keys)

        x = self.x[0]
        with self.assertRaises(ValueError):
            apply_exposure_readout(x)

    def test_apply_exposure_readout_negative_exposure_cut(self):
        with self.assertRaises(ValueError):
            xu.apply_exposure_readout(self.exposures, -1, self.keys)

    def test_apply_exposure_readout_with_None_keys(self):
        apply_exposure_readout = xu.apply_exposure_readout(self.single_exposure, self.exposure_cut,
                                                           keys=None)
        result = apply_exposure_readout(self.x[0])
        expected_result = jft.Vector({'masked input': self.exposed_sky_1[self.exposed_sky_1 != 0]})
        self.assertEqual(result.tree.keys(), expected_result.tree.keys())
        self.assertEqual(list(result.tree.values())[0].all(),
                         list(expected_result.tree.values())[0].all())

    # TODO: more tests on this module. Test the apply_from_exposure_file. Test exp_cut
if __name__ == '__main__':
    unittest.main()
