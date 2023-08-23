import unittest
import numpy as np
import xubik0 as xu
import nifty8.re as jft


class TestBuildExposureReadoutFunction(unittest.TestCase):
    def setUp(self):
        size = 100
        self.exposures = np.random.uniform(0., 3e3, size=2 * size ** 2).reshape((2, size, size))
        self.exposure_cut = 20
        self.single_exposure = np.expand_dims(self.exposures[0], axis=0)
        self.keys = ('tm1', 'tm2')
        exposed_sky_1 = np.ones((size, size)) * 100
        exposed_sky_2 = np.ones_like(exposed_sky_1) * 40

        self.single_exposured_sky = np.expand_dims(np.ones((size, size)) * 100, axis=0)
        self.x = np.stack((exposed_sky_1, exposed_sky_2))
        mask = self.exposures < self.exposure_cut
        self.expected_result = jft.Vector({'tm1': exposed_sky_1[~mask[0]],
                                           'tm2': exposed_sky_2[~mask[1]]})

    def test_build_exposure_readout_function(self):
        build_exposure_readout = xu.build_exposure_readout_function(self.exposures,
                                                                    self.exposure_cut,
                                                                    self.keys)
        result = build_exposure_readout(self.x)
        self.assertEqual(result.tree.keys(), self.expected_result.tree.keys())
        np.testing.assert_array_equal(list(result.tree.values())[0],
                         list(self.expected_result.tree.values())[0])

    def test_build_exposure_readout_wrong_input_shape(self):
        build_exposure_readout = xu.build_exposure_readout_function(self.exposures,
                                                                    self.exposure_cut,
                                                                    self.keys)

        x = self.x[0]
        with self.assertRaises(IndexError):
            build_exposure_readout(x)

    def test_build_exposure_readout_negative_exposure_cut(self):
        with self.assertRaises(ValueError):
            xu.build_exposure_readout_function(self.exposures, -1, self.keys)

    def test_build_exposure_readout_with_None_keys(self):
        build_exposure_readout = xu.build_exposure_readout_function(self.single_exposure,
                                                                    exposure_cut=self.exposure_cut,
                                                                    keys=None)
        result = build_exposure_readout(self.x)
        exposure = self.single_exposure.copy()
        exposure[exposure < self.exposure_cut] = 0
        mask = exposure != 0

        expected_result = jft.Vector({'masked input': self.single_exposured_sky[mask]})
        self.assertEqual(result.tree.keys(), expected_result.tree.keys())
        np.testing.assert_array_equal(list(result.tree.values())[0],
                                      list(expected_result.tree.values())[0])


# TODO: more tests on this module. Test the build_from_exposure_file. Test exp_cut.
if __name__ == '__main__':
    unittest.main()
