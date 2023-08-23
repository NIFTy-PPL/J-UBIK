from jax import config
from jax import random
import numpy as np
import unittest
import xubik0 as xu


class TestLoadData(unittest.TestCase):

    def test_load_and_generate_data(self):
        config_file_path = 'config_test_sky_model.yaml'
        cfg = xu.get_config(config_file_path)

        file_info = cfg['files']
        grid_info = cfg['grid']
        tel_info = cfg['telescope']
        keys = tel_info['tm_ids']
        exposures = np.random.uniform(0.,
                                      3e3,
                                      size=len(keys) * grid_info['npix'] ** 2).reshape(len(keys),
                                                                                       grid_info['npix'],
                                                                                       grid_info['npix'])

        apply_exposure = xu.apply_exposure(exposures, 500)
        apply_mask = xu.apply_exposure_readout(exposures, 500, keys)
        apply_response = lambda x: apply_mask(apply_exposure(x))
        mock_data = xu.generate_erosita_data_from_config(config_file_path,
                                                         apply_response)
        masked_data = xu.load_erosita_masked_data(file_info, tel_info, apply_mask)

        self.assertEqual(set(mock_data.tree.keys()), set(masked_data.tree.keys()))
        for key in mock_data.tree.keys():
            self.assertEqual(mock_data.tree[key].shape, masked_data.tree[key].shape)


if __name__ == '__main__':
    unittest.main()



