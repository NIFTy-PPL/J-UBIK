import numpy as np

import jubik0 as ju


def test_load_and_generate_data():
    config_file_path = 'config_test_sky_model.yaml'
    cfg = ju.get_config(config_file_path)

    file_info = cfg['files']
    grid_info = cfg['grid']
    tel_info = cfg['telescope']
    keys = tel_info['tm_ids']
    exposures = np.random.uniform(0.,
                                  3e3,
                                  size=len(keys) * grid_info[
                                      'npix'] ** 2).reshape(len(keys),
                                                            grid_info['npix'],
                                                            grid_info['npix'])

    apply_exposure = ju.apply_exposure(exposures, 500)
    apply_mask = ju.apply_exposure_readout(exposures, 500, keys)
    apply_response = lambda x: apply_mask(apply_exposure(x))
    mock_data = ju.generate_erosita_data_from_config(config_file_path,
                                                     apply_response)
    masked_data = ju.load_erosita_masked_data(file_info, tel_info, apply_mask)

    assert set(mock_data.tree.keys()) == set(masked_data.tree.keys())
    for key in mock_data.tree.keys():
        assert mock_data.tree[key].shape == masked_data.tree[key].shape
