from jax import config
from jax import random
import numpy as np
import xubik0 as xu

config.update('jax_enable_x64', True)
#TODO turn into pytest compatible test function
if __name__ == "__main__":
    seed = 42
    key = random.PRNGKey(seed)
    key, subkey = random.split(key)
    config_file_path = 'config_test_sky_model.yaml'
    cfg = xu.get_config(config_file_path)

    file_info = cfg['files']
    priors = cfg['priors']
    grid_info = cfg['grid']
    tel_info = cfg['telescope']

    keys = tel_info['tm_ids']
    exposures = np.random.uniform(0.,
                                  3e3,
                                  size=len(keys)*grid_info['npix']**2).reshape(len(keys),
                                                                    grid_info['npix'],
                                                                    grid_info['npix'])
    apply_exposure = xu.apply_exposure(exposures, 500)
    mock_data = xu.generate_erosita_data_from_config(config_file_path,
                                                     apply_exposure)

    apply_mask = xu.apply_exposure_readout(exposures, 500, keys)
    for _, val in apply_mask(mock_data).items():
        print(val.shape)

    masked_data = xu.load_erosita_masked_data(file_info, tel_info, apply_mask)
    print(masked_data)
