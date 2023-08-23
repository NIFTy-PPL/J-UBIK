import numpy as np
import xubik0 as xu
from jax import random
import nifty8.re as jft


def test_load_and_generate_data():
    config_file_path = 'config_test_sky_model.yaml'
    cfg = xu.get_config(config_file_path)
    key = random.PRNGKey(cfg['seed'])
    key, subkey = random.split(key)
    sky_dict = xu.create_sky_model_from_config('config_test_sky_model.yaml')
    mock_pos = jft.random_like(subkey, sky_dict['sky'].domain)

    loglikelihood = xu.generate_erosita_likelihood_from_config(config_file_path) @ sky_dict['sky']
    assert isinstance(loglikelihood(mock_pos), float)