import jubik0 as ju
from jax import random
import nifty8.re as jft


def test_likelihood():
    config_file_path = 'config_test_sky_model.yaml'
    cfg = ju.get_config(config_file_path)
    key = random.PRNGKey(cfg['seed'])
    key, subkey = random.split(key)
    sky = ju.SkyModel('config_test_sky_model.yaml').create_sky_model()
    mock_pos = jft.random_like(subkey, sky_dict['sky'].domain)

    loglikelihood = ju.generate_erosita_likelihood_from_config(config_file_path) @ sky
    loglikelihood(mock_pos)
    # FIXME: Add asserts here

