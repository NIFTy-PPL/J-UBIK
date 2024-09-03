from jax import random
import nifty8.re as jft
import jubik0 as ju


def test_init_sky_model():
    seed = 42
    key = random.PRNGKey(seed)
    key, subkey = random.split(key)
    sky_model = ju.SkyModel('config_test_sky_model.yaml')
    sky = sky_model.create_sky_model()
    comp_pos = jft.random_like(subkey, sky.domain)
    applied_sky = sky(comp_pos)
