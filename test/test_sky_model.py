from jax import config
from jax import random
import matplotlib.pyplot as plt

import nifty8.re as jft
import jubik0 as ju


config.update('jax_enable_x64', True)
#TODO turn into pytest compatible test function
if __name__ == "__main__":
    # Load config file
    seed = 42
    key = random.PRNGKey(seed)
    key, subkey = random.split(key)
    sky_dict = ju.create_sky_model_from_config('config_test_sky_model.yaml')
    sky_dict.pop('pspec')
    for component in sky_dict.keys():
        comp_pos = jft.random_like(subkey, sky_dict[component].domain)
        image = sky_dict[component](comp_pos)
        plt.imshow(image)
        plt.show()
        plt.close()
