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
    sky_model = ju.SkyModel('config_test_sky_model.yaml')
    sky = sky_model.create_sky_model()

    # Dev test
    # cfg = ju.get_config('config_test_sky_model.yaml')
    # edim = cfg['grid']['edim']
    # emax = cfg['grid']['energy_bin']['e_max']
    # emin = cfg['grid']['energy_bin']['e_min']
    # edistances = (emax -emin)/ edim
    # points_dev_cf, _ = sky_model._create_correlated_field(edim,
    #                                                       edistances,
    #                                                       cfg['priors']['point_sources']['dev'])
    # pos_init = jft.random_like(subkey, points_dev_cf.domain)
    # image = points_dev_cf(pos_init)
    comp_pos = jft.random_like(subkey, sky.domain)
    image = sky(comp_pos)
    for i in range(image.shape[0]):
        plt.imshow(image[i, :, :])
        plt.colorbar()
        plt.show()
        plt.close()
