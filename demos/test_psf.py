import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

import jubik0 as ju
from jax import config, random, linear_transpose
config.update('jax_enable_x64', True)


if __name__ == "__main__":
    config_filename = "configs/eROSITA_demo_full_test.yaml"
    cfg = ju.get_config(config_filename)
    seed = 88
    key = random.PRNGKey(seed)

    # Load sky model
    sky_model = ju.SkyModel(cfg)
    sky = sky_model.create_sky_model()
    sky_dict = sky_model.sky_model_to_dict()
    # ju.generate_erosita_data_from_config(cfg)
    # Load parameters
    tm_ids = cfg['telescope']['tm_ids']
    n_modules = len(tm_ids)

    spix = cfg['grid']['sdim']
    epix = cfg['grid']['edim']

    # Load response
    # response_dict = ju.build_erosita_response_from_config(cfg)

    # mask_adj = linear_transpose(response_dict['mask'],
    #                             np.zeros(
    #                                 (n_modules, epix, spix, spix)))
    # if 'kernel' in response_dict:
    #     R = lambda x: mask_adj(
    #         response_dict['R'](x, response_dict['kernel']))[0]
    psf_func, kernel = ju.build_psf_from_config(cfg)

    # plot random schtuff
    key, subkey = random.split(key)
    random_sky = sky(sky.init(subkey))

    random_sky = np.zeros((1024, 1024), dtype=int)


    spacing = 100
    random_sky[::spacing, ::spacing] = 1
    random_sky = random_sky[None, :]

    plt.imshow(psf_func(random_sky, kernel)[0][0], #[62:-62,62:-62],
               origin='lower', vmax=0.01) #, norm=LogNorm())
    plt.colorbar()
    plt.show()
