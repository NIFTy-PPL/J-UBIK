import matplotlib.pyplot as plt
import numpy as np
import pickle
from jax import linear_transpose, vmap, random
import jax.numpy as jnp


import jubik0 as ju
from os.path import join
import astropy.io.fits as fits
import nifty8.re as jft
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plot_eROSITA_image import plot, plot_rgb


# Script for plotting the data, position and reconstruction images
if __name__ == "__main__":
    results_path = "results/LMC-12092024-002M"
    config_name = "eROSITA_config_02.yaml"
    output_dir = ju.create_output_directory(join(results_path, 'paper'))
    config_path = join(results_path, config_name)
    config_dict = ju.get_config(config_path)

    sky_model = ju.SkyModel(config_path)
    sky = sky_model.create_sky_model()
    sky_dict = sky_model.sky_model_to_dict()

    key = random.PRNGKey(81)
    key, subkey = random.split(key)

    pos = jft.Vector(jft.random_like(subkey, sky.domain))


    for key, op in sky_dict.items():
        real_pos = op(pos)
        bbox_info = [(28, 16), 28, 160, 'black']
        plot_rgb(real_pos, sat_min=[1e-10, 1e-10, 1e-10],
                 sat_max=[4e-7, 1e-7, 1e-8],
                 sigma=None, log=True,
                 title='simulated sky', fs=18, pixel_measure=112,
                 output_file=join(output_dir, f'prior_{key}_rgb.png'),
                 alpha=0.5,
                 pixel_factor=4,
                 bbox_info=bbox_info
                 )

        plot(real_pos,
             pixel_factor=4,
             pixel_measure=112,
             fs=12,
                        title=['0.2-1.0 keV',
                               '1.0-2.0 keV',
                               '2.0-4.5 keV'],
                        logscale=True,
                        colorbar=True,
                        common_colorbar=True,
                        n_rows=1,
                        bbox_info=bbox_info,
                        output_file=join(output_dir,
                        f'prior_{key}.png'))

    alpha_diffuse = sky_model.alpha_cf(pos)
    ju.plot_result(alpha_diffuse, output_file=join(output_dir, "alpha_diffuse.png"))
    spatial_diffuse = sky_model.spatial_cf(pos)
    ju.plot_result(spatial_diffuse, output_file=join(output_dir, "spatial_diffuse.png"))
