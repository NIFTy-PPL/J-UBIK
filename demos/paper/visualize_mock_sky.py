import matplotlib.pyplot as plt
import numpy as np
import pickle
from jax import linear_transpose, vmap
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
    results_path = "results/Mock-17092024-001M"
    config_name = "eROSITA_demo.yaml"
    output_dir = ju.create_output_directory(join(results_path, 'paper'))
    config_path = join(results_path, config_name)
    config_dict = ju.get_config(config_path)

    pos = ju.load_mock_position_from_config(config_path)

    sky_model = ju.SkyModel(config_path)
    sky = sky_model.create_sky_model()
    sky_dict = sky_model.sky_model_to_dict()

    sat_max = {'sky': [2e-6, 2e-7, 5e-8], 'diffuse': [2e-6, 2e-7, 5e-8], 'points': [5e-8, 5e-8, 5e-8]}
    for key, op in sky_dict.items():
        real_pos = op(pos)
        bbox_info = [(7, 4), 7, 24,  'black']
        plot_rgb(real_pos, sat_min=[1e-9, 1e-9, 1e-9],
                 sat_max=sat_max[key],
                 sigma=None, log=True,
                 title='simulated sky', fs=18, pixel_measure=28,
                 output_file=join(output_dir, f'simulated_{key}_rgb.png'),
                 alpha=0.5,
                 bbox_info=bbox_info
                 )

        plot(real_pos,
             pixel_measure=28,
             fs=8,
                        title=['0.2-1.0 keV',
                               '1.0-2.0 keV',
                               '2.0-4.5 keV'],
                        logscale=True,
                        colorbar=True,
                        common_colorbar=True,
                        n_rows=1,
                        bbox_info=bbox_info,
                        output_file=join(output_dir,
                        f'simulated_{key}.png'))

