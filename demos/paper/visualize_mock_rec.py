import matplotlib.pyplot as plt
import numpy as np
import pickle
from jax import linear_transpose, vmap
import jax.numpy as jnp
import jax

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
    results_path = "results/LMC-06082024-002M-mock"
    config_name = "eROSITA_config_small.yaml"
    output_dir = ju.create_output_directory(join(results_path, 'paper'))
    config_path = join(results_path, config_name)
    config_dict = ju.get_config(config_path)

    with open(os.path.join(results_path, 'last.pkl'), "rb") as file:
        samples, _ = pickle.load(file)

    sky_model = ju.SkyModel(config_path)
    sky = sky_model.create_sky_model()
    sky_dict = sky_model.sky_model_to_dict()

    sat_max = {'sky': [2e-6, 2e-7, 5e-8], 'diffuse': [2e-6, 2e-7, 5e-8], 'points': [5e-8, 5e-8, 5e-8]}
    sat_min = {'sky': [1e-9, 1e-9, 1e-9], 'diffuse': [1e-9, 1e-9, 1e-9], 'points': [1e-12, 1e-12, 1e-12]}
    for key, op in sky_dict.items():
        op = jax.vmap(op)
        real_samples = op(samples.samples)
        real_mean = jnp.mean(real_samples, axis=0)
        bbox_info = [(7, 4), 7, 24]
        plot_rgb(real_mean, sat_min=sat_min[key],
                 sat_max=sat_max[key],
                 sigma=None, log=True,
                 title= f'reconstructed {key}', fs=18, pixel_measure=28,
                 output_file=join(output_dir, f'mock_rec_{key}.png'),
                 alpha=0.5,
                 bbox_info=bbox_info
                 )

