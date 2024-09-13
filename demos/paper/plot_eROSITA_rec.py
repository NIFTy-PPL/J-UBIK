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
    results_path = "results/LMC-12092024-002M"
    config_name = "eROSITA_config_02.yaml"
    output_dir = ju.create_output_directory(join(results_path, 'paper'))
    config_path = join(results_path, config_name)
    config_dict = ju.get_config(config_path)
    tel_info = config_dict["telescope"]
    tm_ids = tel_info["tm_ids"]

    grid_info = config_dict["grid"]
    epix = grid_info['edim']
    spix = grid_info['sdim']

    with open(os.path.join(results_path, 'last.pkl'), "rb") as file:
        samples, _ = pickle.load(file)

    sky_model = ju.SkyModel(config_path)
    sky = sky_model.create_sky_model()
    sky_dict = sky_model.sky_model_to_dict()

    response_dict = ju.build_erosita_response_from_config(config_path)
    exit()
    mask = ju.build_readout_function()
    mask = response_dict['mask']
    mask_adj = linear_transpose(mask,
                                np.zeros((len(tm_ids), epix, spix, spix)))
    mask_adj_func = lambda x: mask_adj(x)[0]



    sat_max = {'sky': [1e-4, 8e-5, 3e-5], 'diffuse': [1e-4, 8e-5, 3e-5], 'points': [1e-4, 8e-5, 3e-5]}
    sat_min = {'sky': [8e-10, 5e-10, 2e-10], 'diffuse': [8e-10, 5e-10, 2e-10], 'points': [8e-10, 5e-10, 2e-10]}
    for key, op in sky_dict.items():
        op = jax.vmap(op)
        real_samples = op(samples.samples)
        real_mean = jnp.mean(real_samples, axis=0)
        real_mean = np.mean(mask_adj_func(mask(np.repeat(real_mean[np.newaxis, :,
                                                 :, :], 5, axis=0))), axis=0)

        bbox_info = [(7, 4), 7, 24, 'black']
        plot_rgb(real_mean, sat_min=sat_min[key],
                 sat_max=sat_max[key],
                 sigma=None, log=True,
                 title= f'reconstructed {key}', fs=18, pixel_measure=28,
                 output_file=join(output_dir, f'mock_rec_{key}_rgb.png'),
                 alpha=0.5,
                 bbox_info=bbox_info
                 )
        plotting_kwargs_rec = {}
        plot(real_mean,
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
                        f'rec_{key}.png'),
                        **plotting_kwargs_rec)
        real_std = jnp.std(real_samples, axis=0)
        real_std = np.mean(mask_adj_func(mask(np.repeat(real_std[np.newaxis, :,
                                                 :, :], 5, axis=0))), axis=0)
        plotting_kwargs_unc = {'cmap': 'seismic'}
        plot(real_std,
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
                        f'unc_{key}.png'),
                        **plotting_kwargs_unc)


