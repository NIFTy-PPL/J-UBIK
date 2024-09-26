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
    results_path = "results/Mock-17092024-001M"
    config_name = "eROSITA_demo.yaml"
    output_dir = ju.create_output_directory(join(results_path, 'paper'))
    config_path = join(results_path, config_name)
    config_dict = ju.get_config(config_path)

    with open(os.path.join(results_path, 'last.pkl'), "rb") as file:
        samples, _ = pickle.load(file)

    sky_model = ju.SkyModel(config_path)
    sky = sky_model.create_sky_model()
    sky_dict = sky_model.sky_model_to_dict()

    for key, op in sky_dict.items():
        op = jax.vmap(op)
        real_samples = op(samples.samples)
        real_mean = jnp.mean(real_samples, axis=0)
        bbox_info = [(7, 4), 7, 24, 'black']
        plot_rgb(real_mean,
                 sat_min=[1e-10, 1e-10, 1e-10],
                 sat_max=[5e-8, 5e-8, 5e-8],
                 sigma=None,
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
                        f'mock_rec_{key}.png'),
                        **plotting_kwargs_rec)
        real_std = jnp.std(real_samples, axis=0)
        plotting_kwargs_unc = {'cmap': 'Blues'}
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
                        f'mock_unc_{key}.png'),
                        **plotting_kwargs_unc)
    points_op = jax.vmap(sky_dict['points'])
    point_samples = points_op(samples.samples)
    real_points_cut = np.mean(point_samples, axis=0).at[\
        np.mean(point_samples, axis=0)<2.5e-9].set(0)
    diffuse_op = jax.vmap(sky_dict['diffuse'])
    diffuse_samples = diffuse_op(samples.samples)
    real_diffuse = np.mean(diffuse_samples, axis=0)

    cut_sky = real_points_cut + real_diffuse

    plot_rgb(cut_sky,
             sat_min=[1e-10, 1e-10, 1e-10],
             sat_max=[5e-8, 5e-8, 5e-8],
             sigma=None,
             title=f'reconstructed {key}', fs=18, pixel_measure=28,
             output_file=join(output_dir, f'mock_rec_cut_{key}_rgb.png'),
             alpha=0.5,
             bbox_info=bbox_info
             )


