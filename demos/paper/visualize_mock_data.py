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
    results_path = "results/LMC-06082024-002M-mock"
    config_name = "eROSITA_config_small.yaml"
    output_dir = ju.create_output_directory(join(results_path, 'paper'))
    config_path = join(results_path, config_name)
    config_dict = ju.get_config(config_path)
    grid_info = config_dict["grid"]
    tm_ids = config_dict["telescope"]["tm_ids"]
    e_min = grid_info['energy_bin']['e_min']
    e_max = grid_info['energy_bin']['e_max']
    data_path = join(results_path, 'data.pkl')


    data = ju.load_masked_data_from_config(config_path)
    plottable_vector = jft.Vector({key: val.astype(float) for key, val
                                   in data.tree.items()})
    n_modules = len(data.tree)
    epix = grid_info['edim']
    spix = grid_info['sdim']
    response_dict = ju.build_erosita_response_from_config(config_path)
    mask = response_dict['mask']
    mask_adj = linear_transpose(mask,
                                np.zeros((n_modules, epix, spix, spix)))
    mask_adj_func = lambda x: mask_adj(x)[0]
    tms = plottable_vector.tree.keys()
    # Plotting the data
    unmasked_data = mask_adj_func(plottable_vector)
    plotting_kwargs = {'vmin':1e0, 'vmax':1.1e2}
    for i in range(unmasked_data.shape[0]):
        plot(unmasked_data[i],
             pixel_measure=28,
             fs=8,
                        title=['0.2-1.0 keV',
                               '1.0-2.0 keV',
                               '2.0-4.5 keV'],
                        logscale=True,
                        colorbar=True,
                        common_colorbar=True,
                        n_rows=1,
                        output_file=join(output_dir,
                        f'tm{list(tms)[i]}_data.png'),
                        **plotting_kwargs)
    summed_data = np.sum(unmasked_data, axis=0)
    bbox_info = [(7, 4), 7,  20]
    plot(unmasked_data[i],
         pixel_measure=28,
         fs=8,
         title=['0.2-1.0 keV',
                '1.0-2.0 keV',
                '2.0-4.5 keV'],
         logscale=True,
         colorbar=True,
         common_colorbar=True,
         n_rows=1,
         output_file=join(output_dir,
                          f'summed_data.png'),
         bbox_info=bbox_info,
         **plotting_kwargs)
    bbox_info = [(7, 4), 7,  24]
    plot_rgb(summed_data, sat_min=[0, 0, 0],
             sat_max=[2e2, 8e1, 8e0],
             sigma=None, log=True,
             title='simulated data', fs=18, pixel_measure=28,
             output_file=join(output_dir, 'simulated_data_rgb.png'),
             alpha=0.5,
             bbox_info=bbox_info
             )

