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
    output_dir = "paper/"
    config_path = join(output_dir, 'erosita_data_plotting_config.yaml')
    config_dict = ju.get_config(config_path)
    grid_info = config_dict["grid"]
    file_info = config_dict["files"]
    tel_info = config_dict["telescope"]
    tm_ids = tel_info["tm_ids"]
    e_min = grid_info['energy_bin']['e_min']
    e_max = grid_info['energy_bin']['e_max']

    epix = grid_info['edim']
    spix = grid_info['sdim']
    response_dict = ju.build_erosita_response_from_config(config_path)
    mask = response_dict['mask']
    data = ju.mask_erosita_data_from_disk(file_info=file_info,
                                   grid_info=grid_info,
                                   tel_info=tel_info,
                                   mask_func=mask)
    plottable_vector = jft.Vector({key: val.astype(float) for key, val
                                   in data.tree.items()})
    mask_adj = linear_transpose(mask,
                                np.zeros((len(tm_ids), epix, spix, spix)))
    mask_adj_func = lambda x: mask_adj(x)[0]


    # Plotting the data
    unmasked_data = mask_adj_func(plottable_vector)
    plotting_kwargs = {'vmin':1e0, 'vmax':4e2}

    bbox_info = [(28, 16), 28, 160, 'black']
    for i in range(unmasked_data.shape[0]):
        plot(unmasked_data[i],
             pixel_measure=112,
             fs=8,
                        title=['0.2-1.0 keV',
                               '1.0-2.0 keV',
                               '2.0-4.5 keV'],
                        logscale=True,
                        colorbar=True,
                        common_colorbar=True,
                        n_rows=1,
                        alpha=0.5,
                        bbox_info= bbox_info,
                        pixel_factor=4,
                        output_file=join(output_dir,
                        f'eROSITA_tm{tm_ids[i]}_data.png'),
                        **plotting_kwargs)
    summed_data = np.sum(unmasked_data, axis=0)
    bbox_info = [(28, 16), 28,  160, 'black']
    plot(unmasked_data[i],
         pixel_measure=112,
         fs=8,
         title=['0.2-1.0 keV',
                '1.0-2.0 keV',
                '2.0-4.5 keV'],
         logscale=True,
         colorbar=True,
         common_colorbar=True,
         n_rows=1,
         output_file=join(output_dir,
                          f'summed_eROSITA_data.png'),
         pixel_factor=4,
         bbox_info=bbox_info,
         alpha=0.5,
         **plotting_kwargs)
    bbox_info = [(28, 16), 28,  160, 'black']
    plot_rgb(summed_data, sat_min=[0, 0, 0],
             sat_max=[4e2, 1e2, 1e1],
             sigma=None, log=True,
             title='eROSITA LMC data', fs=18, pixel_measure=112,
             output_file=join(output_dir, 'eRSOITA_data_rgb.png'),
             alpha=0.0,
             pixel_factor=4,
             bbox_info=bbox_info
             )