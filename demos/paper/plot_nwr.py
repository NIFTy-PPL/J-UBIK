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
    results_path = "results/LMC-10092024-002M"
    config_name = "eROSITA_config.yaml"
    output_dir = ju.create_output_directory(join(results_path, 'paper'))
    config_path = join(results_path, config_name)
    config_dict = ju.get_config(config_path)
    tm_ids = config_dict["telescope"]["tm_ids"]

    with open(os.path.join(results_path, 'last.pkl'), "rb") as file:
        samples, _ = pickle.load(file)

    sky_model = ju.SkyModel(config_path)
    sky = sky_model.create_sky_model()
    sky_dict = sky_model.sky_model_to_dict()
    response_dict = ju.build_erosita_response_from_config(config_path)

    data = ju.load_masked_data_from_config(config_path)
    plottable_vector = jft.Vector({key: val.astype(float) for key, val
                                   in data.tree.items()})
    grid_info = config_dict["grid"]
    epix = grid_info['edim']
    spix = grid_info['sdim']
    mask_adj = linear_transpose(response_dict['mask'],
                                np.zeros((len(tm_ids), epix, spix, spix)))
    response_dict['mask_adj'] = mask_adj
    # mask_adj_func = lambda x: mask_adj(x)[0]

    #  unmasked_data = mask_adj_func(plottable_vector)
    for key, op in sky_dict.items():
        nwrs, exp_mask = ju.calculate_nwr(samples.samples, op,
                                          data,
                                          response_dict)
        bbox_info = [(28, 16), 28, 160, 'black']

        plotting_kwargs = {'vmin': -5, 'vmax': 5, 'cmap': 'RdBu'}
        plot(np.mean(np.mean(nwrs, axis=0),axis=0),
             pixel_factor=4,
             pixel_measure=112,
             fs=12,
                        title=['0.2-1.0 keV',
                               '1.0-2.0 keV',
                               '2.0-4.5 keV'],
                        logscale=False,
                        colorbar=True,
                        n_rows=1,
             bbox_info=bbox_info,
                        output_file=join(output_dir,
                        f'nwr_{key}.png'),
             **plotting_kwargs)
