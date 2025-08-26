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
    eROSITA_config_name = "configs/eROSITA_demo.yaml"
    chandra_config_name = "configs/chandra_demo.yaml"
    output_dir = ju.create_output_directory("paper/")
    eROSITA_cfg_dict = ju.get_config(eROSITA_config_name)
    chandra_cfg_dict = ju.get_config(chandra_config_name)
    config_dicts = {'eROSITA': eROSITA_cfg_dict,
                    'Chandra': chandra_cfg_dict}

    prior_config_path = "paper/prior_config.yaml"
    prior_config_dict = ju.get_config(prior_config_path)

    sky_model = ju.SkyModel(prior_config_path)
    sky = sky_model.create_sky_model()

    pos = ju.load_from_pickle('paper/pos.pkl')

    # eROSITA:
    response_dict = ju.build_erosita_response_from_config(eROSITA_config_name)
    masked_mock_data = response_dict['R'](sky(pos))
    key = random.PRNGKey(67)
    key, subkey = random.split(key)
    masked_mock_data = jft.Vector({
        tm: random.poisson(subkey, data).astype(int)
        for i, (tm, data) in enumerate(masked_mock_data.tree.items())
    })
    plottable_vector = jft.Vector({key: val.astype(float) for key, val
                                   in masked_mock_data.tree.items()})
    mask = response_dict['mask']
    mask_adj = linear_transpose(mask,
                                np.zeros((1, 1, 1024, 1024)))
    mask_adj_func = lambda x: mask_adj(x)[0]
    tms = plottable_vector.tree.keys()
    # Plotting the data
    unmasked_data = mask_adj_func(plottable_vector)

    plottabel_data_list = {'eROSITA': unmasked_data}
    exit()
    plotting_kwargs = {'vmin':1e0, 'vmax':4e2}
    bbox_info = [(7, 4), 7,  20, 'black']
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
                        bbox_info=bbox_info,
                        output_file=join(output_dir,
                        f'tm{list(tms)[i]}_data.png'),
                        **plotting_kwargs)
    summed_data = np.sum(unmasked_data, axis=0)
    bbox_info = [(7, 4), 7,  20, 'black']
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
    bbox_info = [(7, 4), 7,  24, 'black']
    plot_rgb(summed_data, sat_min=[0, 0, 0],
             sat_max=[4e2, 1e2, 1e1],
             sigma=None, log=True,
             title='simulated data', fs=18, pixel_measure=28,
             output_file=join(output_dir, 'simulated_data_rgb.png'),
             alpha=0.5,
             bbox_info=bbox_info
             )

