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
    tm_ids = config_dict["telescope"]["tm_ids"]

    with open(os.path.join(results_path, 'last.pkl'), "rb") as file:
        samples, _ = pickle.load(file)

    sky_model = ju.SkyModel(config_path)
    sky = sky_model.create_sky_model()
    sky_dict = sky_model.sky_model_to_dict()
    response_dict = ju.build_erosita_response_from_config(config_path)

    mask_adj = jax.linear_transpose(response_dict['mask'],
                                    np.zeros((len(tm_ids),) + sky.target.shape))
    response_dict['mask_adj'] = mask_adj
    mask_func = response_dict['mask']

    gt_dict = {}
    masked_data = ju.load_masked_data_from_config(config_path)
    pos = ju.load_mock_position_from_config(config_path)
    for key, comp in sky_dict.items():
        gt_dict[key] = comp(pos)

    masked_data = jax.tree_map(lambda x: np.array(x, dtype=np.float64),
                            masked_data)

    for key, op in sky_dict.items():
        uwrs, exp_mask = ju.calculate_uwr(samples.samples, op, gt_dict[key], response_dict,
                                        abs=False, exposure_mask=mask_func, log=True)
        bbox_info = [(7, 4), 7, 24]
        plotting_kwargs = {'vmin': -5, 'vmax': 5, 'cmap': 'RdYlBu_r'}
        plot(uwrs,
             pixel_measure=28,
             fs=8,
                        title=['0.2-1.0 keV',
                               '1.0-2.0 keV',
                               '2.0-4.5 keV'],
                        logscale=False,
                        colorbar=True,
                        n_rows=1,
                        output_file=join(output_dir,
                        f'uwr_{key}.png'),
             **plotting_kwargs)
        op = jax.vmap(op)
        real_samples = op(samples.samples)
        real_mean = jnp.mean(real_samples, axis=0)
        residual = real_mean - gt_dict[key]
        rel_residual = np.abs(residual) / gt_dict[key]
        plot(rel_residual,
             pixel_measure=28,
             fs=8,
                        title=['0.2-1.0 keV',
                               '1.0-2.0 keV',
                               '2.0-4.5 keV'],
                        logscale=False,
                        colorbar=True,
                        n_rows=1,
                        output_file=join(output_dir,
                        f'relresidual_{key}.png'),
             **plotting_kwargs)
