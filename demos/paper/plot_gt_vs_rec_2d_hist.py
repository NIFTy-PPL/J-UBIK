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
from plot_eROSITA_image import plot_2d_gt_vs_rec_histogram


# Script for plotting the data, position and reconstruction images
if __name__ == "__main__":
    results_path = "results/Mock-17092024-001M"
    config_name = "eROSITA_demo.yaml"
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
    plotting_kwargs =  {'bins': 600, 'x_label': '$s_{gt}$', 'y_label': 'a', 'dpi': 400, 'title': 'standardised error',
                        'x_lim': (1e-11,1e-6), 'y_lim': (8e-2, 1e2)}
    plot_2d_gt_vs_rec_histogram(samples=samples, operator_dict=sky_dict, diagnostics_path=output_dir,
                                response_dict=response_dict, type='sampled',relative=True, response=False,
                                reference_dict=gt_dict, plot_kwargs=plotting_kwargs, base_filename='gtvsrec',
                                fs=18, alpha=0.5, max_counts=1e2)