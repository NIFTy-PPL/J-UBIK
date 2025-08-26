import matplotlib.pyplot as plt
import numpy as np
import pickle
from jax import linear_transpose, vmap, random
import jax.numpy as jnp

import jubik0 as ju
from os.path import join
import astropy.io.fits as fits
import nifty.re as jft
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from joss_paper_plotting import plot, plot_rgb


# Script for plotting the data, position and reconstruction images
if __name__ == "__main__":
    config_path = "paper/prior_config.yaml"
    output_dir = ju.create_output_directory('paper')
    config_dict = ju.get_config(config_path)

    sky_model = ju.SkyModel(config_dict)
    sky = sky_model.create_sky_model()
    sky_dict = sky_model.sky_model_to_dict()

    key = random.PRNGKey(81)
    key, subkey = random.split(key)

    pos = jft.Vector(jft.random_like(subkey, sky.domain))

    real_pos = []
    titles = []
    for key, op in sky_dict.items():
        real_pos.append(100*op(pos))
        titles.append(key)
    real_pos = np.vstack(real_pos)

    ju.save_to_pickle(pos, 'paper/pos.pkl')

    bbox_info = [(7, 4), 28, 96,  'black']
    plot(real_pos,
         pixel_measure=112,
         fs=8,
         title=titles,
         logscale=True,
         colorbar=True,
         common_colorbar=True,
         n_rows=1,
         vmin=5e-7,
         vmax=1e-5,
         bbox_info=bbox_info,
         output_file=join(output_dir,
         f'simulated_sky.png'))

