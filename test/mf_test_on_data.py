import pickle
import os
import numpy as np
from jax import random
from matplotlib.colors import LogNorm, LinearSegmentedColormap, Normalize

import nifty8.re as jft
import jubik0 as ju

path_to_mock_rec = '../presentation_results'
config_path = 'config_test_sky_model.yaml'

data_arr = []
for color in ['blue', 'red', 'green']:
    folder_dir = os.path.join(path_to_mock_rec, f'LMC_{color}/diagnostics/')
    tm=1
    tm_dir = os.path.join(folder_dir, f'tm{tm}')
    with open(os.path.join(tm_dir, f'tm{tm}_data.pkl'), "rb") as f:
        data = pickle.load(f)
    data_arr.append(data.val)

data_arr = np.stack(data_arr, axis=0)

# Load sky model
cfg = ju.get_config(config_path)
sky_model = ju.SkyModel(config_path)
sky = sky_model.create_sky_model()

file_info = cfg['files']
ju.save_config(cfg, os.path.basename(config_path), file_info['res_dir'])
response_func = ju.build_erosita_response_from_config(config_path)['R']

log_likelihood = jft.Poissonian(data_arr).amend(response_func).amend(sky)

minimization_config = cfg['minimization']
key = random.PRNGKey(cfg['seed'])
key, subkey = random.split(key)
pos_init = 0.1 * jft.Vector(jft.random_like(subkey, sky.domain))

kl_solver_kwargs = minimization_config.pop('kl_kwargs')
kl_solver_kwargs['minimize_kwargs']['absdelta'] *= cfg['grid'][
    'sdim']  # FIXME: Replace by domain information

# Plot
plot = lambda s, x: ju.plot_sample_and_stats(file_info["res_dir"],
                                             sky,
                                             s,
                                             iteration=x.nit)

samples, state = jft.optimize_kl(log_likelihood,
                                 pos_init,
                                 key=key,
                                 kl_kwargs=kl_solver_kwargs,
                                 # callback=plot,
                                 odir=file_info["res_dir"],
                                 **minimization_config
                                 )




