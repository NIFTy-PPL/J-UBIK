"""
eROSITA Priors Plotting Script
------------------------------

This script generates and plots samples from the eROSITA priors using the
provided configuration file and priors directory.
The script creates a configurable number of samples, optionally plots the
signal response, and adjusts plotting aesthetics according to
the provided parameters.

Parameters:
-----------
- n_samples (int): Number of samples to generate from the priors.
- seed (int): Seed for the random number generator to ensure reproducibility.
- path_to_config (str): Path to the configuration file defining the setup.
- priors_directory (str): Directory containing the priors to be sampled.
- plot_signal_response (bool): Flag to decide whether to plot the signal
response.
- kwgs (dict): Additional keyword arguments for plot formatting.

"""

import jubik0 as ju
from jax import random

if __name__ == '__main__':
    n_samples = 6
    seed = 96

    key = random.PRNGKey(seed)
    path_to_config = 'configs/eROSITA_config.yaml'
    priors_directory = 'jubik_priors_mf/'

    plot_signal_response = True # decides whether to plot signal response

    kwgs = {'n_cols': 3, 'n_rows': 1}
    ju.plot_erosita_priors(key,
                           n_samples,
                           path_to_config,
                           priors_directory,
                           plot_signal_response,
                           plotting_kwargs=kwgs,
                           adjust_figsize=True)
