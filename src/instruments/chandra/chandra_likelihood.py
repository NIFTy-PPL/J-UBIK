# SPDX-License-Identifier: BSD-2-Clause
# Authors: Vincent Eberle, Matteo Guardiani, Margret Westerkamp

# Copyright(C) 2024 Max-Planck-Society

# %%

import nifty8.re as jft

from .chandra_response import build_chandra_response_from_config
from .chandra_data import create_chandra_data_from_config
from ...data import load_masked_data_from_config


def generate_chandra_likelihood_from_config(config):
    """ Creates the Chandra Poissonian log-likelihood from a config dictionary.

    Parameters
    ----------
    config : dict
        Dictionary containing the configuration parameters.
    Returns
    -------
    Poissonian: jft.Likelihood
        Poissonian likelihood for the eROSITA data and response, specified
        in the config.
    """

    # load config
    response_dict = build_chandra_response_from_config(config)

    # Create data files
    create_chandra_data_from_config(config, response_dict)
    # Load data files
    masked_data = load_masked_data_from_config(config)
    response_func = response_dict['R']
    return jft.Poissonian(masked_data).amend(response_func)
