import os

import nifty8.re as jft

from .data import load_erosita_masked_data, generate_mock_erosita_data_from_config, \
    load_masked_data_from_pickle
from .response import build_erosita_response_from_config
from .utils import get_config


def generate_erosita_likelihood_from_config(config_file_path):
    """ Creates the eROSITA Poissonian log-likelihood given the path to the config file.

    Parameters
    ----------
    config_file_path : string
        Path to config file
    Returns
    -------
    masked_data_vector: jft.Likelihood
        Poissoninan likelihood for the eROSITA data and response, specified in the config
    """

    # load config
    cfg = get_config(config_file_path)
    file_info = cfg['files']
    tel_info = cfg['telescope']
    grid_info = cfg['grid']

    response_dict = build_erosita_response_from_config(config_file_path)

    response_func = response_dict['R']
    mask_func = response_dict['mask']

    if cfg['mock']:
        masked_data = generate_mock_erosita_data_from_config(config_file_path, response_func,
                                                        file_info['res_dir'])
    elif cfg['load_mock_data']:
        masked_data = load_masked_data_from_pickle(os.path.join(file_info['res_dir'],
                                                   'mock_data_dict.pkl'), mask_func)
    else:
        masked_data = load_erosita_masked_data(file_info, tel_info, grid_info, mask_func)
    return jft.Poissonian(masked_data).amend(response_func)
