from nifty8 import re as jft

from ...data import load_masked_data_from_config
from .erosita_data import create_erosita_data_from_config
from .erosita_response import build_erosita_response_from_config


def generate_erosita_likelihood_from_config(config_file_path):
    """ Creates the eROSITA Poissonian log-likelihood given the path to the
    config file.

    Parameters
    ----------
    config_file_path : string
        Path to config file
    Returns
    -------
    poissonian: jft.Likelihood
        Poissoninan likelihood for the eROSITA data and response, specified
        in the config.
    """

    # load config
    response_dict = build_erosita_response_from_config(config_file_path)

    # Create data files
    create_erosita_data_from_config(config_file_path, response_dict)

    # Load data files
    masked_data = load_masked_data_from_config(config_file_path)
    response_func = response_dict['R']
    return jft.Poissonian(masked_data).amend(response_func)
