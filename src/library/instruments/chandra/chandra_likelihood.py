import nifty8.re as jft

from .chandra_response import build_chandra_response_from_config
from .chandra_data import create_chandra_data_from_config
from ...data import load_masked_data_from_config

def generate_chandra_likelihood_from_config(config_file_path):
    """ Creates the Chandra Poissonian log-likelihood given the path to the
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
    response_dict = build_chandra_response_from_config(config_file_path)

    # Create data files
    create_chandra_data_from_config(config_file_path, response_dict)
    # Load data files
    masked_data = load_masked_data_from_config(config_file_path)
    response_func = response_dict['R']
    return jft.Poissonian(masked_data).amend(response_func)