import nifty8.re as jft

from .data import load_masked_data_from_config
from .instruments.erosita.erosita_data import create_erosita_data_from_config
from .. import build_erosita_response_from_config


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


def get_n_constrained_dof(likelihood: jft.Likelihood) -> int:
    """
    Extacts the number of constrained degrees of freedom (DOF)
    based on the likelihood.

    Parameters
    ----------
    likelihood : jft.Likelihood
        The likelihood object which contains information about
       the model and data.

    Returns
    -------
    int
        The number of constrained degrees of freedom, which is the
        minimum of the model degrees of freedom and the data
        degrees of freedom.
    """

    n_dof_data = jft.size(likelihood.left_sqrt_metric_tangents_shape)
    n_dof_model = jft.size(likelihood.domain)
    return min(n_dof_model, n_dof_data)
