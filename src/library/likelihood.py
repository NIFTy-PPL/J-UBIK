import nifty8.re as jft

from .data import create_data_from_config, load_masked_data_from_config
from .response import build_erosita_response_from_config


def generate_erosita_likelihood_from_config(config_file_path):
    """ Creates the eROSITA Poissonian log-likelihood given the path to the config file.

    Parameters
    ----------
    config_file_path : string
        Path to config file
    Returns
    -------
    poissonian: jft.Likelihood
        Poissoninan likelihood for the eROSITA data and response, specified in the config
    """

    # load config
    response_dict = build_erosita_response_from_config(config_file_path)

    # Create data files
    create_data_from_config(config_file_path, response_dict)
    # Load data files
    masked_data = load_masked_data_from_config(config_file_path)
    response_func = response_dict['R']
    return jft.Poissonian(masked_data).amend(response_func)


def model_wrap(model, target_domain=None):
    if target_domain is None:
        def wrapper(x):
            out = model(x)
            for x, val in x.items():
                out[x] = val
            return out
    else:
        def wrapper(x):
            out = model(x)
            for key in target_domain.keys():
                out[key] = x[key]
            return out
    return wrapper


def connect_likelihood_to_model(
    likelihood: jft.Likelihood,
    model: jft.Model
) -> jft.Likelihood:
    '''Connect the likelihood and model, this is necessery when some models are
    inside the likelihood.
    In this case the keys necessery are passed up the chain, such that white
    priors are passed to the respective keys of the likelihood.
    '''

    ldom = likelihood.domain.tree
    tdom = {t: ldom[t] for t in ldom.keys() if t not in model.target.keys()}
    mdom = tdom | model.domain

    model_wrapper = model_wrap(model, tdom)
    model = jft.Model(
        lambda x: jft.Vector(model_wrapper(x)),
        domain=jft.Vector(mdom)
    )

    return likelihood.amend(model, domain=model.domain)


def build_gaussian_likelihood(
    data,
    std
):
    if not isinstance(std, float):
        assert data.shape == std.shape

    var = std**2

    return jft.Gaussian(
        data=data,
        noise_cov_inv=lambda x: x/var,
        noise_std_inv=lambda x: x/std
    )

