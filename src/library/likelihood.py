import os

import nifty8.re as jft

from .data import (load_erosita_masked_data, generate_erosita_data_from_config,
                   load_masked_data_from_pickle)
from .jwst_data import (load_jwst_data)
from .response import build_erosita_response_from_config
from .jwst_response import build_jwst_response, build_mask_operator
from .utils import get_config


def generate_erosita_likelihood_from_config(config_file_path):
    """ Creates the eROSITA Poissonian likelihood given the path to the config file.

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

    response_dict = build_erosita_response_from_config(config_file_path)

    response_func = response_dict['R']
    mask_func = response_dict['mask']

    if cfg['mock']:
        masked_data = generate_erosita_data_from_config(
            config_file_path, response_func, file_info['res_dir'])
    elif cfg['load_mock_data']:
        masked_data = load_masked_data_from_pickle(
            os.path.join(file_info['res_dir'], 'mock_data_dict.pkl'), mask_func)
    else:
        masked_data = load_erosita_masked_data(file_info, tel_info, mask_func)
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

    ldom = likelihood.domain.tree
    tdom = {t: ldom[t] for t in ldom.keys() if t not in model.target.keys()}
    mdom = {t: model.domain[t] for t in model.domain.keys()}
    mdom.update(tdom)

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


def generate_jwst_likelihood_from_config(
    sky_dict: dict,
    config_file_path: str
):
    """Creates the JWST Gaussian likelihood from the path to the config file.

    Parameters
    ----------
    sky_dict: dict
        Dictionary of sky component models, including the target Domain

    config_file_path : string
        Path to config file

    Returns
    -------
    likelihood: jft.Likelihood
        Gaussian likelihood for the JWST data with applied response and model

    data: dict
        dictionary containing data dicts, containing all necessery information
        to build the response operator, which is also kept for book-keeping.

    """
    from functools import reduce

    cfg = get_config(config_file_path)

    SKY_KEY = 'sky'

    data_dict = {}
    likelihoods = []
    for key, lh_cfg in cfg['files']['data'].items():
        data_dict[key] = load_jwst_data(lh_cfg)

        R, response_no_psf = build_jwst_response(
            domain_key=SKY_KEY,
            domain=sky_dict['target'],
            data_pixel_size=data_dict[key].pixel_size,
            likelihood_key=key,
            likelihood_config=lh_cfg,
            telescope_cfg=cfg.get('telescope', None),
        )

        # Load mask and Mask operator
        Mask_d = build_mask_operator(R.target, ~data_dict[key].mask)

        likelihood = build_gaussian_likelihood(
            data=data_dict[key].data_2d[~(data_dict[key].mask)],
            std=data_dict[key].noise_2d[~(data_dict[key].mask)]
        )

        likelihood = likelihood.amend(Mask_d)
        likelihood = likelihood.amend(R, domain=R.domain)

        data_dict[key].response = R
        data_dict[key].response_no_psf = response_no_psf
        data_dict[key].mask2data = Mask_d
        likelihoods.append(likelihood)

    likelihood = reduce(lambda x, y: x + y, likelihoods)

    # Connect Model to likelihood, first create model
    likelihood = connect_likelihood_to_model(
        likelihood,
        jft.Model(jft.wrap_left(sky_dict['sky'], SKY_KEY),
                  domain=sky_dict['sky'].domain)

    )

    return likelihood, data_dict
