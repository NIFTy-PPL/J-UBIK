import os
import pickle
from jax import random
from jax import numpy as jnp
from astropy.io import fits

import nifty8.re as jft
import xubik0 as xu

from .erosita_observation import ErositaObservation
from .sky_models import create_sky_model
from .utils import get_config

from typing import NamedTuple


class Domain(NamedTuple):
    """Mimicing NIFTy Domain.

    Paramters:
    ----------
    shape: tuple
    distances: tuple
    """

    shape: tuple
    distances: tuple


# GENERIC
def load_masked_data_from_pickle(file_path, mask_func):
    """ Load data from pickle file as a data-dictionary and create a jft.Vector
    of masked data out of it

    Parameters
    ----------
    file_path : string
        Path to data file (.pkl)
    mask_func : Callable
        Mask function, which takes a three dimensional array and makes a jft.Vector
        out of it containing a dictionary of masked arrays
    Returns
    -------
    masked_data : jft.Vector
        Dictionary of masked data
    """
    with open(file_path, "rb") as f:
        data_dict = pickle.load(f)
    masked_data_dict = mask_func(jnp.stack(list(data_dict.values())))
    if set(data_dict.keys()) != set(masked_data_dict):
        raise ValueError('The loaded data dictionary and the given mask function '
                         'are not compatible!')
    return masked_data_dict


def save_data_dict_to_pickle(data_dict, file_path):
    """ Save data dictionary to pickle file

    Parameters
    ----------
    data_dict : dict
        Data dictionary, which is saved.
    file_path : string
        Path to data file (.pkl)
    Returns
    -------
    """
    with open(file_path, "wb") as file:
        pickle.dump(data_dict, file)


# MOCK
def generate_mock_sky_from_prior_dict(npix, padding_ratio, fov, priors, seed=42,
                                      default_point_source_prior=None):
    """ Generates a mock sky position for the given grid and prior information

    Parameters
    ----------
    npix : int
        Number of pixels
    padding_ratio : float
        Ratio between padded and actual space
    fov : int
        FOV of the telescope
    priors : dict
        Dictionary of prior information containing the hyperparameters for the
        used models etc. correlated field etc. (see sky_models.py)
    seed : int
        Random seed for mock sky generation
    default_point_source_prior: dict
        Default values for the point sources for the mock sky generation if a reconstruction
        with only diffuse is used.

    Returns
    -------
    mock_sky_position: jft.Vector
        Random position of the latent parameters
    """
    if priors['point_sources'] is None and default_point_source_prior is None:
        raise ValueError('Point source information is needed for the generation of a mock sky.')
    if priors['point_sources'] is None:
        priors['point_sources'] = default_point_source_prior
    sky_dict = create_sky_model(npix, padding_ratio, fov, priors)
    key = random.PRNGKey(seed)
    key, subkey = random.split(key)
    return jft.random_like(subkey, sky_dict['sky'].domain)


# eROSITA
def load_erosita_masked_data(file_info, tel_info, mask_func):
    """ Load eROSITA data from file, mask it and generate a jft.Vector out of it

    Parameters
    ----------
    file_info : dict
        Dictionary of file paths
    tel_info : dict
        Dictionary of telescope information
    mask_func : Callable
        Mask function, which takes a three dimensional array and makes a jft.Vector
        out of it containing a dictionary of masked arrays
    priors : dict
        Dictionary of prior information
    seed : int
        Random seed for mock sky generation
    default_point_source_prior: dict
        Default values for the point sources for the mock sky generation if a reconstruction
        with only diffuse is used.
    Returns
    -------
    masked_data_vector: jft.Vector
        Dictionary of masked data arrays
    """
    data_list = []
    for tm_id in tel_info['tm_ids']:
        output_filename = f'{tm_id}_' + file_info['output']
        data = jnp.array(fits.open(os.path.join(file_info['obs_path'], output_filename))[0].data, dtype=int)
        data_list.append(data)
    masked_data_vector = mask_func(jnp.stack(data_list))
    return masked_data_vector


def generate_erosita_data_from_config(config_file_path, response_func, output_path=None):
    """ Generated mock data for the information given in the config_file and a response func

    Parameters
    ----------
    config_file_path : string
        Path to config
    response_func : Callable
        Response function, which takes an array and makes a jft.Vector
        out of it containing a dictionary of mock data
    output_path: string
        If output_path is given the generated data is saved to a pickle file
    Returns
    -------
    masked_mock_data: jft.Vector
        Dictionary of masked mock data arrays
    """
    cfg = get_config(config_file_path)

    tel_info = cfg['telescope']
    grid_info = cfg['grid']
    priors = cfg['priors']

    mock_sky_position = generate_mock_sky_from_prior_dict(grid_info['npix'],
                                                          grid_info['padding_ratio'],
                                                          tel_info['fov'],
                                                          priors,
                                                          cfg['seed'],
                                                          cfg['point_source_defaults'])
    sky = xu.create_sky_model(grid_info['npix'], grid_info['padding_ratio'],
                              tel_info['fov'], priors)['sky']
    masked_mock_data = response_func(sky(mock_sky_position))
    if output_path is not None:
        xu.create_output_directory(output_path)
        save_data_dict_to_pickle(masked_mock_data.tree,
                                 os.path.join(output_path, 'mock_data_dict.pkl'))
    return jft.Vector({key: val.astype(int) for key, val in masked_mock_data.tree.items()})

