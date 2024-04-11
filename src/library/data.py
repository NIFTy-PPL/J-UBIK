import os
import pickle
from os.path import join

import numpy as np
from jax import random, tree_map
from jax import numpy as jnp
from astropy.io import fits

import nifty8.re as jft
import jubik0 as ju

from .erosita_observation import ErositaObservation
from .messages import log_file_exists
from .sky_models import SkyModel
from .utils import get_config

from typing import NamedTuple


class Domain(NamedTuple):
    """Mimicking NIFTy Domain.

    Paramters:
    ----------
    shape: tuple
    distances: tuple
    """

    shape: tuple
    distances: tuple


# GENERIC
def load_masked_data_from_pickle(file_path):
    """ Load data from pickle file as a data-dictionary and create a jft.Vector
    of masked data out of it

    Parameters
    ----------
    file_path : string
        Path to data file (.pkl)
    mask_func : Callable
        Mask function, which takes a 3D array and makes a jft.Vector
        out of it containing a dictionary of masked arrays
    Returns
    -------
    masked_data : jft.Vector
        Dictionary of masked data
    """
    with open(file_path, "rb") as f:
        data_dict = pickle.load(f)
    return data_dict


def save_dict_to_pickle(dictionary, file_path):
    """ Save data dictionary to pickle file

    Parameters
    ----------
    dictionary : dict
        Data dictionary, which is saved.
    file_path : string
        Path to data file (.pkl)
    Returns
    -------
    """
    with open(file_path, "wb") as file:
        pickle.dump(dictionary, file)


# MOCK
def generate_mock_sky_from_prior_dict(npix, padding_ratio, fov, priors, key,
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
    key : jax.random.PRNGKey
        Random key for mock sky generation
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
    sky = SkyModel.create_sky_model(sdim=npix, padding_ratio=padding_ratio, fov=fov,
                                         priors=priors)
    key = random.PRNGKey(seed)
    key, subkey = random.split(key)
    return jft.random_like(subkey, sky.domain)


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
        Mask function, which takes a 3D array and makes a jft.Vector
        out of it containing a dictionary of masked arrays
    Returns
    -------
    masked_data_vector: jft.Vector
        Dictionary of masked data arrays
    """
    data_list = []
    for tm_id in tel_info['tm_ids']:
        output_filename = f'{tm_id}_' + file_info['output']
        data = jnp.array(fits.open(join(file_info['obs_path'], output_filename))[0].data, dtype=int)
        data_list.append(data)
    masked_data_vector = mask_func(jnp.stack(data_list))
    return masked_data_vector


def create_erosita_data_from_config_dict(config_dict):
    """ Create eROSITA data from config dictionary
        (calls the eSASS interface)
    """
    tel_info = config_dict["telescope"]
    file_info = config_dict["files"]
    grid_info = config_dict['grid']

    input_filenames = file_info["input"]
    obs_path = file_info["obs_path"]
    e_min = grid_info['energy_bin']['e_min']
    e_max = grid_info['energy_bin']['e_max']
    npix = grid_info['npix']

    tm_ids = tel_info["tm_ids"]
    fov = tel_info['fov']
    detmap = tel_info['detmap']

    rebin = int(np.floor(20 * fov // npix))  # FIXME: USE DISTANCES!

    for tm_id in tm_ids:
        output_filename = f'{tm_id}_' + file_info['output']
        exposure_filename = f'{tm_id}_' + file_info['exposure']
        observation_instance = ju.ErositaObservation(input_filenames, output_filename, obs_path,
                                                     esass_image=config_dict['esass_image'])
        if not os.path.exists(join(obs_path, output_filename)):
            _ = observation_instance.get_data(emin=e_min,
                                              emax=e_max,
                                              image=True,
                                              rebin=rebin,
                                              size=npix,
                                              pattern=tel_info['pattern'],
                                              telid=tm_id)  # FIXME: exchange rebin by fov? 80 = 4arcsec
        else:
            log_file_exists(join(obs_path, output_filename))

        observation_instance = ju.ErositaObservation(output_filename, output_filename, obs_path,
                                                     esass_image=config_dict['esass_image'])

        # Exposure
        if not os.path.exists(join(obs_path, exposure_filename)):
            observation_instance.get_exposure_maps(output_filename, e_min, e_max,
                                                   withsinglemaps=True,
                                                   singlemaps=[exposure_filename],
                                                   withdetmaps=detmap,
                                                   badpix_correction=tel_info['badpix_correction'])

        else:
            log_file_exists(join(obs_path, output_filename))
            
        # Plotting
        plot_info = config_dict['plotting']
        if plot_info['enabled']:
            observation_instance.plot_fits_data(output_filename,
                                                os.path.splitext(output_filename)[0],
                                                slice=plot_info['slice'],
                                                dpi=plot_info['dpi'])
            observation_instance.plot_fits_data(exposure_filename,
                                                f'{os.path.splitext(exposure_filename)[0]}.png',
                                                slice=plot_info['slice'],
                                                dpi=plot_info['dpi'])


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
    key = random.PRNGKey(cfg['seed'])

    tel_info = cfg['telescope']
    grid_info = cfg['grid']
    priors = cfg['priors']

    key, subkey = random.split(key)
    mock_sky_position = generate_mock_sky_from_prior_dict(grid_info['npix'],
                                                          grid_info['padding_ratio'],
                                                          tel_info['fov'],
                                                          priors,
                                                          subkey,
                                                          cfg['point_source_defaults'])
    sky_comps = ju.create_sky_model(grid_info['npix'], grid_info['padding_ratio'],
                              tel_info['fov'], priors)
    masked_mock_data = response_func(sky_comps['sky'](mock_sky_position))
    key, subkey = random.split(key)
    masked_mock_data = tree_map(lambda x: random.poisson(subkey, x), masked_mock_data.tree)
    masked_mock_data = jft.Vector(masked_mock_data)
    if output_path is not None:
        ju.create_output_directory(output_path)
        save_dict_to_pickle(masked_mock_data.tree,
                            join(output_path, 'mock_data_dict.pkl'))
        for key, sky_comp in sky_comps.items():
            save_dict_to_pickle(sky_comp(mock_sky_position),
                                join(output_path, f'{key}_gt.pkl'))
    return jft.Vector({key: val.astype(int) for key, val in masked_mock_data.tree.items()})

