import os
import pickle

import numpy as np
from jax import random
from jax import numpy as jnp
from astropy.io import fits

import nifty8.re as jft
import jubik0 as ju

from .erosita_observation import ErositaObservation
from .sky_models import create_sky_model
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
        Mask function, which takes a three dimensional array and makes a jft.Vector
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


def create_erosita_data_from_config_dict(config_dict):
    """ Create eROSITA data from config dictionary
        (calls the eSASS interface)
    """
    tel_info = config_dict["telescope"]
    file_info = config_dict["files"]
    obs_path = config_dict["obs_path"]
    input_filenames = config_dict["input"]

    grid_info = config_dict['grid']
    e_min = grid_info['energy_bin']['e_min']
    e_max = grid_info['energy_bin']['e_max']
    npix = grid_info['npix']

    tm_ids = tel_info["tm_ids"]
    fov = tel_info['fov']
    detmap = tel_info['detmap']

    rebin = np.floor(20 * fov // npix)  # FIXME: USE DISTANCES!

    log = 'Output file {} already exists and is not regenerated. '\
          'If the observation parameters have changed please'\
           ' delete or rename the current output file.'

    for tm_id in tm_ids:
        output_filename = f'{tm_id}_' + file_info['output']
        exposure_filename = f'{tm_id}_' + file_info['exposure']
        observation_instance = ju.ErositaObservation(input_filenames, output_filename, obs_path)
        if not os.path.exists(os.path.join(obs_path, output_filename)):
            _ = observation_instance.get_data(emin=e_min,
                                              emax=e_max,
                                              image=True,
                                              rebin=rebin,
                                              size=npix,
                                              pattern=tel_info['pattern'],
                                              telid=tm_id)  # FIXME: exchange rebin by fov? 80 = 4arcsec
        else:
            print(log.format(os.path.join(obs_path, output_filename)))

        observation_instance = ju.ErositaObservation(output_filename, output_filename, obs_path)

        # Exposure
        if not os.path.exists(os.path.join(obs_path, exposure_filename)):
            observation_instance.get_exposure_maps(output_filename, e_min, e_max,
                                                   mergedmaps=exposure_filename,
                                                   withdetmaps=detmap)

        else:
            print(log.format(os.path.join(obs_path, output_filename)))

        # Exposure
        if not os.path.exists(os.path.join(obs_path, exposure_filename)):
            observation_instance.get_exposure_maps(output_filename, e_min, e_max,
                                                   mergedmaps=exposure_filename,
                                                   withdetmaps=detmap)

        else:
            print(log.format(os.path.join(obs_path, output_filename)))

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

    tel_info = cfg['telescope']
    grid_info = cfg['grid']
    priors = cfg['priors']

    mock_sky_position = generate_mock_sky_from_prior_dict(grid_info['npix'],
                                                          grid_info['padding_ratio'],
                                                          tel_info['fov'],
                                                          priors,
                                                          cfg['seed'],
                                                          cfg['point_source_defaults'])
    sky_comps = ju.create_sky_model(grid_info['npix'], grid_info['padding_ratio'],
                              tel_info['fov'], priors)
    masked_mock_data = response_func(sky_comps['sky'](mock_sky_position))
    if output_path is not None:
        ju.create_output_directory(output_path)
        save_dict_to_pickle(masked_mock_data.tree,
                            os.path.join(output_path, 'mock_data_dict.pkl'))
        for key, sky_comp in sky_comps.items():
            save_dict_to_pickle(sky_comp(mock_sky_position),
                                os.path.join(output_path, f'{key}_gt.pkl'))
    return jft.Vector({key: val.astype(int) for key, val in masked_mock_data.tree.items()})

