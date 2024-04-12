import os
import pickle
from os.path import join, splitext

import numpy as np
from jax import random, tree_map
from jax import numpy as jnp
from astropy.io import fits

import nifty8.re as jft

from .erosita_observation import ErositaObservation
from .messages import log_file_exists
from .sky_models import SkyModel
from .utils import get_config, create_output_directory

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


def generate_mock_xi_from_prior_dict(sdim, edim, s_padding_ratio, e_padding_ratio,
                                     fov, energy_range, priors, subkey,
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
    sky = SkyModel().create_sky_model(sdim=sdim,
                                    edim=edim,
                                    s_padding_ratio=s_padding_ratio,
                                    e_padding_ratio=e_padding_ratio,
                                    fov=fov,
                                    energy_range=energy_range,
                                    priors=priors)

    return jft.random_like(subkey, sky.domain)


# eROSITA
def load_erosita_masked_data(file_info, tel_info, grid_info, mask_func):
    """ Load eROSITA data from file, mask it and generate a jft.Vector out of it

    Parameters
    ----------
    file_info : dict
        Dictionary of file paths
    tel_info : dict
        Dictionary of telescope information
    grid_info : dict
        Dictionary with grid information
    mask_func : Callable
        Mask function, which takes a 3D array and makes a jft.Vector
        out of it containing a dictionary of masked arrays
    Returns
    -------
    masked_data_vector: jft.Vector
        Dictionary of masked data arrays
    """
    # load energies
    e_min = grid_info['energy_bin']['e_min']
    e_max = grid_info['energy_bin']['e_max']

    if not isinstance(e_min, list):
        raise TypeError("e_min must be a list!")

    if not isinstance(e_max, list):
        raise TypeError("e_max must be a list!")

    if len(e_min) != len(e_max):
        raise ValueError("e_min and e_max must have the same length!")

    data_list = []
    for tm_id in tel_info['tm_ids']:
        output_filenames = f'tm{tm_id}_' + file_info['output']
        output_filenames = [f"{output_filenames.split('.')[0]}_emin{e}_emax{E}.fits" for e, E in
                            zip(e_min, e_max)]
        data = []
        for output_filename in output_filenames:
            data.append(fits.open(join(file_info['obs_path'], "processed", output_filename))[0].data)
        data = jnp.stack(jnp.array(data, dtype=int))
        data_list.append(data)
    data = jnp.stack(jnp.array(data_list, dtype=int))
    masked_data_vector = mask_func(data)
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
    sdim = grid_info['sdim']
    e_min = grid_info['energy_bin']['e_min']
    e_max = grid_info['energy_bin']['e_max']

    if not isinstance(e_min, list):
        raise TypeError("e_min must be a list!")

    if not isinstance(e_max, list):
        raise TypeError("e_max must be a list!")

    if len(e_max) != len(e_max):
        raise ValueError("e_min and e_max must have the same length!")

    tm_ids = tel_info["tm_ids"]
    fov = tel_info['fov']
    detmap = tel_info['detmap']

    rebin = int(np.floor(20 * fov // sdim))  # FIXME: USE DISTANCES!

    processed_obs_path = create_output_directory(join(obs_path, 'processed'))

    for tm_id in tm_ids:
        # TODO: implement the following by changing the eSASS interface ErositaObservation
        # tm_processed_path = create_output_directory(join(processed_obs_path, f'tm{tm_id}'))
        output_filenames = f'tm{tm_id}_' + file_info['output']
        exposure_filenames = f'tm{tm_id}_' + file_info['exposure']
        output_filenames = [f"{output_filenames.split('.')[0]}_emin{e}_emax{E}.fits"
                            for e, E in zip(e_min, e_max)]
        exposure_filenames = [f"{exposure_filenames.split('.')[0]}_emin{e}_emax{E}.fits"
                              for e, E in zip(e_min, e_max)]

        for e, output_filename in enumerate(output_filenames):
            observation_instance = ErositaObservation(input_filenames,
                                                         join("processed", output_filename),
                                                         obs_path,
                                                         esass_image=config_dict['esass_image'])
            if not os.path.exists(join(processed_obs_path, output_filename)):
                _ = observation_instance.get_data(emin=e_min[e],
                                                  emax=e_max[e],
                                                  image=True,
                                                  rebin=rebin,
                                                  size=sdim,
                                                  pattern=tel_info['pattern'],
                                                  telid=tm_id)  # FIXME: exchange rebin by fov? 80 = 4arcsec
            else:
                log_file_exists(join(processed_obs_path, output_filename))

            observation_instance = ErositaObservation(output_filename, output_filename, processed_obs_path,
                                                         esass_image=config_dict['esass_image'])

            # Exposure
            if not os.path.exists(join(processed_obs_path, exposure_filenames[e])):
                observation_instance.get_exposure_maps(output_filename, e_min[e], e_max[e],
                                                       withsinglemaps=True,
                                                       singlemaps=[exposure_filenames[e]],
                                                       withdetmaps=detmap,
                                                       badpix_correction=tel_info['badpix_correction'])

            else:
                log_file_exists(join(processed_obs_path, exposure_filenames[e]))

            # Plotting
            plot_info = config_dict['plotting']
            if plot_info['enabled']:
                observation_instance.plot_fits_data(output_filename,
                                                    f'{splitext(output_filename)[0]}.png',
                                                    slice=plot_info['slice'],
                                                    dpi=plot_info['dpi'])
                observation_instance.plot_fits_data(exposure_filenames[e],
                                                    f'{splitext(exposure_filenames[e])[0]}.png',
                                                    slice=plot_info['slice'],
                                                    dpi=plot_info['dpi'])

def generate_mock_erosita_data_from_config(config_file_path, response_func, output_path=None):
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

    e_min = cfg['grid']['energy_bin']['e_min']
    e_max = cfg['grid']['energy_bin']['e_max']
    energy_range = np.array(e_max) - np.array(e_min)

    key, subkey = random.split(key)
    mock_sky_position = generate_mock_xi_from_prior_dict(grid_info['sdim'],
                                                         grid_info['edim'],
                                                         grid_info['s_padding_ratio'],
                                                         grid_info['e_padding_ratio'],
                                                         tel_info['fov'],
                                                         energy_range,
                                                         priors=priors,
                                                         subkey=subkey,
                                                         default_point_source_prior=cfg['point_source_defaults'])
    sky_model = SkyModel(config_file_path)
    sky = sky_model.create_sky_model()
    sky_comps = sky_model.sky_model_to_dict()
    key, subkey = random.split(key)
    if output_path is not None:
        create_output_directory(output_path)
    for key, sky_comp in sky_comps.items():
        masked_mock_data = response_func(sky_comp(mock_sky_position))
        masked_mock_data = tree_map(lambda x: random.poisson(subkey, x), masked_mock_data.tree)
        masked_mock_data = jft.Vector(masked_mock_data)
        if output_path is not None:
            save_dict_to_pickle(response_func(sky_comp(mock_sky_position)),
                                join(output_path, f'{key}_mock_data.pkl'))
            save_dict_to_pickle(sky_comp(mock_sky_position),
                                join(output_path, f'{key}_gt.pkl'))
    return jft.Vector({key: val.astype(int) for key, val in masked_mock_data.tree.items()})

