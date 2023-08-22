import os
import pickle
from jax import random
import numpy as np

import nifty8.re as jft
import xubik0 as xu

from .erosita_observation import ErositaObservation
from .sky_models import create_sky_model
from .utils import get_config


# GENERIC
def load_masked_data_from_file(file_path, mask):
    with open(file_path, "rb") as f:
        data_dict = pickle.load(f)
    return jft.Vector({key: mask(value) for key, value in data_dict.items()})


def save_data_dict_to_file(data_dict, file_path):
    #FIXME: Format check
    with open(file_path, "wb") as file:
        pickle.dump(data_dict, file)


# MOCK
def generate_mock_sky_from_prior_dict(npix, padding_ratio, fov, priors, seed=42,
                                      default_point_source_prior=None):
    if priors['point_sources'] is None and default_point_source_prior is None:
        raise ValueError('Point source information is needed for the generation of a mock sky.')
    if priors['point_sources'] is None:
        priors['point_sources'] = default_point_source_prior
    sky_dict = create_sky_model(npix, padding_ratio, fov, priors)
    key = random.PRNGKey(seed)
    key, subkey = random.split(key)
    return jft.random_like(subkey, sky_dict['sky'].domain)


# eROSITA
def load_erosita_masked_data(file_info, tel_info, mask):
    data_list = []
    for tm_id in tel_info['tm_ids']:
        output_filename = f'{tm_id}_' + file_info['output']
        observation_instance = ErositaObservation(file_info['input'], output_filename, file_info['obs_path'])
        data = np.array(observation_instance.load_fits_data(output_filename)[0].data, dtype=int)
        data_list.append(data)
    masked_data_vector = jft.Vector(mask(np.stack(data_list)))
    return masked_data_vector


def generate_erosita_data_from_config(config_file_path, response, output_path=None):
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
    masked_mock_data = response(sky(mock_sky_position))
    if output_path is not None:
        save_data_dict_to_file(masked_mock_data, output_path)
    return masked_mock_data

