# SPDX-License-Identifier: BSD-2-Clause
# Authors: Vincent Eberle, Matteo Guardiani, Margret Westerkamp

# Copyright(C) 2024 Max-Planck-Society

# %%

from os.path import join, exists
from typing import NamedTuple

import nifty8.re as jft
import numpy as np
from jax import random, linear_transpose

from .sky_models import SkyModel
from .utils import (get_config, create_output_directory,
                    save_to_pickle, load_from_pickle)


class Domain(NamedTuple):
    """Mimicking NIFTy Domain.

    Parameters
    ----------
    shape: tuple
    distances: tuple
    """

    shape: tuple
    distances: tuple


def load_masked_data_from_config(config_path):
    """ Wrapper function load masked data from config path
        from generated pickle-files.

    Parameters
    ----------
    config_path : str
        Path to inference config file

    Returns
    ----------
    masked data: jft.Vector
        Vector of masked eROSITA (mock) data for each TM
    """
    cfg = get_config(config_path)
    file_info = cfg['files']
    data_path = join(file_info['res_dir'], file_info['data_dict'])
    if exists(data_path):
        jft.logger.info('...Loading data from file')
        masked_data = jft.Vector(load_from_pickle(data_path))
    else:
        raise ValueError('Data path does not exist.')
    return masked_data


def load_mock_position_from_config(config_path):
    """ Wrapper function to load the mock sky position for the
    mock data config path from pickle-file.

    Parameters
    ----------
    config_path : str
        Path to inference config file

    Returns
    ----------
    mock_pos : jft.Vector
        Vector of latent parameters for the mock sky position

    """
    cfg = get_config(config_path)
    file_info = cfg['files']
    pos_path = join(file_info['res_dir'], file_info['pos_dict'])
    if exists(pos_path):
        jft.logger.info('...Loading mock position')
        mock_pos = load_from_pickle(pos_path)
    else:
        raise ValueError('Mock position path does not exist.')
    return mock_pos


def create_mock_data(tel_info,
                     file_info,
                     grid_info,
                     prior_info,
                     plot_info,
                     seed,
                     response_dict
                    ):
    """ Generates and saves mock data to pickle file.

    Parameters
    ----------
    tel_info : dict
        Dictionary of telescope information.
    file_info : dict
        Dictionary of file paths.
    grid_info : dict
        Dictionary with grid information.
    grid_info : dict
        Dictionary with prior information.
    prior_info: dict
        Dictionary with prior information.
    plot_info: dict
        Dictionary with plotting information.
    seed: int
        Random seed for mock position generataion.
    response_dict : dict
        Dictionary of all available response functionalities
        i.e. response, mask, psf.
    
    Returns
    -------
    masked_mock_data: jft.Vector
        Dictionary of masked mock data arrays
    """
    key = random.PRNGKey(seed)

    key, subkey = random.split(key)
    sky_model = SkyModel()
    sky = sky_model.create_sky_model(
        sdim=grid_info['sdim'],
        edim=grid_info['edim'],
        s_padding_ratio=grid_info['s_padding_ratio'],
        e_padding_ratio=grid_info['e_padding_ratio'],
        fov=tel_info['fov'],
        e_min=grid_info['energy_bin']['e_min'],
        e_max=grid_info['energy_bin']['e_max'],
        e_ref=grid_info['energy_bin']['e_ref'],
        priors=prior_info)

    jft.random_like(subkey, sky.domain)
    sky_comps = sky_model.sky_model_to_dict()
    key, subkey = random.split(key)
    output_path = create_output_directory(file_info['res_dir'])
    mock_sky_position = jft.Vector(jft.random_like(subkey, sky.domain))
    # TODO: unify cases
    if 'kernel' in response_dict:
        masked_mock_data = response_dict['R'](sky(mock_sky_position),
                                              response_dict['kernel'])
    else:
        masked_mock_data = response_dict['R'](sky(mock_sky_position))
    subkeys = random.split(subkey, len(masked_mock_data.tree))
    masked_mock_data = jft.Vector({
        tm: random.poisson(subkeys[i], data).astype(int)
        for i, (tm, data) in enumerate(masked_mock_data.tree.items())
    })
    save_to_pickle(masked_mock_data.tree,
                   join(output_path, file_info['data_dict']))
    save_to_pickle(mock_sky_position.tree,
                   join(output_path, file_info['pos_dict']))
    if plot_info['enabled']:
        jft.logger.info('Plotting mock data and mock sky.')
        plottable_vector = jft.Vector({key: val.astype(float) for key, val
                                        in masked_mock_data.tree.items()})
        mask_adj = linear_transpose(response_dict['mask'],
                                    np.zeros((len(tel_info['tm_ids']),)
                                             + sky.target.shape))
        mask_adj_func = lambda x: mask_adj(x)[0]
        plottable_data_array = np.stack(mask_adj_func(plottable_vector), axis=0)
        from .plot import plot_rgb
        from .plot import plot_result

        mock_output = create_output_directory(join(file_info['res_dir'],
                                                        'mock_setup'))
        for tm_id in range(plottable_data_array.shape[0]):
            plot_rgb(
                plottable_data_array[tm_id],
                name=join(mock_output, f'mock_data_tm_rgb_log{tm_id+1}'),
                log=True)
            plot_rgb(
                plottable_data_array[tm_id],
                name=join(mock_output, f'mock_data_tm_rgb_{tm_id+1}'),
                log=False,
                sat_min=(np.min(plottable_data_array[0],
                                axis=(1, 2))).tolist(),
                sat_max=(0.1*np.max(plottable_data_array[0],
                                    axis=(1, 2))).tolist())
            plot_result(
                plottable_data_array[tm_id],
                logscale=True,
                output_file=join(mock_output, f'mock_data_tm{tm_id+1}.png'))
        for key, sky_comp in sky_comps.items():
            plot_rgb(sky_comp(mock_sky_position),
                     name=join(mock_output,
                               f'mock_rgb_log_{key}'),
                     log=True)
            plot_rgb(
                sky_comp(mock_sky_position),
                name=join(mock_output, f'mock_rgb_{key}'),
                log=False,
                sat_min=(np.min(sky_comp(mock_sky_position),
                                axis=(1, 2))).tolist(),
                sat_max=(0.1 * np.max(sky_comp(mock_sky_position),
                                      axis=(1, 2))).tolist())
            plot_result(
                sky_comp(mock_sky_position),
                logscale=True,
                output_file=join(mock_output, f'mock_{key}.png'))
        if hasattr(sky_model, 'alpha_cf'):
            diffuse_alpha = sky_model.alpha_cf
            plot_result(
                diffuse_alpha(mock_sky_position),
                logscale=False,
                output_file=join(mock_output, f'mock_diffuse_alpha.png'))
    return masked_mock_data


