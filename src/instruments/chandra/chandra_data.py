# SPDX-License-Identifier: BSD-2-Clause
# Authors: Vincent Eberle, Matteo Guardiani, Margret Westerkamp

# Copyright(C) 2024 Max-Planck-Society

# %%

from os.path import join, exists
from jax import numpy as jnp

import nifty8.re as jft

from ...utils import get_config, create_output_directory, save_to_pickle,\
    copy_config, load_from_pickle
from .chandra_observation import ChandraObservationInformation
from ...data import create_mock_data


def create_chandra_data_from_config(config, response_dict):
    """
    Create Chandra data from a given configuration file.

    This function reads the configuration file and generates the necessary data
    for Chandra observations based on the provided parameters. If the data
    already exists, it skips the data generation process. If the config file
    contains a mock generation configuration, it generates mock data instead.

    Parameters
    ----------
    config : dict
        YAML configuration dictionary.
    response_dict : dict
        A dictionary containing the response information, including:
        - 'mask_func': Function to apply a mask to the data.

    Returns
    -------
    None
    """
    tel_info = config["telescope"]
    file_info = config["files"]
    grid_info = config['grid']
    plot_info = config['plotting']
    obs_info = config['obs_info']
    data_path = join(file_info['res_dir'], file_info["data_dict"])
    tel_info['tm_ids'] = list(obs_info.keys())
    if not exists(data_path):
        if bool(file_info.get("mock_gen_config")):
            jft.logger.info(f'Generating new mock data in '
                            f'{file_info["res_dir"]}...')
            mock_prior_info = get_config(file_info["mock_gen_config"])
            create_mock_data(tel_info, file_info, grid_info,
                             mock_prior_info['priors'],
                             plot_info, config['seed'], response_dict)
            copy_config(file_info['mock_gen_config'],
                             output_dir=file_info['res_dir'])
        else:
            jft.logger.info(f'Generating masked Chandra'
                            f'data in {file_info["res_dir"]}...')
            data_array = generate_chandra_data(file_info, tel_info,
                                               grid_info, obs_info)
            mask_func = response_dict['mask']
            masked_data_vector = mask_func(data_array)
            save_to_pickle(masked_data_vector.tree,
                           data_path)
    else:
        jft.logger.info(f'Data in {file_info["res_dir"]} already '
                        f'exists. No data generation.')


def generate_chandra_data(file_info, tel_info, grid_info, obs_info):
    """
    Generate the necessary binned data files from Chandra observation info.

    This function processes the observation information for Chandra 
    and generates the corresponding Chandra data arrays.

    Parameters
    ----------
    file_info : dict
        Dictionary containing file path and name information.
    tel_info : dict
        Dictionary containing telescope information, including field of view.
    grid_info : dict
        Dictionary containing grid dimensions and energy bin information.
    obs_info : dict
        Dictionary containing observation information.
    Returns
    -------
    data_array : jnp.ndarray
        The generated Chandra data array.
    """
    outroot = create_output_directory(join(file_info['res_dir'],
                                           file_info['processed_obs_folder']))

    obslist = list(obs_info.keys())
    center_obs_id = tel_info.get('center_obs_id', None)
    if center_obs_id is not None and center_obs_id in obslist:
        obslist.remove(center_obs_id)
        obslist.insert(0, center_obs_id)

    center = None
    data_list = []

    energy_bins = grid_info['energy_bin']
    energy_ranges = tuple(set(energy_bins['e_min']+energy_bins['e_max']))
    elim = (min(energy_ranges), max(energy_ranges))

    data_path = join(outroot, 'data.pkl')
    if exists(data_path):
        data_array = load_from_pickle(data_path)
    else:
        for i, obsnr in enumerate(obslist):
            info = ChandraObservationInformation(obs_info[obsnr],
                                                npix_s=grid_info['sdim'],
                                                npix_e=grid_info['edim'],
                                                fov=tel_info['fov'],
                                                elim=elim,
                                                energy_ranges=energy_ranges,
                                                center=center)
            # retrieve data from observation
            data = info.get_data(join(outroot, f"data_{obsnr}.fits"))
            data_list.append(jnp.transpose(data))
        data_array = jnp.stack(jnp.array(data_list, dtype=int))
        if i == 0:
            center = (info.obsInfo["aim_ra"], info.obsInfo["aim_dec"])
    return data_array
