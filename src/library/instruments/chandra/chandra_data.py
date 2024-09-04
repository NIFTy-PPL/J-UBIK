from os.path import join, exists
import numpy as np

import nifty8.re as jft
import jubik0 as ju

from ...utils import get_config, create_output_directory, save_to_pickle
from .chandra_observation import ChandraObservationInformation
from ...plot import plot_result
from ...data import create_mock_data

def create_chandra_data_from_config(config_path, response_dict):
    """
    Create Chandra data from a given configuration file.

    This function reads the configuration file and generates the necessary data
    for Chandra observations based on the provided parameters. If the data
    already exists, it skips the data generation process. If the config file
    contains a mock generation configuration, it generates mock data instead.

    Parameters
    ----------
    config_path : str
        Path to the configuration file.
    response_dict : dict
        A dictionary containing the response information, including:
        - 'mask_func': Function to apply a mask to the data.

    Returns
    -------
    None
    """
    cfg = get_config(config_path)
    tel_info = cfg["telescope"]
    file_info = cfg["files"]
    grid_info = cfg['grid']
    plot_info = cfg['plotting']
    obs_info = cfg['obs_info']
    data_path = join(file_info['res_dir'], file_info["data_dict"])
    tel_info['tm_ids'] = list(obs_info.keys())
    if not exists(data_path):
        if bool(file_info.get("mock_gen_config")):
            jft.logger.info(f'Generating new mock data in '
                            f'{file_info["res_dir"]}...')
            mock_prior_info = get_config(file_info["mock_gen_config"])
            create_mock_data(tel_info, file_info, grid_info,
                             mock_prior_info['priors'],
                             plot_info, cfg['seed'], response_dict)
            save_config_copy(file_info['mock_gen_config'],
                             output_dir=file_info['res_dir'])
        else:
            jft.logger.info(f'Generating masked eROSITA '
                            f'data in {file_info["res_dir"]}...')
            data_array = generate_chandra_data(file_info, tel_info,
                                               grid_info, obs_info)
            mask_func = response_dict['mask_func']
            masked_data_vector = mask_func(data_array)
            save_to_pickle(masked_data_vector.tree,
                           data_path)
    else:
        jft.logger.info(f'Data in {file_info["res_dir"]} already '
                        f'exists. No data generation.')

def generate_chandra_data(file_info, tel_info, grid_info, obs_info):
    """
    Generate Chandra data based on the provided observation information.

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
    outroot = create_output_directory(join(file_info['obs_path'],
                                           file_info['processed_obs_folder']))

    obslist = list(obs_info.keys())
    center = None
    data_list = []

    energy_bins = grid_info['energy_bin']
    energy_ranges = tuple(set(energy_bins['e_min']+energy_bins['e_max']))
    elim = (min(energy_ranges), max(energy_ranges))

    for obsnr in obslist:
        info = ChandraObservationInformation(obs_info[obsnr],
                                             npix_s=grid_info['sdim'],
                                             npix_e=grid_info['edim'],
                                             fov=tel_info['fov'],
                                             elim=elim,
                                             energy_ranges=energy_ranges,
                                             center=center)
        # retrieve data from observation
        data = info.get_data(os.path.join(outroot, f"data_{obsnr}.fits"))
        ju.plot_result(data, output_file=os.path.join(outroot,
                                                      f"data_{obsnr}.png"))
        psf_list.append(psf_sim)
    data_array = jnp.stack(jnp.array(data_list, dtype=int))
    return data_array
