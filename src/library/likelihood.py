import os
import nifty8.re as jft
import xubik0 as xu
import numpy as np
from .data import (load_erosita_masked_data, generate_erosita_data_from_config,
                   load_masked_data_from_pickle, Domain)
from .response import (build_callable_from_exposure_file, build_erosita_psf,
                       build_readout_function, build_exposure_function)
from .erosita_observation import ErositaObservation


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
    cfg = xu.get_config(config_file_path)
    tel_info = cfg['telescope']
    file_info = cfg['files']
    psf_info = cfg['psf']
    # lists for exposure and psf files
    exposure_file_names = [os.path.join(file_info['obs_path'], f'{key}_'+file_info['exposure'])
                           for key in tel_info['tm_ids']]
    psf_file_names = [os.path.join(file_info['psf_path'], 'tm'+f'{key}_'+file_info['psf_base_filename'])
                      for key in tel_info['tm_ids']]

    # Get pointings for different telescope modules in RA/DEC
    obs_instance = ErositaObservation(file_info['input'],
                                      file_info['output'],
                                      file_info['obs_path'])
    center_stats = []
    for tm_id in tel_info['tm_ids']:
        tmp_center_stat = obs_instance.get_pointing_coordinates_stats(tm_id, file_info['input'])
        tmp_center_stat = [tmp_center_stat['RA'][0], tmp_center_stat['DEC'][0]]
        center_stats.append(tmp_center_stat)
    center_stats = np.array(center_stats)
    # with respect to TM1
    ref_center = center_stats[0]
    d_centers = center_stats - ref_center
    # Set the Image pointing to the center and associate with TM1 pointing
    image_pointing_center = np.array(tuple([cfg['telescope']['fov']/2.]*2))
    pointing_center = d_centers + image_pointing_center
    domain = Domain(tuple([cfg['grid']['npix']]*2), tuple([cfg['telescope']['fov']/cfg['grid']['npix']]*2))

    # get psf/exposure/mask function
    psf_func = build_erosita_psf(psf_file_names, psf_info['energy'], pointing_center, domain,
                                 psf_info['npatch'], psf_info['margfrac'], psf_info['want_cut'],
                                 psf_info['method'])

    exposure_func = build_callable_from_exposure_file(build_exposure_function,
                                                      exposure_file_names,
                                                      exposure_cut=tel_info['exp_cut'])

    mask_func = build_callable_from_exposure_file(build_readout_function,
                                                  exposure_file_names,
                                                  threshold=tel_info['exp_cut'],
                                                  keys=tel_info['tm_ids'])
    # plugin
    response_func = lambda x: mask_func(exposure_func(psf_func(x))[:,43:-43,43:-43])

    if cfg['mock']:
        masked_data = generate_erosita_data_from_config(config_file_path, response_func,
                                                        file_info['res_dir'])
    elif cfg['load_mock_data']:
        masked_data = load_masked_data_from_pickle(os.path.join(file_info['res_dir'],
                                                   'mock_data_dict.pkl'), mask_func)
    else:
        masked_data = load_erosita_masked_data(file_info, tel_info, mask_func)
    return jft.Poissonian(masked_data) @ response_func
