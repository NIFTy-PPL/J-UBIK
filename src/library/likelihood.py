import os
import nifty8.re as jft
import xubik0 as xu
import numpy as np
from .data import (load_erosita_masked_data, generate_erosita_data_from_config,
                   load_masked_data_from_pickle)
from .response import (build_callable_from_exposure_file, build_erosita_response,
                       build_readout_function, build_exposure_function)


# FIXME: Include into init
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

    cfg = xu.get_config(config_file_path)
    tel_info = cfg['telescope']
    file_info = cfg['files']
    exposure_file_names = [os.path.join(file_info['obs_path'], f'{key}_'+file_info['exposure'])
                           for key in tel_info['tm_ids']]
    exposure_func = build_callable_from_exposure_file(build_exposure_function,
                                                      exposure_file_names,
                                                      exposure_cut=tel_info['exp_cut'])
    mask_func = build_callable_from_exposure_file(build_readout_function,
                                                  exposure_file_names,
                                                  threshold=tel_info['exp_cut'],
                                                  keys=tel_info['tm_ids'])
    psf_func = lambda x: x
    response_func = lambda x: mask_func(exposure_func(psf_func(x)))

    if cfg['mock']:
        masked_data = generate_erosita_data_from_config(config_file_path, response_func,
                                                        file_info['res_dir'])
    elif cfg['load_mock_data']:
        masked_data = load_masked_data_from_pickle(os.path.join(file_info['res_dir'],
                                                   'mock_data_dict.pkl'), mask_func)
    else:
        masked_data = load_erosita_masked_data(file_info, tel_info, mask_func)
    return jft.Poissonian(masked_data) @ response_func