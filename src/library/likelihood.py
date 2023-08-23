import os
import nifty8.re as jft
import xubik0 as xu
from .data import (load_erosita_masked_data, generate_erosita_data_from_config,
                   load_masked_data_from_pickle)
from .response import apply_callable_from_exposure_file, response, mask


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
    exposure_file_names = ['{key}_'+file_info['exposure']]
    response_func = apply_callable_from_exposure_file(response, exposure_file_names, tel_info['exp_cut'])
    mask_func = apply_callable_from_exposure_file(mask, exposure_file_names, tel_info['exp_cut'])
    if cfg['mock']:
        masked_data = generate_erosita_data_from_config(config_file_path, response_func,
                                                        file_info['res_dir'])
    elif cfg['load_mock_data']:
        masked_data = load_masked_data_from_pickle(os.path.join(file_info['res_dir'],
                                                   'mock_data_dict.pkl'), mask_func)
    else:
        masked_data = load_erosita_masked_data(file_info, tel_info, mask_func)
    return jft.Poissonian(masked_data) @ response_func