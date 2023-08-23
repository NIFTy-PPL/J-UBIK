import nifty8.re as jft
import xubik0 as xu
from .data import load_erosita_masked_data
from .response import apply_callable_from_exposure_file, response, mask


# FIXME: Include into init
def generate_erosita_likelihood_from_config(config_file_path):
    cfg = xu.get_config(config_file_path)
    tel_info = cfg['telescope']
    file_info = cfg['files']
    exposure_file_names = ['{key}_'+file_info['exposure']]
    response_func = apply_callable_from_exposure_file(response, exposure_file_names, tel_info['exp_cut'])
    mask_func = apply_callable_from_exposure_file(mask, exposure_file_names, tel_info['exp_cut'])
    masked_data = xu.load_erosita_masked_data(file_info, tel_info, mask_func)
    return jft.Poissonian(masked_data) @ response_func