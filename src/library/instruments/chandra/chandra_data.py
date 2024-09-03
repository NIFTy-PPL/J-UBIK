import os
import numpy as np

import nifty8 as ift
import jubik0 as ju
from test.library.test_mock_data_generation import exposure

from ...utils import get_config, create_output_directory, save_to_pickle
from .chandra_observation import ChandraObservationInformation
from ...plot import plot_result
#FIXME: Mock
def generate_chandra_data_from_config(config_path, response_dict):
    cfg = get_config(config_path)
    obs_info = cfg['obs_info']
    grid_info = cfg['grid']
    file_info = cfg['files']
    outroot = create_output_directory(file_info["data_outroot"])

    obslist = list(obs_info.keys())
    center = None
    data_list = []
    psf_list = []
    exposure_list = []
    for obsnr in obslist:
        info = ChandraObservationInformation(obs_info[f"obs{obsnr}"],
                                             **grid_info,
                                             center=center)
        # retrieve data from observation
        data = info.get_data(os.path.join(outroot, f"data_{obsnr}.fits"))

        ju.plot_result(data, os.path.join(outroot, f"data_{obsnr}.png"), logscale=True)

        # compute the exposure map
        exposure = info.get_exposure(os.path.join(outroot, f"exposure_{obsnr}"))
        ju.plot_result(exposure, os.path.join(outroot, f"exposure_{obsnr}.png"), logscale=True)

        # compute the point spread function
        psf_sim = info.get_psf_fromsim((info.obsInfo["aim_ra"],
                                        info.obsInfo["aim_dec"]),
                                        "./psf",
                                        num_rays=img_cfg["psf"]['num_rays'])
        ju.plot_result(psf_sim, os.path.join(outroot, f"psfSIM_{obsnr}.png"), logscale=False)
        data_list.append(data)
        psf_list.append(psf_sim)
        exposure_list.append(exposure)
    data = jnp.stack(jnp.array(data_list, dtype=int))
    mask_func = response_dict['mask_func']
    masked_data_vector = mask_func(data)
    save_to_pickle(masked_data_vector.tree, join(file_info['res_dir'], file_info["data_dict"]))
    return masked_data_vector