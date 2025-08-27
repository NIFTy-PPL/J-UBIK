# SPDX-License-Identifier: BSD-2-Clause
# Authors: Vincent Eberle, Matteo Guardiani, Margret Westerkamp

# Copyright(C) 2024 Max-Planck-Society

# %%

from os.path import join, exists
from jax import vmap
import numpy as np

from .chandra_observation import ChandraObservationInformation
from .chandra_psf import get_psfpatches
from ...utils import create_output_directory, load_from_pickle, save_to_pickle
from ...response import build_readout_function, build_exposure_function
from ...convolve import linpatch_convolve, integrate
from ...data import Domain


def build_chandra_response_from_config(config):
    """
    Build the Chandra response from the configuration file.
    
    Parameters
    ----------
    config : dict
        Dictionary containing the configuration parameters.
        
    Returns 
    -------
    response_dict : dict
        A dictionary containing the response information, including:
        - 'pix_area': The pixel area.
        - 'psf': The point spread function.
        - 'exposure': The exposure function.
        - 'mask': The mask function.
        - 'R': The response function.
    """
    obs_info = config['obs_info']
    grid_info = config['grid']
    file_info = config['files']
    psf_info = config['psf']
    outroot = create_output_directory(join(file_info['res_dir'],
                                           file_info['processed_obs_folder']))

    tel_info = config['telescope']

    obslist = list(obs_info.keys())
    psf_list = []
    exposure_list = []

    energy_bins = grid_info['energy_bin']
    energy_ranges = tuple(set(energy_bins['e_min']+energy_bins['e_max']))
    elim = (min(energy_ranges), max(energy_ranges))

    exposure_path = join(outroot, 'exposure.pkl')
    psf_path = join(outroot, 'psf.pkl')

    domain = Domain(tuple([grid_info["edim"]] + [grid_info["sdim"]] * 2),
                    tuple([1] + [tel_info["fov"] / grid_info["sdim"]] * 2))
    shp = (domain.shape[-2], domain.shape[-1])
    margin = max((int(np.ceil(psf_info["margfrac"] * ss)) for ss in shp))

    # Load exposures if the file exists, otherwise compute it
    if exists(exposure_path):
        exposures = load_from_pickle(exposure_path)
    else:
        exposure_list = []

    # Load PSFs if the file exists, otherwise compute it
    if exists(psf_path):
        psfs = load_from_pickle(psf_path)
    else:
        psf_list = []

    center = tel_info['center']

    if not exists(exposure_path) or not exists(psf_path):
        for i, obsnr in enumerate(obslist):
            # Observation information for both exposure and PSF
            info = ChandraObservationInformation(
                obs_info[obsnr],
                npix_s=grid_info['sdim'],
                npix_e=grid_info['edim'],
                fov=tel_info['fov'],
                elim=elim,
                energy_ranges=energy_ranges,
                center=center
            )

            # Compute exposure if it hasn't been loaded
            if not exists(exposure_path):
                exposure = info.get_exposure(
                    join(outroot, f"exposure_{obsnr}"))
                exposure_list.append(np.transpose(exposure))

            # Compute PSF if it hasn't been loaded
            if not exists(psf_path):
                tmp_psfs = []
                for ebin in range(grid_info["edim"]):
                    psf_array = get_psfpatches(info,
                                               psf_info["npatch"],
                                               grid_info["sdim"],
                                               ebin,
                                               num_rays=psf_info["num_rays"],
                                               Norm=False)
                    tmp_psfs.append(psf_array)
                psf_list.append(np.moveaxis(np.array(tmp_psfs), 0, 1))
        # Save exposures if they were computed
        if not exists(exposure_path):
            exposures = np.stack(np.array(exposure_list, dtype=int))
            save_to_pickle(exposures, exposure_path)

        # Save PSFs if they were computed
        if not exists(psf_path):
            psfs = np.stack(np.array(psf_list, dtype=int))
            norm = np.max(integrate(psfs, domain, [-2, -1]))
            psfs = psfs / norm
            save_to_pickle(psfs, psf_path)

    def psf_func(x):
        return vmap(linpatch_convolve, in_axes=(None, None, 0, None, None))(
            x, domain, psfs, psf_info["npatch"], margin
        )

    mask_func = build_readout_function(exposures, keys=obslist,
                                       threshold=tel_info['exp_cut'])
    exposure_func = build_exposure_function(exposures)

    pixel_area = (tel_info['fov'] / grid_info['sdim']) ** 2

    def response_func(x):
        conv = psf_func(x*pixel_area)
        exposed = exposure_func(conv)
        masked = mask_func(exposed)
        return masked

    response_dict = {'pix_area': pixel_area,
                     'psf': psf_func,
                     'exposure': exposure_func,
                     'mask': mask_func,
                     'R': response_func}
    return response_dict
