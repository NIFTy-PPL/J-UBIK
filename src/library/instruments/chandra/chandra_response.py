from os.path import join, exists
from jax import numpy as jnp
from jax import vmap
import numpy as np

from .chandra_observation import ChandraObservationInformation
from ...utils import get_config, create_output_directory, load_from_pickle,\
    save_to_pickle
from ...plot import plot_result
from ...response import build_readout_function, build_exposure_function
from ...data import Domain
from ...jifty_convolution_operators import linpatch_convolve


def build_chandra_response_from_config(config_file_path):
    """
    Build the Chandra response from the configuration file.
    
    Parameters
    ----------
    config_file_path : str
        Path to the configuration file.
        
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
    cfg = get_config(config_file_path)
    obs_info = cfg['obs_info']
    grid_info = cfg['grid']
    file_info = cfg['files']
    psf_info = cfg['psf']
    outroot = create_output_directory(join(file_info['obs_path'],
                                           file_info['processed_obs_folder']))

    tel_info = cfg['telescope']

    obslist = list(obs_info.keys())
    psf_list = []
    exposure_list = []

    energy_bins = grid_info['energy_bin']
    energy_ranges = tuple(set(energy_bins['e_min']+energy_bins['e_max']))
    elim = (min(energy_ranges), max(energy_ranges))

    exposure_path = join(outroot, 'exposure.pkl')
    psf_path = join(outroot, 'psf.pkl')

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

    if not exists(exposure_path) or not exists(psf_path):
        center = None
        for obsnr in obslist:
            # Observation information for both exposure and PSF
            obs_info_instance = ChandraObservationInformation(
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
                exposure = obs_info_instance.get_exposure(
                    join(outroot, f"exposure_{obsnr}"))
                exposure_list.append(np.transpose(exposure))

            # Compute PSF if it hasn't been loaded
            if not exists(psf_path):
                psf_sim = obs_info_instance.get_psf_fromsim(
                    (info.obsInfo["aim_ra"], info.obsInfo["aim_dec"]),
                    join(outroot, "psf"),
                    num_rays=obs_info_instance['num_rays']
                )
                psf_list.append(psf_sim)

        # Save exposures if they were computed
        if not exists(exposure_path):
            exposures = np.stack(np.array(exposure_list, dtype=int))
            save_to_pickle(exposures, exposure_path)

        # Save PSFs if they were computed
        if not exists(psf_path):
            psfs = jnp.stack(jnp.array(psf_list, dtype=int))
            save_to_pickle(psfs, psf_path)

    def psf_func(x): #FIXME: Veberle please adjust with spat.inv. psf
        return jnp.expand_dims(x, axis=0)
    mask_func = build_readout_function(exposures, keys=obslist,
                                       threshold=tel_info['exp_cut'])
    exposure_func = build_exposure_function(exposures)

    pixel_area = (tel_info['fov'] / grid_info['sdim']) ** 2

    response_func = lambda x: mask_func(exposure_func(psf_func(x \
                                                               * pixel_area)))
    response_dict = {'pix_area': pixel_area,
                     'psf': psf_func,
                     'exposure': exposure_func,
                     'mask': mask_func,
                     'R': response_func}
    return response_dict