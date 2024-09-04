from os.path import join, exists
from jax import numpy as jnp
from jax import vmap
import numpy as np

from .chandra_observation import ChandraObservationInformation
from ...utils import get_config, create_output_directory
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
    center = None
    psf_list = []
    exposure_list = []

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
        # compute the exposure map
        exposure = info.get_exposure(join(outroot, f"exposure_{obsnr}"))

        # compute the point spread function
        psf_sim = info.get_psf_fromsim((info.obsInfo["aim_ra"],
                                        info.obsInfo["aim_dec"]),
                                        join(outroot, "psf"),
                                        num_rays=psf_info['num_rays'])
        exposure_list.append(exposure)
        psf_list.append(psf_sim)

    domain = Domain(tuple([grid_info['edim']] + [grid_info['sdim']] * 2),
                    tuple([1] + [tel_info['fov'] / grid_info['sdim']] * 2))
    psfs = jnp.stack(jnp.array(psf_list, dtype=int))
    def psf_func(x): #FIXME: Please check
        return vmap(linpatch_convolve,
                    in_axes=(None, None, 0, None, None))(x,
                                                         domain,
                                                         psfs,
                                                         psf_info['npatch'],
                                                         psf_info['margfrac'])
    exposures = np.stack(np.array(exposure_list, dtype=int))
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