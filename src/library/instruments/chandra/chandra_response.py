from ...utils import get_config
from ...response import build_readout_function, build_exposure_function
from ...convolution_utils import linpatch


def build_chandra_response_from_config(config_file_path):
    cfg = get_config(config_file_path)
    obs_info = cfg['obs_info']
    grid_info = cfg['grid']
    file_info = cfg['files']
    psf_info = cfg['psf']
    outroot = create_output_directory(file_info["data_outroot"])

    obslist = list(obs_info.keys())
    center = None
    psf_list = []
    exposure_list = []
    for obsnr in obslist:
        info = ChandraObservationInformation(obs_info[f"obs{obsnr}"],
                                             **grid_info,
                                             center=center)
        # compute the exposure map
        exposure = info.get_exposure(os.path.join(outroot, f"exposure_{obsnr}"))
        ju.plot_result(exposure, os.path.join(outroot, f"exposure_{obsnr}.png"), logscale=True)

        # compute the point spread function
        psf_sim = info.get_psf_fromsim((info.obsInfo["aim_ra"],
                                        info.obsInfo["aim_dec"]),
                                        "./psf",
                                        num_rays=psf_info['num_rays'])
        ju.plot_result(psf_sim, os.path.join(outroot, f"psfSIM_{obsnr}.png"), logscale=False)
        exposure_list.append(exposure)
        psf_list.append(psf)

    domain = Domain(tuple([grid_info['npix_e']] + [grid_info['npix_s']] * 2),
                    tuple([1] + [grid_info['fov'] / grid_info['npix_s']] * 2))
    psfs = jnp.stack(jnp.array(psf_list, dtype=int))
    def psf_func(x): #FIXME: Please check
        return vmap(linpatch_convolve, in_axes=(None, None, 0, None, None))(x, domain, psfs,
                                                                            psf_info['npatch'],
                                                                            psf_info['margin'])
    exposures = jnp.stack(jnp.array(exposure_list_list, dtype=int))
    mask_func = build_readout_function(exposures, keys=obslist)
    exposure_func = build_exposure_function(exposures)

    pixel_area = (grid_info['fov'] / grid_info['npix_s']) ** 2

    response_func = lambda x: mask_func(exposure_func(psf_func(x * pixel_area)))
    response_dict = {'pix_area': pixel_area,
                     'psf': psf_func,
                     'exposure': exposure_func,
                     'mask': mask_func,
                     'R': response_func}
    return response_dict