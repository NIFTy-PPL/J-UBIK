import os.path

import numpy as np
import pickle

import nifty8 as ift
import xubik0 as xu

if __name__ == "__main__":
    """This is the postprocessing pipeline for the eROSITA reconstruction:
    The uncertainty weighted residual (UWR) as well as their distribution 
    and the uncertainty weighted mean (UWM) are calculated. For mock data
    additionally the UWR in signal space is calculated."""

    # Paths -Set by user
    reconstruction_path = "results/LMC/"  # FIXME filepath
    diagnostics_path = reconstruction_path + "diagnostics/"
    config_filename = "eROSITA_config.yaml"
    sl_path_base = reconstruction_path + "pickle/last"  # NIFTy dependency
    data_base = "data.pkl"
    exposure_base = "exposure.pkl"
    response_base = None  # FIXME response operator shall be loaded from path
    # Ground Truth path Only needed for mock run
    ground_truth_filename = "mock_sky.pkl"

    # Config
    cfg = xu.get_cfg(reconstruction_path + config_filename)
    mock_run = cfg['mock']
    mock_psf = cfg['mock_psf']
    file_info = cfg['files']
    obs_path = file_info['obs_path']
    exposure_filename = file_info['exposure']

    # Telescope Info
    tel_info = cfg['telescope']
    tm_ids = tel_info['tm_ids']
    start_center = None

    # Exposure Info
    det_map = tel_info['detmap']

    # Operators
    # Sky
    sky_model = xu.SkyModel(reconstruction_path + config_filename)
    sky_dict = sky_model.create_sky_model()
    signal_space_uwrs = []
    data_space_uwrs = []

    for tm_id in tm_ids:
        # Path
        tm_directory = xu.create_output_directory(os.path.join(diagnostics_path, f'tm{tm_id}/'))
        data_path = tm_directory + f"tm{tm_id}_{data_base}"

        # Load observation
        output_filename = f'{tm_id}_' + file_info['output']
        observation_instance = xu.ErositaObservation(output_filename, output_filename, obs_path)

        # Repsonse
        if response_base is not None:
            response_path = tm_directory + f"tm{tm_id}_{response_base}"
            with open(response_path, "rb") as f:
                R = pickle.load(f)
        elif exposure_base is not None:
            exposure_path = tm_directory + f"tm{tm_id}_{exposure_base}"
            print('Not able to load response from file. Generating response from config ...')
            with open(exposure_path, "rb") as f:
                exposure_field = pickle.load(f)
            padded_exposure_field = sky_model.pad(exposure_field)
            exposure_op = ift.makeOp(padded_exposure_field)
            mask = xu.get_mask_operator(exposure_field)

            # PSF
            psf_filename = cfg['files']['psf_path'] + f'tm{tm_id}_' + cfg['files']['psf_base_filename']
            if cfg['psf']['method'] in ['MSC', 'LIN']:
                center_stats = observation_instance.get_pointing_coordinates_stats(1)  # FIXME
                center = (center_stats['RA'][0], center_stats['DEC'][0])
                psf_file = xu.eROSITA_PSF(psf_filename)
                if start_center is None:
                    dcenter = (0, 0)
                    start_center = center
                else:
                    dcenter = (cc - ss for cc, ss in zip(center, start_center))

                dom = sky_model.extended_space
                center = tuple(0.5 * ss * dd for ss, dd in zip(dom.shape, dom.distances))
                center = tuple(cc + dd for cc, dd in zip(center, dcenter))

                energy = cfg['psf']['energy']
                conv_op = psf_file.make_psf_op(energy, center, sky_model.extended_space,
                                               conv_method=cfg['psf']['method'],
                                               conv_params=cfg['psf'])

            elif cfg['psf']['method'] == 'invariant':
                if mock_psf:
                    conv_op = xu.get_gaussian_psf(sky_dict['sky'], var=cfg['psf']['gauss_var'])
                else:
                    center = observation_instance.get_center_coordinates(output_filename)
                    psf_file = xu.eROSITA_PSF(cfg["files"]["psf_path"])
                    psf_function = psf_file.psf_func_on_domain('3000', center, sky_model.extended_space)
                    psf_kernel = psf_function(*center)
                    psf_kernel = ift.makeField(sky_model.extended_space, np.array(psf_kernel))
                    conv_op = xu.get_fft_psf_op(psf_kernel, sky_dict['sky'])

            R = mask @ sky_model.pad.adjoint @ exposure_op @ conv_op
        else:
            raise NotImplementedError

        if mock_run:
            ground_truth_path = diagnostics_path + ground_truth_filename
            signal_space_uwrs.append(xu.signal_space_uwr_from_file(sl_path_base=sl_path_base,
                                                                   ground_truth_path=ground_truth_path,
                                                                   sky_op=sky_dict['sky'],
                                                                   output_dir_base=tm_directory +
                                                                                   f'/{tm_id}_signal_space_uwr'))
        data_space_uwrs.append(xu.data_space_uwr_from_file(sl_path_base=sl_path_base, data_path=data_path,
                                                           sky_op=sky_dict['sky'], response_op=R, mask_op=mask,
                                                           output_dir_base=tm_directory + f'/{tm_id}_data_space_uwr'))
        xu.weighted_residual_distribution(sl_path_base=sl_path_base, data_path=data_path,
                                          sky_op=sky_dict['sky'], response_op=R, mask_op=mask,
                                          output_dir_base=tm_directory + f'/{tm_id}_res_distribution')
    xu.signal_space_uwm_from_file(sl_path_base=sl_path_base, sky_op=sky_dict['sky'],
                                  output_dir_base=diagnostics_path + '/uwm')
    field_name_list = [f'tm{tm_id}' for tm_id in tm_ids]
    if mock_run:
        xu.plot_energy_slice_overview(signal_space_uwrs, field_name_list=field_name_list,
                                      file_name='signal_space_uwrs.png', title='signal_space_uwrs',
                                      logscale=True)
    xu.plot_energy_slice_overview(data_space_uwrs, field_name_list=field_name_list,
                                  file_name=diagnostics_path+'data_space_uwrs.png', title='data_space_uwrs',
                                  logscale=True)

