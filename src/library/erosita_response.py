import math
import os
import pickle

import nifty8 as ift
import numpy as np

from .erosita_observation import ErositaObservation
from .erosita_psf import eROSITA_PSF
from .sky_models import SkyModel
from .utils import get_cfg, create_output_directory, get_gaussian_psf, get_fft_psf_op, \
    get_mask_operator


def load_erosita_response(config_filename, diagnostics_directory):
    cfg = get_cfg(config_filename)
    fov = cfg['telescope']['fov']
    rebin = math.floor(20 * fov // cfg['grid']['npix'])  # FIXME USE DISTANCES!
    mock_run = cfg['mock']
    mock_psf = cfg['mock_psf']

    # Grid Info
    grid_info = cfg['grid']
    e_min = grid_info['energy_bin']['e_min']
    e_max = grid_info['energy_bin']['e_max']
    npix = grid_info['npix']

    # Telescope Info
    tel_info = cfg['telescope']
    tm_ids = tel_info['tm_ids']
    start_center = None

    # Exposure Info
    det_map = tel_info['detmap']

    # Load file location
    file_info = cfg['files']
    obs_path = file_info['obs_path']
    input_filenames = file_info['input']
    sky_model = SkyModel(config_filename)
    sky_dict = sky_model.create_sky_model()

    log = 'Output file {} already exists and is not regenerated. ' \
          'If the observations parameters shall be changed please delete or rename the current ' \
          'output file.'

    response_dict = {}
    for tm_id in tm_ids:
        tm_directory = create_output_directory(os.path.join(diagnostics_directory, f'tm{tm_id}'))
        output_filename = f'{tm_id}_' + file_info['output']
        exposure_filename = f'{tm_id}_' + file_info['exposure']
        observation_instance = ErositaObservation(input_filenames, output_filename, obs_path)
        # Create subdictionary to store individual tm response
        tm_key = f'tm_{tm_id}'
        response_dict[tm_key] = {}
        response_subdict = response_dict[tm_key]

        if not mock_run:
            if not os.path.exists(os.path.join(obs_path, output_filename)):
                observation_instance.get_data(emin=e_min, emax=e_max, image=True,
                                              rebin=rebin,
                                              size=npix, pattern=tel_info['pattern'],
                                              telid=tm_id)  # FIXME: exchange rebin
                # by fov? 80 = 4arcsec
            else:
                print(log.format(os.path.join(obs_path, output_filename)))

            observation_instance = ErositaObservation(output_filename, output_filename, obs_path)

        # Exposure
        if not os.path.exists(os.path.join(obs_path, exposure_filename)):
            observation_instance.get_exposure_maps(output_filename, e_min, e_max,
                                                   singlemaps=exposure_filename,
                                                   withdetmaps=det_map)

        else:
            print(log.format(os.path.join(obs_path, output_filename)))

        # Plotting
        plot_info = cfg['plotting']
        if plot_info['enabled']:
            if not mock_run:
                observation_instance.plot_fits_data(output_filename,
                                                    os.path.splitext(output_filename)[0],
                                                    slice=plot_info['slice'],
                                                    dpi=plot_info['dpi'])
            observation_instance.plot_fits_data(exposure_filename,
                                                f'{os.path.splitext(exposure_filename)[0]}.png',
                                                slice=plot_info['slice'],
                                                dpi=plot_info['dpi'])

        # PSF
        psf_filename = cfg['files']['psf_path'] + f'tm{tm_id}_' + cfg['files']['psf_base_filename']
        if cfg['psf']['method'] in ['MSC', 'LIN']:
            center_stats = observation_instance.get_pointing_coordinates_stats(tm_id,
                                                                               input_filename=input_filenames)
            print(center_stats['ROLL'])
            center = (center_stats['RA'][0], center_stats['DEC'][0])
            psf_file = eROSITA_PSF(psf_filename)
            if start_center is None:
                dcenter = (0, 0)
                start_center = center
            else:
                dcenter = (cc - ss for cc, ss in zip(center, start_center))

            dom = sky_model.position_space
            center = tuple(0.5 * ss * dd for ss, dd in zip(dom.shape, dom.distances))
            center = tuple(cc + dd for cc, dd in zip(center, dcenter))

            energy = cfg['psf']['energy']
            conv_op = psf_file.make_psf_op(energy, center, sky_model.extended_space,
                                           conv_method=cfg['psf']['method'],
                                           conv_params=cfg['psf'])

        elif cfg['psf']['method'] == 'invariant':
            if mock_psf:
                conv_op = get_gaussian_psf(sky_dict['sky'], var=cfg['psf']['gauss_var'])
            else:
                center = observation_instance.get_pointing_coordinates_stats(tm_id)
                psf_file = eROSITA_PSF(psf_filename)
                psf_function = psf_file.psf_func_on_domain('3000', center, sky_model.extended_space)
                psf_kernel = psf_function(*center)
                psf_kernel = ift.makeField(sky_model.extended_space, np.array(psf_kernel))
                conv_op = get_fft_psf_op(psf_kernel, sky_dict['sky'])
        else:
            raise NotImplementedError
        response_subdict[f'convolution_op'] = conv_op

        # Exposure
        exposure = observation_instance.load_fits_data(exposure_filename)[0].data
        exposure_cut = tel_info["exp_cut"]
        if exposure_cut is not None:
            exposure[exposure < exposure_cut] = 0
        exposure_field = ift.makeField(sky_model.position_space, exposure)

        with open(tm_directory + f"/tm{tm_id}_exposure.pkl", "wb") as f:
            pickle.dump(exposure_field, f)
        padded_exposure_field = sky_model.pad(exposure_field)
        exposure_op = ift.makeOp(padded_exposure_field)
        response_subdict[f'exposure_op'] = exposure_op

        # Mask
        mask = get_mask_operator(exposure_field)
        response_subdict[f'mask'] = mask

        # Response
        R = mask @ sky_model.pad.adjoint @ exposure_op @ conv_op
        response_subdict[f'R'] = R

    return response_dict
