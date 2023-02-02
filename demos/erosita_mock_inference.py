import math
import os
import sys
import pickle
import numpy as np
from matplotlib import colors
from sys import exit


import nifty8 as ift
import xubik0 as xu


currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from src.library.plot import plot_sample_and_stats
from src.library.utils import create_output_directory


mockrun = True
mock_psf = False
hyperparamerter_search = False
load_mock_data = True

if __name__ == "__main__":

    # ########################################## CONFIGS #############################################################
    if load_mock_data and not mockrun:
        print('WARNING: Mockrun is set to False: Actual data is loaded')
        load_mock_data = False
    config_filename = "eROSITA_config_mw.yaml"
    try:
        config_path = config_filename
        cfg = xu.get_cfg(config_path)
    except:
        config_path = 'demos/' + config_filename
        cfg = xu.get_cfg(config_path)
    output_directory = create_output_directory("retreat_first_reconstruction")
    fov = cfg['telescope']['fov']
    rebin = math.floor(20 * fov // cfg['grid']['npix'])

    # File Location
    file_info = cfg['files']
    obs_path = file_info['obs_path']
    input_filenames = file_info['input']
    output_filename = file_info['output']
    exposure_filename = file_info['exposure']
    observation_instance = xu.ErositaObservation(input_filenames, output_filename, obs_path)
    sky_model = xu.SkyModel(config_path)
    point_sources, diffuse, sky = sky_model.create_sky_model()

    # Grid Info
    grid_info = cfg['grid']
    e_min = grid_info['energy_bin']['e_min']
    e_max = grid_info['energy_bin']['e_max']
    npix = grid_info['npix']

    # Telescope Info
    tel_info = cfg['telescope']
    tm_id = tel_info['tm_id']

    # ####################################### GET DATA ################################################################
    log = 'Output file {} already exists and is not regenerated. ' \
          'If the observations parameters shall be changed please delete or rename the current output file.'

    if not os.path.exists(os.path.join(obs_path, output_filename)):
        observation = observation_instance.get_data(emin=e_min, emax=e_max, image=True, rebin=rebin,
                                                    size=npix, pattern=tel_info['pattern'],
                                                    telid=tm_id)
    else:
        print(log.format(os.path.join(obs_path, output_filename)))

    observation_instance = xu.ErositaObservation(output_filename, output_filename, obs_path)

    # Get Exposure
    if not os.path.exists(os.path.join(obs_path, exposure_filename)):
        observation_instance.get_exposure_maps(output_filename, e_min, e_max, mergedmaps=exposure_filename)

    else:
        print(log.format(os.path.join(obs_path, output_filename)))

    # Plotting
    plot_info = cfg['plotting']
    if plot_info['enabled']:
        observation_instance.plot_fits_data(output_filename,
                                            os.path.splitext(output_filename)[0],
                                            slice=plot_info['slice'],
                                            dpi=plot_info['dpi'])
        observation_instance.plot_fits_data(exposure_filename,
                                            f'{os.path.splitext(exposure_filename)[0]}.png',
                                            slice=plot_info['slice'],
                                            dpi=plot_info['dpi'])


    # ################################### LIKELIHOOD ###################################################################

    # PSF
    if mock_psf:
        psf_kernel = None
    else:
        center = observation_instance.get_center_coordinates(output_filename)
        psf_file = xu.eROSITA_PSF(cfg["files"]["psf_path"])
        psf_function = psf_file.psf_func_on_domain('3000', center, sky_model.extended_space)
        psf_kernel = psf_function(*center)
        psf_kernel = ift.makeField(sky_model.extended_space, np.array(psf_kernel))




    # Exposure
    exposure = observation_instance.load_fits_data(exposure_filename)[0].data
    exposure_field = ift.makeField(sky_model.position_space, exposure)
    padded_exposure_field = sky_model.pad(exposure_field)
    exposure_op = ift.makeOp(padded_exposure_field)

    # Mask
    mask = xu.get_mask_operator(exposure_field)

    # Response
    R = mask @ sky_model.pad.adjoint @ exposure_op

    # Actual Data

    data = observation_instance.load_fits_data(output_filename)[0].data
    data = ift.makeField(sky_model.position_space, data)

    # Mock data and likelihood

    if mockrun:
        if hyperparamerter_search:
            n_mock_samples = 1
            for n in range(n_mock_samples):
                for alpha in [1.0001]:
                    for q in [0.0000001]:
                        sky_model = xu.SkyModel(config_path, alpha=alpha, q=q)
                        mock_sky_data, _ = xu.generate_mock_data(sky_model, exposure_field,
                                                                 sky_model.pad, psf_kernel,
                                                                 alpha, q, n, var=tel_info['var'],
                                                                 output_directory=output_directory)
            exit()
        else:
            if load_mock_data:
                #FIXME: name of output folder for diagnostics into config
                with open('diagnostics/mock_sky_data.pkl', "rb") as f:
                    mock_sky_data = pickle.load(f)
                if psf_kernel is None:
                    convolved = xu.get_gaussian_psf(sky, var=tel_info['var'])
                else:
                    convolved = xu.convolve_field_operator(psf_kernel, sky)
            else:
                mock_sky_data, convolved = xu.generate_mock_data(sky_model, exposure_field,
                                                                 sky_model.pad, psf_kernel,
                                                                 var=tel_info['var'],
                                                                 output_directory=output_directory)
            masked_data = mask(mock_sky_data)
            #Likelihood
            log_likelihood = ift.PoissonianEnergy(masked_data) @ R @ convolved
    else:
        masked_data = mask(data)
        log_likelihood = ift.PoissonianEnergy(masked_data) @ R @ sky

    # Print Exposure norm
    # norm = xu.get_norm(exposure, data)
    # print(norm)

    # ######################################## INFERENCE ############################################################

    # Load minimization config
    minimization_config = cfg['minimization']

    # Save config file in output_directory
    xu.save_config(cfg, config_filename, output_directory)

    # Minimizers
    ic_newton = ift.AbsDeltaEnergyController(**minimization_config['ic_newton'])
    ic_sampling = ift.AbsDeltaEnergyController(**minimization_config['ic_sampling'])
    ic_sampling_nl = ift.AbsDeltaEnergyController(**minimization_config['ic_sampling_nl'])
    minimizer = ift.NewtonCG(ic_newton)
    minimizer_sampling = ift.NewtonCG(ic_sampling_nl)

    # Prepare results
    operators_to_plot = {'reconstruction': sky, 'point_sources': point_sources, 'diffuse_component': diffuse}
    plot = lambda x, y: plot_sample_and_stats(output_directory, operators_to_plot, x, y,
                                              plotting_kwargs={'norm': colors.SymLogNorm(linthresh=10e-1)})

    if minimization_config['geovi']:
        # geoVI
        ift.optimize_kl(log_likelihood, minimization_config['total_iterations'], minimization_config['n_samples'],
                        minimizer, ic_sampling, minimizer_sampling, output_directory=output_directory,
                        export_operator_outputs=operators_to_plot, inspect_callback=plot, resume=True)
    else:
        # MGVI
        ift.optimize_kl(log_likelihood, minimization_config['total_iterations'], minimization_config['n_samples'],
                        minimizer, ic_sampling, None, export_operator_outputs=operators_to_plot,
                        output_directory=output_directory, inspect_callback=plot, resume=True)
