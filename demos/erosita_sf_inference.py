import argparse
import math
import os
import sys

import numpy as np
from matplotlib import colors
import nifty8 as ift
import xubik0 as xu
from demos.sky_model import ErositaSky

from src.library.plot import plot_sample_and_stats, create_output_directory

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from src.library.erosita_observation import ErositaObservation

mockrun = True
hyperparamerter_search= True
if __name__ == "__main__":
    config_filename = "eROSITA_config.yaml"
    try:
        cfg = xu.get_cfg(config_filename)
    except:
        cfg = xu.get_cfg('demos/' + config_filename)
    fov = cfg['telescope']['field_of_view']
    rebin = math.floor(20 * fov // cfg['grid']['npix'])

    # File Location
    file_info = cfg['files']
    obs_path = file_info['obs_path']
    input_filenames = file_info['input']
    output_filename = file_info['output']
    exposure_filename = file_info['exposure']
    observation_instance = ErositaObservation(input_filenames, output_filename, obs_path)
    sky_model = ErositaSky(config_filename)
    point_sources, diffuse, sky = sky_model.create_sky_model()

    # Grid Info
    grid_info = cfg['grid']
    e_min = grid_info['energy_bin']['e_min']
    e_max = grid_info['energy_bin']['e_max']
    npix = grid_info['npix']

    # Telescope Info
    tel_info = cfg['telescope']
    tm_id = tel_info['tm_id']

    log = 'Output file {} already exists and is not regenerated. ' \
          'If the observations parameters shall be changed please delete or rename the current output file.'

    if not os.path.exists(os.path.join(obs_path, output_filename)):
        observation = observation_instance.get_data(emin=e_min, emax=e_max, image=True, rebin=rebin,
                                                    size=npix, pattern=tel_info['pattern'],
                                                    telid=tm_id)  # FIXME: exchange rebin by fov? 80 = 4arcsec
    else:
        print(log.format(os.path.join(obs_path, output_filename)))

    observation_instance = ErositaObservation(output_filename, output_filename, obs_path)

    # Exposure
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

    # PSF
    # FIXME: Make sure that this is in arcseconds!
    center = observation_instance.get_center_coordinates(output_filename)

    if mockrun:
        def gaussian_psf(op, var):
            dist_x = op.target[0].distances[0]
            dist_y = op.target[0].distances[1]

            # Periodic Boundary conditions
            x_ax = np.arange(op.target[0].shape[0])
            x_ax = np.minimum(x_ax, op.target[0].shape[0] - x_ax) * dist_x
            y_ax = np.arange(op.target[0].shape[1])
            y_ax = np.minimum(y_ax, op.target[0].shape[1] - y_ax) * dist_y

            center = (0, 0)
            x_ax -= center[0]
            y_ax -= center[1]
            X, Y = np.meshgrid(x_ax, y_ax, indexing='ij')

            var *= op.target[0].scalar_dvol  # ensures that the variance parameter is specified with respect to the

            # normalized psf
            log_psf = - (0.5 / var) * (X ** 2 + Y ** 2)
            log_kernel = ift.makeField(op.target[0], log_psf)
            log_kernel = log_kernel - np.log(log_kernel.exp().integrate().val)

            # p = ift.Plot()
            # import matplotlib.colors as colors
            # p.add(log_kernel.exp(), norm=colors.SymLogNorm(linthresh=10e-8))
            # p.output(nx=1)

            conv = xu.convolve_field_operator(log_kernel.exp(), op)
            return conv

        var = 10
        convolved_sky = gaussian_psf(op=sky, var=var)
        convolved_ps = gaussian_psf(op=point_sources, var=var)
        convolved_diffuse = gaussian_psf(op=diffuse, var=var)
    else:
        # TODO instantiate actual eROSITA PSF
        # PSF_op = ... instantiate psf op(args)
        # args contains pointing_center, domain, ...
        raise NotImplementedError


    # Exposure
    exposure = observation_instance.load_fits_data(exposure_filename)[0].data
    exposure_field = ift.makeField(sky_model.position_space, exposure)
    padded_exposure_field = sky_model.pad(exposure_field)
    exposure_op = ift.makeOp(padded_exposure_field)

    # Mask
    mask = xu.get_mask_operator(exposure_field)

    # Response
    R = mask @ sky_model.pad.adjoint @ exposure_op

    # Data
    data = observation_instance.load_fits_data(output_filename)[0].data
    data = ift.makeField(sky_model.position_space, data)
    padded_data = sky_model.pad(data)
    masked_data = mask(data)

    if mockrun:
        ift.random.push_sseq_from_seed(cfg['seed'])

        def get_data_realization(op, data=True):
            mock_position = ift.from_random(op.domain)
            resp = sky_model.pad.adjoint @ exposure_op
            res = op(mock_position)
            if data:
                res = resp(op(mock_position))
                res = ift.random.current_rng().poisson(res.val.astype(np.float64))
                res = ift.makeField(sky_model.pad.adjoint.target, res)
            return res


        if plot_info['enabled']:
            if hyperparamerter_search:
                for alpha in list(np.linspace(0.5, 5, 20)):
                    for q in [5 * 1e-5, 1e-5, 5 * 1e-4, 1e-4, 5 * 1e-3]:
                        sky_model_new = ErositaSky(config_filename, alpha=alpha, q=q)
                        sky, point_sources, diffuse = sky_model_new.create_sky_model()
                        convolved_sky = gaussian_psf(op=sky, var=var)
                        convolved_ps = gaussian_psf(op=point_sources, var=var)
                        convolved_diffuse = gaussian_psf(op=diffuse, var=var)

                        # Get mock data
                        mock_data = get_data_realization(convolved_sky)
                        mock_ps_data = get_data_realization(convolved_ps)
                        mock_diffuse_data = get_data_realization(convolved_diffuse)

                        # Get mock signal
                        mock_sky = get_data_realization(convolved_sky, data=False)
                        mock_ps = get_data_realization(convolved_ps, data=False)
                        mock_diffuse = get_data_realization(convolved_diffuse, data=False)

                        print(f'Plotting  mock data for alpha = {alpha} and q = {q}.')
                        p = ift.Plot()
                        norm = colors.SymLogNorm(linthresh=10e-8)
                        p.add(mock_ps, title='point sources response', norm=norm)
                        p.add(mock_diffuse, title='diffuse component response', norm=norm)
                        p.add(mock_sky, title='sky', norm=norm)
                        p.add(mock_ps_data, title='mock point source data', norm=norm)
                        p.add(data, title='data', norm=norm)
                        p.add(mock_data, title='mock data', norm=norm)
                        p.add(mock_diffuse_data, title='mock diffuse data', norm=norm)
                        p.output(nx=4, name=f'mock_data_a{alpha}_q{q}.png')

    # Print Exposure norm
    # norm = xu.get_norm(exposure, data)
    # print(norm)

    # Set up likelihood
    if mockrun:
        log_likelihood = ift.PoissonianEnergy(masked_data) @ R @ convolved_sky
    else:
        log_likelihood = ift.PoissonianEnergy(masked_data) @ R @ sky

    # Load minimization config
    minimization_config = cfg['minimization']

    # Minimizers
    ic_newton = ift.AbsDeltaEnergyController(**minimization_config['ic_newton'])
    ic_sampling = ift.AbsDeltaEnergyController(**minimization_config['ic_sampling'])
    ic_sampling_nl = ift.AbsDeltaEnergyController(**minimization_config['ic_sampling_nl'])
    minimizer = ift.NewtonCG(ic_newton)
    minimizer_sampling = ift.NewtonCG(ic_sampling_nl)

    # Prepare results
    operators_to_plot = {'reconstruction': sky, 'point_sources': point_sources, 'diffuse_component': diffuse}

    output_directory = create_output_directory("retreat_first_reconstruction")

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
