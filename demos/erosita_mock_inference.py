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
hyperparamerter_search = True
if __name__ == "__main__":
    config_filename = "eROSITA_config_mw.yaml"
    try:
        cfg = xu.get_cfg(config_filename)
    except:
        config_filename = 'demos/' + config_filename
        cfg = xu.get_cfg(config_filename)
    fov = cfg['telescope']['fov']
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
    center = observation_instance.get_center_coordinates(output_filename)
    if mockrun:
        def gaussian_psf(sky_space, var):
            dist_x = sky_space.distances[0]
            dist_y = sky_space.distances[1]

            # Periodic Boundary conditions
            x_ax = np.arange(sky_space.shape[0])
            x_ax = np.minimum(x_ax, sky_space.shape[0] - x_ax) * dist_x
            y_ax = np.arange(sky_space.shape[1])
            y_ax = np.minimum(y_ax, sky_space.shape[1] - y_ax) * dist_y

            center = (0, 0)
            x_ax -= center[0]
            y_ax -= center[1]
            X, Y = np.meshgrid(x_ax, y_ax, indexing='ij')

            var *= sky_space.scalar_dvol  # ensures that the variance parameter is specified with respect to the

            # normalized psf
            log_psf = - (0.5 / var) * (X ** 2 + Y ** 2)
            log_kernel = ift.makeField(sky_space, log_psf)
            gaussian = log_kernel.exp()
            norm = ift.ScalingOperator(sky_space, gaussian.integrate().val ** -1)
            psf = norm(gaussian)

            # p = ift.Plot()
            # import matplotlib.colors as colors
            # p.add(log_kernel.exp(), norm=colors.SymLogNorm(linthresh=10e-8))
            # p.output(nx=1)
            mock_sky_position = ift.from_random(sky.domain)
            conv = xu.convolve_field_operator(psf, sky)
            return conv

        convolved = gaussian_psf(sky_space=sky_model.extended_space, var=5)
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
        if hyperparamerter_search:
            n_mock_samples = 1
            for n in range(n_mock_samples):
                for alpha in [1.0001]:
                    for q in [0.0000001]:
                        # ift.random.push_sseq_from_seed(cfg['seed'])
                        sky_model = ErositaSky(config_filename, alpha=alpha, q=q)
                        point_sources, diffuse, sky = sky_model.create_sky_model()
                        convolved = gaussian_psf(sky_space=sky_model.extended_space, var=2)
                        mock_sky_position = ift.from_random(sky.domain)
                        mock_sky = sky(mock_sky_position)
                        conv_mock_sky = convolved(mock_sky_position)
                        mock_points = point_sources.force(mock_sky_position)
                        mock_diffuse = diffuse.force(mock_sky_position)

                        # Mock data for point sources without convolution
                        mock_points_data = np.random.poisson(exposure_op(mock_points).val.astype(np.float64))
                        mock_points_data = sky_model.pad.adjoint(ift.Field.from_raw(sky_model.extended_space,
                                                                                    mock_points_data))

                        # Mock data for diffuse sources without convolution
                        mock_diffuse_data = np.random.poisson(exposure_op(mock_diffuse).val.astype(np.float64))
                        mock_diffuse_data = sky_model.pad.adjoint(ift.Field.from_raw(sky_model.extended_space,
                                                                                     mock_diffuse_data))

                        # Mock data for whole sky including convolution
                        mock_sky_data_conv = ift.Field.from_raw(sky_model.extended_space,
                                                                np.random.poisson(exposure_op(conv_mock_sky).val.astype(np.float64)))
                        mock_sky_data_conv = sky_model.pad.adjoint(mock_sky_data_conv)

                        mock_sky_data = ift.Field.from_raw(sky_model.extended_space,
                                                           np.random.poisson(exposure_op(mock_sky).val.astype(np.float64)))
                        mock_sky_data = sky_model.pad.adjoint(mock_sky_data)

                        masked_data = mask(mock_sky_data_conv)

                        print(f'Plotting  mock data for alpha = {alpha} and q = {q}.')
                        p = ift.Plot()
                        #  vmin=0, vmax=10e5
                        p.add(data, title='data', norm=colors.SymLogNorm(linthresh=10e-10))
                        p.add(mock_sky_data_conv, title='Mock data sky (conv)', norm=colors.SymLogNorm(linthresh=10e-10))
                        p.add(mock_sky_data, title='Mock data sky', norm=colors.SymLogNorm(linthresh=10e-10))
                        p.add(mock_points_data, title='Mock data points', norm=colors.SymLogNorm(linthresh=10e-10))
                        p.add(mock_diffuse_data, title='Mock data diffuse', norm=colors.SymLogNorm(linthresh=10e-10))
                        p.add(mock_sky, title='mock_sky', norm=colors.SymLogNorm(linthresh=10e-10))
                        p.add(exposure_field, title='exposure', norm=colors.SymLogNorm(linthresh=10e-10))
                        p.output(nx=3, name=f'mock_data_a{alpha}_q{q}_sample{n}.png')
            exit()
        else:
            sky_model = ErositaSky(config_filename, alpha=alpha, q=q)
            point_sources, diffuse, sky = sky_model.create_sky_model()
            convolved = gaussian_psf(sky_space=sky_model.extended_space, var=2)
            mock_sky_position = ift.from_random(sky.domain)
            conv_mock_sky = convolved(mock_sky_position)
            mock_sky_data_conv = ift.Field.from_raw(sky_model.extended_space,
                                                    np.random.poisson(
                                                        exposure_op(conv_mock_sky).val.astype(np.float64)))
            mock_sky_data_conv = sky_model.pad.adjoint(mock_sky_data_conv)
            masked_data = mask(mock_sky_data_conv)

            #Likelihood
            log_likelihood = ift.PoissonianEnergy(masked_data) @ R @ convolved
    else:
        log_likelihood = ift.PoissonianEnergy(masked_data) @ R @ sky

    # Print Exposure norm
    # norm = xu.get_norm(exposure, data)
    # print(norm)


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
