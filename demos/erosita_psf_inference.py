import math
import os

from jax import config
import numpy as np

from matplotlib.colors import LogNorm, SymLogNorm
import nifty8 as ift
import xubik0 as xu


config.update('jax_enable_x64', True)

if __name__ == "__main__":
    config_filename = "eROSITA_config_mg.yaml"
    try:
        cfg = xu.get_cfg(config_filename)
    except:
        cfg = xu.get_cfg('demos/' + config_filename)
    fov = cfg['telescope']['fov']
    rebin = math.floor(20 * fov // cfg['grid']['npix']) # FIXME USE DISTANCES!
    mock_run = cfg['mock']

    # File Location
    file_info = cfg['files']
    obs_path = file_info['obs_path']
    input_filenames = file_info['input']
    output_filename = file_info['output']
    exposure_filename = file_info['exposure']
    observation_instance = xu.ErositaObservation(input_filenames, output_filename, obs_path)
    sky_model = xu.SkyModel(config_filename)
    point_sources, diffuse, sky = sky_model.create_sky_model()

    # Grid Info
    grid_info = cfg['grid']
    e_min = grid_info['energy_bin']['e_min']
    e_max = grid_info['energy_bin']['e_max']
    npix = grid_info['npix']

    # Telescope Info
    tel_info = cfg['telescope']
    tm_id = tel_info['tm_id']

    # Bool to enable only diffuse reconstruction
    reconstruct_point_sources = cfg['priors']['point_sources'] is not None

    log = 'Output file {} already exists and is not regenerated. ' \
          'If the observations parameters shall be changed please delete or rename the current output file.'

    if not os.path.exists(os.path.join(obs_path, output_filename)):
        observation = observation_instance.get_data(emin=e_min, emax=e_max, image=True, rebin=rebin,
                                                    size=npix, pattern=tel_info['pattern'],
                                                    telid=tm_id)
    else:
        print(log.format(os.path.join(obs_path, output_filename)))

    observation_instance = xu.ErositaObservation(output_filename, output_filename, obs_path)

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
    psf_file = xu.eROSITA_PSF(cfg["files"]["psf_path"])

    # Places the pointing in the center of the image (or equivalently defines
    # the image to be centered around the pointing).
    dom = sky_model.extended_space
    center = tuple(0.5*ss*dd for ss, dd in zip(dom.shape, dom.distances))

    psf_function = psf_file.psf_func_on_domain('3000', center, sky_model.extended_space)

    psf_kernel = psf_function(*center)
    psf_kernel = ift.makeField(sky_model.extended_space, np.array(psf_kernel))

    convolved_sky = xu.convolve_field_operator(psf_kernel, sky)
    if reconstruct_point_sources:
        convolved_ps = xu.convolve_field_operator(psf_kernel, point_sources)
    convolved_diffuse = xu.convolve_field_operator(psf_kernel, diffuse)

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
    masked_data = mask(data)

    p = ift.Plot()

    if mock_run:
        ift.random.push_sseq_from_seed(cfg['seed'])
        mock_position = ift.from_random(sky.domain)

        # Get mock data
        mock_data = xu.get_data_realization(convolved_sky, mock_position, exposure=exposure_op, padder=sky_model.pad)
        if reconstruct_point_sources:
            mock_ps_data = xu.get_data_realization(convolved_ps, mock_position, exposure=exposure_op, padder=sky_model.pad)
        mock_diffuse_data = xu.get_data_realization(convolved_diffuse, mock_position, exposure=exposure_op,
                                                 padder=sky_model.pad)

        # Mask mock data
        masked_data = mask(mock_data)

        # Get mock signal
        mock_sky = xu.get_data_realization(convolved_sky, mock_position, data=False)
        if reconstruct_point_sources:
            mock_ps = xu.get_data_realization(convolved_ps, mock_position, data=False)
        mock_diffuse = xu.get_data_realization(convolved_diffuse, mock_position, data=False)

        norm = SymLogNorm(linthresh=5e-3)
        if reconstruct_point_sources:
            p.add(mock_ps, title='point sources response', norm=LogNorm())
        p.add(mock_diffuse, title='diffuse component response', norm=LogNorm())
        p.add(mock_sky, title='sky', norm=LogNorm())
        if reconstruct_point_sources:
            p.add(mock_ps_data, title='mock point source data', norm=LogNorm())
        p.add(data, title='data', norm=norm)
        p.add(mock_data, title='mock data', norm=norm)
        p.add(mock_diffuse_data, title='mock diffuse data', norm=LogNorm())
        p.output(nx=4, name=f'mock_data.png')

    # Set up likelihood
    log_likelihood = ift.PoissonianEnergy(masked_data) @ R @ convolved_sky

    # Load minimization config
    minimization_config = cfg['minimization']

    # Minimizers
    ic_newton = ift.AbsDeltaEnergyController(**minimization_config['ic_newton'])
    ic_sampling = ift.AbsDeltaEnergyController(**minimization_config['ic_sampling'])
    ic_sampling_nl = ift.AbsDeltaEnergyController(**minimization_config['ic_sampling_nl'])
    minimizer = ift.NewtonCG(ic_newton)
    minimizer_sampling = ift.NewtonCG(ic_sampling_nl)

    # Prepare results
    operators_to_plot = {'reconstruction': sky_model.pad.adjoint(sky),
                         'diffuse_component': sky_model.pad.adjoint(diffuse)}

    if reconstruct_point_sources:
        operators_to_plot['point_sources'] = sky_model.pad.adjoint(point_sources)

    # Create the output directory
    output_directory = xu.create_output_directory(file_info['res_dir'])

    # Plot the data in output directory
    p.add(data, norm=LogNorm())
    p.output(name=os.path.join(output_directory, 'data.png'), dpi=800)

    # Save config file in output_directory
    xu.save_config(cfg, config_filename, output_directory)

    plot = lambda x, y: xu.plot_sample_and_stats(output_directory, operators_to_plot, x, y,
                                              plotting_kwargs={'norm': LogNorm()})

    # Initial position
    initial_position = ift.from_random(sky.domain) * 0.1
    if reconstruct_point_sources:
        initial_ps = ift.MultiField.full(point_sources.domain, 0)
        initial_position = ift.MultiField.union([initial_position, initial_ps])

    if minimization_config['geovi']:
        # geoVI
        ift.optimize_kl(log_likelihood, minimization_config['total_iterations'],
                        minimization_config['n_samples'],
                        minimizer,
                        ic_sampling,
                        minimizer_sampling,
                        initial_position=initial_position,
                        output_directory=output_directory,
                        export_operator_outputs=operators_to_plot,
                        inspect_callback=plot,
                        resume=True,
                        comm=xu.library.mpi.comm)
    else:
        # MGVI
        ift.optimize_kl(log_likelihood, minimization_config['total_iterations'],
                        minimization_config['n_samples'],
                        minimizer,
                        ic_sampling,
                        None,
                        export_operator_outputs=operators_to_plot,
                        output_directory=output_directory,
                        inspect_callback=plot,
                        resume=True,
                        comm=xu.library.mpi.comm)
