import math
import os
import pickle
import numpy as np

from matplotlib.colors import LogNorm
import nifty8 as ift
import xubik0 as xu

from jax import config

config.update('jax_enable_x64', True)

if __name__ == "__main__":
    config_filename = "eROSITA_config.yaml"
    cfg = xu.get_cfg(config_filename)
    fov = cfg['telescope']['fov']
    rebin = math.floor(20 * fov // cfg['grid']['npix'])  # FIXME USE DISTANCES!
    mock_run = cfg['mock']
    mock_psf = cfg['mock_psf']
    load_mock_data = cfg['load_mock_data']
    if load_mock_data and not mock_run:
        print('WARNING: Mockrun is set to False: Actual data is loaded')

    # File Location
    file_info = cfg['files']
    obs_path = file_info['obs_path']
    input_filenames = file_info['input']
    output_filename = file_info['output']
    exposure_filename = file_info['exposure']
    observation_instance = xu.ErositaObservation(input_filenames, output_filename, obs_path)
    sky_model = xu.SkyModel(config_filename)
    sky_dict = sky_model.create_sky_model()
    pspec = sky_dict.pop('pspec')

    # Grid Info
    grid_info = cfg['grid']
    e_min = grid_info['energy_bin']['e_min']
    e_max = grid_info['energy_bin']['e_max']
    npix = grid_info['npix']

    # Telescope Info
    tel_info = cfg['telescope']
    tm_id = tel_info['tm_id']

    # Create the output directory
    if (not cfg['minimization']['resume']) and os.path.exists(file_info["res_dir"]):
        raise FileExistsError("Resume is set to False but output directory exists already!")
    output_directory = xu.create_output_directory(file_info["res_dir"])

    log = 'Output file {} already exists and is not regenerated. ' \
          'If the observations parameters shall be changed please delete or rename the current output file.'

    if not os.path.exists(os.path.join(obs_path, output_filename)):
        observation = observation_instance.get_data(emin=e_min, emax=e_max, image=True, rebin=rebin,
                                                    size=npix, pattern=tel_info['pattern'],
                                                    telid=tm_id)  # FIXME: exchange rebin by fov? 80 = 4arcsec
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
    if cfg['psf']['method'] in ['MSC', 'LIN']:
        center = observation_instance.get_center_coordinates(output_filename)
        psf_file = xu.eROSITA_PSF(cfg["files"]["psf_path"])  # FIXME: load from config

        dom = sky_model.extended_space
        center = tuple(0.5*ss*dd for ss, dd in zip(dom.shape, dom.distances))

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
    else:
        raise NotImplementedError

    # Convolution
    conv_sky_dict = {key: (conv_op @ value) for key, value in sky_dict.items()}

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
    if mock_run:
        ift.random.push_sseq_from_seed(cfg['seed'])
        if load_mock_data:
            # FIXME: name of output folder for diagnostics into config
            # FIXME: Put Mockdata to a better place
            with open('diagnostics/mock_sky_data.pkl', "rb") as f:
                mock_data = pickle.load(f)
        else:
            mock_data_dict = xu.generate_mock_data(sky_model, conv_op, exposure_field,
                                                       sky_model.pad,
                                                       output_directory=output_directory)
            mock_data = mock_data_dict['mock_data_sky']

        # Mask mock data
        masked_data = mask(mock_data)
    else:
        data = observation_instance.load_fits_data(output_filename)[0].data
        data = np.array(data, dtype = int)
        data = ift.makeField(sky_model.position_space, data)
        masked_data = mask(data)

    # Print Exposure norm
    # norm = xu.get_norm(exposure, data)
    # print(norm)

    # Set up likelihood
    log_likelihood = ift.PoissonianEnergy(masked_data) @ R @ conv_sky_dict['sky']

    # Load minimization config
    minimization_config = cfg['minimization']

    # Minimizers
    comm = xu.library.mpi.comm
    if comm is not None:
        if not xu.library.mpi.master:
            minimization_config['ic_newton']['name'] = None
        minimization_config['ic_sampling']['name'] += f"({comm.Get_rank()})"
        minimization_config['ic_sampling_nl']['name'] += f"({comm.Get_rank()})"
    ic_newton = ift.AbsDeltaEnergyController(**minimization_config['ic_newton'])
    ic_sampling = ift.AbsDeltaEnergyController(**minimization_config['ic_sampling'])
    minimizer = ift.NewtonCG(ic_newton)
    if minimization_config['geovi']:
        ic_sampling_nl = ift.AbsDeltaEnergyController(**minimization_config['ic_sampling_nl'])
        minimizer_sampling = ift.NewtonCG(ic_sampling_nl)
    else:
        minimizer_sampling = None

    # Prepare results
    operators_to_plot = {key: (sky_model.pad.adjoint(value)) for key, value in sky_dict.items()}
    operators_to_plot = {**operators_to_plot, 'pspec': pspec}


    # Save config file in output_directory
    xu.save_config(cfg, config_filename, output_directory)

    plot = lambda x, y: xu.plot_sample_and_stats(output_directory,
                                                 operators_to_plot,
                                                 x,
                                                 y,
                                                 plotting_kwargs={'norm': LogNorm()})
    # Initial position
    initial_position = ift.from_random(sky_dict['sky'].domain) * 0.1
    if 'point_sources' in sky_dict:
        initial_ps = ift.MultiField.full(sky_dict['point_sources'].domain, 0)
        initial_position = ift.MultiField.union([initial_position, initial_ps])

    if minimization_config['transition']:
        transition = xu.get_equal_lh_transition(
            sky_dict['sky'],
            sky_dict['diffuse'],
            cfg['priors']['point_sources'],
            minimization_config['ic_transition'])
    else:
        transition = None

    ift.optimize_kl(log_likelihood, minimization_config['total_iterations'],
                    minimization_config['n_samples'],
                    minimizer,
                    ic_sampling,
                    minimizer_sampling,
                    transitions=transition,
                    output_directory=output_directory,
                    export_operator_outputs=operators_to_plot,
                    inspect_callback=plot,
                    resume=minimization_config['resume'],
                    comm=xu.library.mpi.comm)