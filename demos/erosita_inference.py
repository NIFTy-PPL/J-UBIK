import os
import argparse

from matplotlib.colors import LogNorm
import nifty8 as ift
import xubik0 as xu

from jax import config

config.update('jax_enable_x64', True)

# Parser Setup
parser = argparse.ArgumentParser()
parser.add_argument('config', type=str, help="Config file (.yaml) for eROSITA inference.",
                    nargs='?', const=1, default="eROSITA_config.yaml")
args = parser.parse_args()

if __name__ == "__main__":
    # Load config file
    config_path = args.config
    cfg = xu.get_cfg(config_path)
    ift.random.push_sseq_from_seed(cfg['seed'])

    # Mock reconstruction setup
    mock_run = cfg['mock']
    load_mock_data = cfg['load_mock_data']

    ift.set_nthreads(cfg["threads"])
    print("Set the number of FFT-Threads to:", ift.nthreads())

    if (cfg['minimization']['resume'] and mock_run) and (not load_mock_data):
        raise ValueError(
            'Resume is set to True on mock run. This is only possible if the mock data is loaded '
            'from file. Please set load_mock_data=True')

    if load_mock_data and not mock_run:
        print('WARNING: Mockrun is set to False: Actual data is loaded')

    # File Location
    file_info = cfg['files']

    # Load sky model
    sky_model = xu.SkyModel(config_path)
    sky_dict = sky_model.create_sky_model()

    # Get power spectrum
    pspec = sky_dict.pop('pspec')

    # Create the output directory
    if (not cfg['minimization']['resume']) and os.path.exists(file_info["res_dir"]):
        raise FileExistsError("Resume is set to False but output directory exists already!")
    if xu.mpi.comm is not None:
        xu.mpi.comm.Barrier()
    output_directory = xu.create_output_directory(file_info["res_dir"])
    diagnostics_directory = xu.create_output_directory(output_directory + '/diagnostics')

    # Load response dictionary
    response_dict = xu.load_erosita_response(config_path, diagnostics_directory)

    # Load data
    _, masked_data_dict = xu.load_erosita_data(config_path, output_directory,
                                               diagnostics_directory, response_dict)

    # Set up likelihood
    log_likelihood = None
    for tm_id in cfg['telescope']['tm_ids']:
        tm_key = f'tm_{tm_id}'
        masked_data = masked_data_dict[tm_key]
        R = response_dict[tm_key]['R']
        lh = ift.PoissonianEnergy(masked_data) @ R
        if log_likelihood is None:
            log_likelihood = lh
        else:
            log_likelihood = log_likelihood + lh

    log_likelihood = log_likelihood @ sky_dict['sky']

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

    # strip of directory of filename
    config_filename = os.path.basename(config_path)
    # Save config file in output_directory
    xu.save_cfg(cfg, config_filename, output_directory)

    plot = lambda x, y: xu.plot_sample_and_stats(output_directory,
                                                 operators_to_plot,
                                                 x,
                                                 y,
                                                 plotting_kwargs={'norm': LogNorm()})
    # Initial position
    initial_position = ift.from_random(sky_dict['sky'].domain) * 0.1
    transition = None
    if 'point_sources' in sky_dict:
        initial_ps = ift.MultiField.full(sky_dict['point_sources'].domain, 0)
        initial_position = ift.MultiField.union([initial_position, initial_ps])

        if minimization_config['transition']:
            transition = xu.get_equal_lh_transition(
                sky_dict['sky'],
                sky_dict['diffuse'],
                cfg['priors']['point_sources'],
                minimization_config['ic_transition'])

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
                    initial_position=initial_position,
                    comm=xu.library.mpi.comm)
