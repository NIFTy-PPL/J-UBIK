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
    cfg = xu.get_config(config_path)
    file_info = cfg['files']
    ift.random.push_sseq_from_seed(cfg['seed'])

    # ift.set_nthreads(cfg["threads"])
    # print("Set the number of FFT-Threads to:", ift.nthreads())

    # Sanity Checks
    if (cfg['minimization']['resume'] and cfg['mock']) and (not cfg['load_mock_data']):
        raise ValueError(
            'Resume is set to True on mock run. This is only possible if the mock data is loaded '
            'from file. Please set load_mock_data=True')

    if cfg['load_mock_data'] and not cfg['mock']:
        print('WARNING: Mockrun is set to False: Actual data is loaded')

    if (not cfg['minimization']['resume']) and os.path.exists(file_info["res_dir"]):
        raise FileExistsError("Resume is set to False but output directory exists already!")

    # Load sky model
    sky_dict = xu.create_sky_model_from_config(config_path)
    pspec = sky_dict.pop('pspec')

    # Create the output directory

    if xu.mpi.comm is not None:
        xu.mpi.comm.Barrier()
    output_directory = xu.create_output_directory(file_info["res_dir"])
    diagnostics_directory = xu.create_output_directory(output_directory + '/diagnostics')

    # Load response dictionary
    response_dict = xu.load_erosita_response(config_path, diagnostics_directory)

    # Load data
    _, masked_data_dict = xu.load_erosita_data(config_path, output_directory,
                                               diagnostics_directory, response_dict)

    # Generate loglikelihood
    log_likelihood = xu.generate_erosita_likelihood_from_config(config_path) @ sky_dict['sky']

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
    operators_to_plot = {**sky_dict, 'pspec': pspec}
    xu.save_config(cfg, os.path.basename(config_path), output_directory)

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
