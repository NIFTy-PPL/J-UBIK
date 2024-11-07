import pickle
from os.path import join
from pprint import pprint

import matplotlib.pyplot as plt
import nifty8.re as jft
import jubik0 as ju
import numpy as np
from jax import random

from demos.xi_experiments import build_peaks_filter_from_config
from src.mf_utils import get_new_ps_locations_from_peaks
from src.multi_component_sky import build_multi_component_sky_from_config
from src.plot import simple_eval_plots, plot_xi_analysis, plot_xi_relaxation, \
    plot_new_position
from src.select_xis import get_relaxed_excitations, \
    get_multi_component_model_new_position
from src.sugar import save_all_minimization_outputs, \
    clear_jax_compilation_cache, get_detection_maps_from_mc_xi_operators, \
    get_point_source_locations, update_state_with_minimization_paser, \
    save_all_minisanities, get_filtered_xi_maps, get_xi_maps_from_samples, \
    get_detection_maps_from_xi_maps
from src.xi_filters import build_radial_filter

if __name__ == "__main__":
    minimize = True
    if not minimize:
        resume_it = 3

    # Reconstruction config
    config_path = '../configs/'
    config_filename = "erosita_mc_sf.yml"
    config = ju.get_config(join(config_path, config_filename))
    output_directory = ju.create_output_directory(config["files"]["res_dir"])
    key = random.PRNGKey(config['seed'])

    # Copy config file to results directory
    ju.copy_config(config_filename, config_path, output_directory)

    # Load response
    response_dict = ju.build_erosita_response_from_config(config)

    # Generate eROSITA data (if it does not already exist)
    ju.generate_erosita_data_from_config(config)
    ju.create_erosita_data_from_config(config, response_dict)

    # Load masked_data
    masked_data = ju.load_masked_data_from_config(config)

    # Creates eROSITA data
    ju.mask_erosita_data_from_disk(
        config['files'],
        config['telescope'],
        config['grid'],
        response_dict['mask'],
    )

    # Load reconstruction model
    ps_locations = np.empty((0, 2), dtype=np.float32)
    ps_loc_dict = dict(background=ps_locations, )
    sky_model = build_multi_component_sky_from_config(
        config, point_source_positions=ps_loc_dict)

    # Load plotting dict
    plot_dict = sky_model.get_plottable_sky_dict()

    # Load xi operators
    xi_ops = sky_model.get_xi_operators()

    # Plot callback
    def plot_callback(s, x):
        return simple_eval_plots(s, x,
                                 output_directory=output_directory,
                                 plotting_cfg=config['plotting'],
                                 sky_dict=plot_dict,
                                 xi_operators=xi_ops
                                 )

    # Generate response
    def response(x):
        return response_dict['exposure'](
            response_dict['psf'](x * response_dict['pix_area'],
            response_dict['kernel']))


    # Generate loglikelihood (Building masked (mock) data and response)
    log_likelihood = ju.generate_erosita_likelihood(response_dict,
                                                    masked_data,
                                                    sky_model,
                                                    )

    # Minimization
    key, subkey = random.split(key)
    detection_config = config['anomaly_detection']
    pos_init = sky_model.init(subkey) * detection_config['relaxation_factor']
    minimization_config = config['minimization']

    def callback(s, x):
        save_all_minimization_outputs(
            x,
            output_directory=output_directory,
        )
        save_all_minisanities(output_directory)
        plot_callback(s, x)
        clear_jax_compilation_cache(x, clear_every_n_iterations=5)


    if minimize:
        n_dof = ju.get_n_constrained_dof(log_likelihood)
        minimization_parser = ju.MinimizationParser(minimization_config,
                                                    n_dof=n_dof)

        samples, state = jft.optimize_kl(
            log_likelihood,
            pos_init,
            key=key,
            n_total_iterations=minimization_config['n_total_iterations'],
            resume=minimization_config['resume'],
            n_samples=minimization_parser.n_samples,
            draw_linear_kwargs=minimization_parser.draw_linear_kwargs,
            nonlinearly_update_kwargs=minimization_parser.nonlinearly_update_kwargs,
            kl_kwargs=minimization_parser.kl_kwargs,
            sample_mode=minimization_parser.sample_mode,
            callback=callback,
            odir=output_directory)

    else:
        with open(join(output_directory, 'minimization',
                       f'samples_{resume_it}.pkl'), 'rb') as f:
            samples, state = pickle.load(f)

    # FIXME: minimize=True logic
    minimization_2 = config.get("minimization_2")
    if minimization_2 is None:
        exit()
    else:
        minimization_config = minimization_2.copy()

    ################# ANOMALY DETECTION ANALYSIS #################
    old_total_iterations = state.nit
    N_LOOPS = minimization_config["n_loops"]
    N_TOTAL_ITERATIONS = (np.arange(1, N_LOOPS + 1) *
                        minimization_config["n_total_iterations"])
    N_TOTAL_ITERATIONS += old_total_iterations
    for LOOP_ITERATION in range(N_LOOPS):
        # Load detection maps
        xi_maps = get_xi_maps_from_samples(samples, xi_ops)
        detection_maps = get_detection_maps_from_xi_maps(xi_maps)

        # Get peak filter
        kernel, _ = build_radial_filter(
            detection_config['filter']['inner_radius'],
            detection_config['filter']['outer_radius'])
        # FIXME: not used, remove
        # peak_filter = build_peaks_filter_from_config(config)

        # Load filtered detection maps
        filtered_xi_maps = get_filtered_xi_maps(xi_maps, kernel)
        filtered_detection_map = get_detection_maps_from_xi_maps(filtered_xi_maps)
        # plt.imshow(filtered_detection_map, origin="lower")
        # plt.show()

        # Get peaks
        foreground_locs = sky_model.get_foreground_locations()
        peaks, peaks_dict = get_point_source_locations(
            detection_maps,
            foreground_locs,
            threshold_abs=detection_config['threshold_abs'],
            min_distance=detection_config['min_distance'],
            exclude_keys=detection_config.get('exclude_keys'),
        )

        # Get relaxed excitation
        key, relaxed_excitations = get_relaxed_excitations(
            state.key, samples, peaks, peaks_dict, xi_ops,
            relaxation_radius=detection_config['relaxation_radius'],
            relaxation_factor=detection_config['relaxation_factor'],
        )

        # Get new ps locations
        ps_locations = get_new_ps_locations_from_peaks(ps_locations, peaks)
        ps_loc_dict = dict(background=ps_locations, )

        # Build new model
        new_sky_model = build_multi_component_sky_from_config(
            config, point_source_positions=ps_loc_dict)

        # Get new latent position
        key, new_mc_model_position = get_multi_component_model_new_position(
            key, new_sky_model, config, samples, relaxed_excitations, peaks,
            position_rescale_factor=None,)

        # Plot
        xi_analysis_dir = ju.create_output_directory(
            join(output_directory, 'xi_analysis', f'iteration_{state.nit}'))
        xi_relax_dir = ju.create_output_directory(join(xi_analysis_dir,
                                                       'relaxation'))
        plot_xi_analysis(detection_maps, peaks_dict,
                         output_directory=xi_analysis_dir,
                         iteration=state.nit,
                         dpi=config['plotting']['dpi'],)
        plot_xi_relaxation(xi_ops, relaxed_excitations, xi_relax_dir, state.nit,
                           dpi=config['plotting']['dpi'])
        plot_new_position(new_sky_model, response, new_mc_model_position,
                          output_directory=xi_analysis_dir)
        plot_dict = new_sky_model.get_plottable_sky_dict()

        def new_callback(s, x):
            callback(s, x)

        # Save the point source locations
        ps_locations_file = join(xi_analysis_dir, 'ps_locations.npy')
        np.save(ps_locations_file, ps_locations)
        print(f"Position saved to {ps_locations_file}.")

        # Update log_likelihood
        log_likelihood = ju.generate_erosita_likelihood(response_dict,
                                                        masked_data,
                                                        new_sky_model
                                                        )

        # Update minimization
        key, subkey = random.split(key)
        n_dof = ju.get_n_constrained_dof(log_likelihood)
        minimization_config["n_total_iterations"] = N_TOTAL_ITERATIONS[LOOP_ITERATION]
        # FIXME: update_switches
        minimization_parser = ju.MinimizationParser(minimization_config,
                                                    n_dof=n_dof)

        update_state_with_minimization_paser(state, minimization_parser)

        # Minimize
        samples, state = jft.optimize_kl(
            log_likelihood,
            new_mc_model_position,
            key=subkey,
            n_total_iterations=N_TOTAL_ITERATIONS[LOOP_ITERATION],
            resume=minimization_config['resume'],
            n_samples=minimization_parser.n_samples,
            draw_linear_kwargs=minimization_parser.draw_linear_kwargs,
            nonlinearly_update_kwargs=minimization_parser
            .nonlinearly_update_kwargs,
            kl_kwargs=minimization_parser.kl_kwargs,
            sample_mode=minimization_parser.sample_mode,
            callback=new_callback,
            odir=output_directory,
            _optimize_vi_state=state,
            )
