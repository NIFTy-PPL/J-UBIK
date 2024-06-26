import os
import argparse

import nifty8.re as jft
import jubik0 as ju

from jax import config, random

config.update('jax_enable_x64', True)
config.update('jax_platform_name', 'cpu')

# Parser Setup
parser = argparse.ArgumentParser()
parser.add_argument('config', type=str, help="Config file (.yaml) for eROSITA inference.",
                    nargs='?', const=1, default="eROSITA_config.yaml")
args = parser.parse_args()


if __name__ == "__main__":
    # Load config file
    config_path = args.config
    cfg = ju.get_config(config_path)
    file_info = cfg['files']

    if (not cfg['minimization']['resume']) and os.path.exists(file_info["res_dir"]):
        file_info["res_dir"] = file_info["res_dir"] + "_new"
        print("FYI: Resume is set to False, but the output directory already exists. "
              "The result_dir has been appended with the string *new*.")

    # Save run configuration
    ju.save_config_copy(os.path.basename(config_path), output_dir=file_info['res_dir'])
    # ju.save_local_packages_hashes_to_txt(['jubik0', 'nifty8'], # FIXME: fix for cluster
    #                                      join(file_info['res_dir'], "packages_hashes.txt"),
    #                                      paths_to_git=[os.path.dirname(os.getcwd()), None],
    #                                      verbose=False)

    # Load sky model
    sky_model = ju.SkyModel(config_path)
    sky = sky_model.create_sky_model()
    sky_dict = sky_model.sky_model_to_dict()

    # Generate eROSITA data (if it does not alread exist)
    _ = ju.create_erosita_data_from_config(config_path)

    # Generate loglikelihood (Building masked (mock) data and response)
    log_likelihood = ju.generate_erosita_likelihood_from_config(config_path).amend(sky)

    # Set initial position
    key = random.PRNGKey(cfg['seed'])
    key, subkey = random.split(key)
    pos_init = 0.1 * jft.Vector(jft.random_like(subkey, sky.domain))

    # Minimization
    minimization_config = cfg['minimization']
    n_dof = ju.calculate_n_constrained_dof(log_likelihood)
    minimization_parser = ju.MinimizationParser(minimization_config, n_dof=n_dof)

    # Plot
    additional_plot_dict = {"diffuse_alpha": sky_model.alpha_cf,
                            "point_sources_alpha": sky_model.points_alpha}

    def simple_eval_plots(s, x):
        """Call plot_sample_and_stat for every iteration."""
        ju.plot_sample_and_stats(file_info["res_dir"],
                                 sky_dict,
                                 s,
                                 dpi=300,
                                 iteration=x.nit,
                                 rgb_min_sat=[3e-8, 3e-8, 3e-8],
                                 rgb_max_sat=[2.0167e-6, 1.05618e-6, 1.5646e-6])
        ju.plot_sample_and_stats(file_info["res_dir"],
                                 additional_plot_dict,
                                 s,
                                 dpi=300,
                                 iteration=x.nit,
                                 log_scale=False,
                                 plot_samples=False,
                                 )

    samples, state = jft.optimize_kl(log_likelihood,
                                     pos_init,
                                     key=key,
                                     n_total_iterations=minimization_config['n_total_iterations'],
                                     resume=minimization_config['resume'],
                                     n_samples=minimization_parser.n_samples,
                                     draw_linear_kwargs=minimization_parser.draw_linear_kwargs,
                                     nonlinearly_update_kwargs=
                                     minimization_parser.nonlinearly_update_kwargs,
                                     kl_kwargs=minimization_parser.kl_kwargs,
                                     sample_mode=minimization_parser.sample_mode,
                                     callback=simple_eval_plots,
                                     odir=file_info["res_dir"],
                                     )
