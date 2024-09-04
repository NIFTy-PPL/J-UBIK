import os
import argparse

import nifty8.re as jft
import jubik0 as ju

from jax import config, random

config.update('jax_enable_x64', True)

# Parser Setup
parser = argparse.ArgumentParser()
parser.add_argument('config', type=str, help="Config file (.yaml) for Chandra inference.",
                    nargs='?', const=1, default="chandra_config.yaml")
args = parser.parse_args()

"""
CHANDRA DEMO:
------------

This demo can be modified easily by the user modifying the chandra_config.yaml
which is located in demos/ as well.

Before running this demo the following installations and downloads have to be
done:

    - Install ciao (https://cxc.cfa.harvard.edu/ciao/threads/ciao_install_tool/)
    - Install marx (https://cxc.cfa.harvard.edu/ciao/ahelp/install_marx.html)
    - Download Chandra data (https://cda.harvard.edu/chaser/)

In the config you should then specify according to the observation:
    - The files:obs_path: The path where, the downloaded data is located
    - The obsinfo: Information about all the filenames needed for building
                   the data and the repsonse
    - Add the following paths if you want to work on simulated data:
        -   mock_gen_config: path to prior config which is used to generate
                             a simulated sky
        -   pos_dict: path to the .pkl-file, where the latent parameters
                      of the simulated sky will be saved to.
For the sake of a suitable reconstruction do the following:
    - adjust the energy_ranges (energy_bin:) and 
    the number of pixels according (sdim, edim) according to the desired resolution
    - adjust the priors according (priors) according to the object looked at.
    A description of the prior search can be found at 
    (https://arxiv.org/abs/2308.09176)
    - For all the information about the minimization part of the config
    please look at (https://arxiv.org/abs/2402.16683)
"""


if __name__ == "__main__":
    # Load config file
    config_path = args.config
    cfg = ju.get_config(config_path)
    file_info = cfg['files']

    if ((not cfg['minimization']['resume']) and
            os.path.exists(file_info["res_dir"])):
        file_info["res_dir"] = file_info["res_dir"] + "_new"
        print("FYI: Resume is set to False, but the output "
              "directory already exists. "
              "The result_dir has been appended with the string *new*.")

    # Uncomment to save local packages git hashes to file
    # ju.save_local_packages_hashes_to_txt(
    #     ['jubik0', 'nifty8'],
    #     os.path.join(file_info['res_dir'], "packages_hashes.txt"),
    #     paths_to_git=[os.path.dirname(os.getcwd()), None],
    #     verbose=False)

    # Save run configuration
    ju.save_config_copy(os.path.basename(config_path),
                        output_dir=file_info['res_dir'])

    # Load sky model
    sky_model = ju.SkyModel(config_path)
    sky = sky_model.create_sky_model()
    sky_dict = sky_model.sky_model_to_dict()

    # Generate loglikelihood (Building masked (mock) data and response)
    log_likelihood = ju.generate_chandra_likelihood_from_config(config_path).\
        amend(sky)

    # Set initial position
    key = random.PRNGKey(cfg['seed'])
    key, subkey = random.split(key)
    pos_init = 0.1 * jft.Vector(jft.random_like(subkey, sky.domain))

    # Minimization
    minimization_config = cfg['minimization']
    n_dof = ju.calculate_n_constrained_dof(log_likelihood)
    minimization_parser = ju.MinimizationParser(minimization_config,
                                                n_dof=n_dof)

    # Plot
    additional_plot_dict = {}
    if hasattr(sky_model, 'alpha_cf'):
        additional_plot_dict['diffuse_alpha'] = sky_model.alpha_cf
    if hasattr(sky_model, 'points_alfa'):
        additional_plot_dict['points_alpha'] = sky_model.points_alfa

    def simple_eval_plots(s, x):
        """Call plot_sample_and_stat for every iteration."""
        ju.plot_sample_and_stats(file_info["res_dir"],
                                 sky_dict,
                                 s,
                                 dpi=300,
                                 iteration=x.nit,
                                 rgb_min_sat=[3e-8, 3e-8, 3e-8],
                                 rgb_max_sat=[2.0167e-6, 1.05618e-6,
                                              1.5646e-6])
        ju.plot_sample_and_stats(file_info["res_dir"],
                                 additional_plot_dict,
                                 s,
                                 dpi=300,
                                 iteration=x.nit,
                                 log_scale=False,
                                 plot_samples=False,
                                 )
        ju.plot_pspec(sky_model.spatial_pspec,
                      sky_model.spatial_cf.target.shape,
                      sky_model.s_distances,
                      s,
                      file_info["res_dir"],
                      iteration=x.nit,
                      dpi=300,
                      )

    samples, state = jft.optimize_kl(log_likelihood,
                                     pos_init,
                                     key=key,
                                     n_total_iterations=minimization_config\
                                         ['n_total_iterations'],
                                     resume=minimization_config['resume'],
                                     n_samples=minimization_parser.n_samples,
                                     draw_linear_kwargs=minimization_parser.\
                                     draw_linear_kwargs,
                                     nonlinearly_update_kwargs=
                                     minimization_parser.\
                                     nonlinearly_update_kwargs,
                                     kl_kwargs=minimization_parser.kl_kwargs,
                                     sample_mode=minimization_parser.\
                                     sample_mode,
                                     callback=simple_eval_plots,
                                     odir=file_info["res_dir"],
                                     )