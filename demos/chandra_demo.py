# %% [markdown]
# # Pipeline Demo: Chandra
#
# This demo showcases the application of j-ubik to Chandra data. Most parameters are set in `chandra_demo.yaml`, which is located in `demos/configs`.
#
# Before running this demo jubik needs to be installed properly, see README for help. After that you'll need some data to apply it to. <br>
# Consider using "download_chandra_obsid" terminal command from ciao or Chaser to do so (https://cda.harvard.edu/chaser/). Don't forget to unpack the data.

# %%
import argparse
import os

from jax import random

import nifty.re as jft
import jubik0 as ju

# %% [markdown]
# ## Config
#
# In the config you should then specify according to the observation:
#
# - The files:obs_path: The path where, the downloaded data is located
# - The obsinfo: Information about all the filenames needed for building the data and the response
# - In case you want to work on simulated data add the following keywords:
#     -   mock_gen_config: path to prior config which is used to generate a simulated sky
#     -   pos_dict: path to the .pkl-file, where the latent parameters of the simulated sky will be saved to.

# %%
# Parser Setup (only for python interpreter, non ipython/jupyter)
parser = argparse.ArgumentParser()
parser.add_argument(
    "config",
    type=str,
    help="Config file (.yaml) for Chandra inference.",
    nargs="?",
    const=1,
    default="configs/chandra_demo.yaml",
)
args = parser.parse_args()
config_path = args.config


# %%
# For ipython / jupyter
# config_path = "configs/chandra_demo.yaml"

# %%
# Load config file
cfg = ju.get_config(config_path)
file_info = cfg["files"]

# %%
# Save run configuration
ju.copy_config(
    os.path.basename(config_path),
    path_to_yaml_file=os.path.dirname(config_path),
    output_dir=file_info["res_dir"],
)

# %% [markdown]
# ## Sky Model
# For the sake of a suitable reconstruction do the following:
# - adjust the energy_ranges (energy_bin:) and <br> the number of pixels according (sdim, edim) according to the desired
#     resolution
# - adjust the priors according (priors) according to the object looked at.
#
# A description of the prior search can be found at (https://doi.org/10.1051/0004-6361/202347750) <br> For all the information about the minimization part of the config
# please look at (https://doi.org/10.21105/joss.06593) or the intoduction notebooks of NIFTy: (https://ift.pages.mpcdf.de/nifty/user/notebooks_re/notebooks.html)

# %%
# Load sky model
sky_model = ju.SkyModel(cfg)
sky = sky_model.create_sky_model()
sky_dict = sky_model.sky_model_to_dict()

# %%
# Generate loglikelihood (Building masked (mock) data and response)
log_likelihood = ju.generate_chandra_likelihood_from_config(cfg).amend(sky)

# %%
# Set initial position
key = random.PRNGKey(cfg["seed"])
key, subkey = random.split(key)
pos_init = 0.1 * jft.Vector(jft.random_like(subkey, sky.domain))

# %%
# Minimization
minimization_config = cfg["minimization"]
n_dof = ju.get_n_constrained_dof(log_likelihood)
minimization_parser = ju.MinimizationParser(minimization_config, n_dof=n_dof)

# %%
# Plot
additional_plot_dict = {}
if hasattr(sky_model, "alpha_cf"):
    additional_plot_dict["diffuse_alpha"] = sky_model.alpha_cf
if hasattr(sky_model, "points_alfa"):
    additional_plot_dict["points_alpha"] = sky_model.points_alfa


# %%
def simple_eval_plots(s, x):
    """Call plot_sample_and_stat for every iteration."""
    ju.plot_sample_and_stats(
        file_info["res_dir"],
        sky_dict,
        s,
        dpi=300,
        iteration=x.nit,
        rgb_min_sat=[3e-8, 3e-8, 3e-8],
        rgb_max_sat=[2.0167e-6, 1.05618e-6, 1.5646e-6],
    )
    ju.plot_sample_and_stats(
        file_info["res_dir"],
        additional_plot_dict,
        s,
        dpi=300,
        iteration=x.nit,
        log_scale=False,
        plot_samples=False,
    )
    ju.plot_pspec(
        sky_model.spatial_pspec,
        sky_model.spatial_cf.target.shape,
        sky_model.s_distances,
        s,
        file_info["res_dir"],
        iteration=x.nit,
        dpi=300,
    )


# %%
samples, state = jft.optimize_kl(
    log_likelihood,
    pos_init,
    key=key,
    n_total_iterations=minimization_config["n_total_iterations"],
    resume=minimization_config["resume"],
    n_samples=minimization_parser.n_samples,
    draw_linear_kwargs=minimization_parser.draw_linear_kwargs,
    nonlinearly_update_kwargs=minimization_parser.nonlinearly_update_kwargs,
    kl_kwargs=minimization_parser.kl_kwargs,
    sample_mode=minimization_parser.sample_mode,
    callback=simple_eval_plots,
    odir=file_info["res_dir"],
)