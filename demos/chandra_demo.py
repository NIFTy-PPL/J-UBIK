# %% [markdown]
# # Pipeline Demo: Chandra
#
# This demo showcases the application of j-ubik to Chandra data. Most parameters are set in `chandra_demo.yaml`, which is located in `demos/configs`, here a [link to the file](https://github.com/NIFTy-PPL/J-UBIK/blob/main/demos/configs/chandra_demo.yaml)
#
# ## Requirements
# Before running this demo `J-UBIK` needs to be installed properly, see README for help. After that 
# - Install [ciao & marx](https://cxc.cfa.harvard.edu/ciao/download/conda.html). We recommend installation of both via conda / conda-forge
# - Download Chandra data with the terminal command "download_chandra_obsid" or use [Chaser](https://cda.harvard.edu/chaser/). Don't forget to unpack the data.

# %%
import argparse
import os

from jax import random

import nifty.re as jft
import jubik as ju

# %% [markdown]
# ## YAML Configuration File Structure
# In the chandra_demo.yaml config file, you need to define several parameters based on your observation data and settings. 
# 
# Below is a breakdown of key settings:
#
# - **seed**: Random seed for generating reproducible results.
# - **obs_info**: Information about all the filenames needed for building the data and the response:
#   - `obsID`: ID of the Chandra observation
#   - `data_location`: directory of the downloaded observation
#   - `event_file`: path to the event file
#   - `aspect_sol`: path to aspect solution file
#   - `bpix_file`: path to bad pixel file
#   - `mask_file`: path to mask file
#   - `instrument`: specify if ACIS-S or ACIS-I, also used for PSF simulation
# - **grid**: Parameters defining the data grid for processing.
#   - `sdim`: Spatial dimensions for the image grid.
#   - `e_dim`: number of energy bins
#   - `energy_bin`: Energy binning with `e_min`, `e_max`, and `e_ref` values
#   for each bin.
# - **telescope**: general information about the x-ray observatory
#   - `fov`: field of view in arcsec
#   - `exp_cut`: minimal exposure time, to prevent artifacts
#   - `center_obs_id`: obsID which implicitely sets the central pointing
# - **files**:
#   - `data_dict`: Name of the pickle file storing the processed observation data
#   (e.g., 'data.pkl').
#   - `processed_obs_folder`: Directory where processed observation files are
#   stored (e.g., 'processed'). Only needed for real data.
#   - `res_dir`: Directory where the results of the run will be saved
#   (e.g., 'results/my_results'). Since the final processed data will be
#   saved here, a new res_dir name should be chosen if the minimization parameter
#   `resume` is set to `False`.
# - In case you want to work on simulated data add the following keywords:
#     -   mock_gen_config: path to prior config which is used to generate a simulated sky
#     -   pos_dict: path to the .pkl-file, where the latent parameters of the simulated sky will be saved to.
# - **psf**:
#   - `num_rays`: number of rays set for the marx simulation of the psf
#   - `n_patch`: number of patches for the linear patched convolution / psf approximation
#   - `margfrac`: percentage of the patch size used for padding to break periodic boundary conditions.
#
# - **plotting**: Settings for output plots.
#   - `enabled`: Enable or disable plotting of data.
#   - `slice`: Specify the slice of data to plot (optional).
#   - `dpi`: DPI setting for plot resolution.
#


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
# A description of the prior search can be found in <br>
#
# [The first spatio-spectral Bayesian imaging of SN1006 in X-rays](https://doi.org/10.1051/0004-6361/202347750) <br> 
#
# For all the information about the minimization part of the config please look into <br>
#
# [Re-Envisioning Numerical Information Field Theory (NIFTy.re)](https://doi.org/10.21105/joss.06593) <br> 
#
# or the <br>
#
# [Intoduction notebooks of NIFTy](https://ift.pages.mpcdf.de/nifty/user/notebooks_re/notebooks.html)
#
# To configure the Sky models used add the following to your config.yaml
#
# - **priors**: Parameters defining the prior distributions for the SkyModel.
#   - **point_sources**: Prior parameters for point sources in the sky model.
#     - `spatial`:
#       - `alpha`: The `alpha` parameter in the Inverse-Gamma distribution.
#       - `q`: The `q` parameter in the Inverse-Gamma distribution.
#       - `key`: A unique identifier for the point sources (e.g., "points").
#     - `plaw`:
#       - `mean`: Mean value for the power-law distribution of point sources.
#       - `std`: Standard deviation for the power-law distribution.
#       - `name`: A prefix for naming the point source power-law component.
#
#   - **diffuse**: Prior parameters for the diffuse emission component.
#     - `spatial`:
#       - `offset`:
#         - `offset_mean`: Mean value for the offset in the spatial component of
#         diffuse emission.
#         - `offset_std`: Standard deviation for the offset.
#       - `fluctuations`: Parameters controlling fluctuations in the diffuse
#       spatial emission.
#         - `fluctuations`: List of fluctuation priors (mean, std).
#         - `loglogavgslope`: List of log-log average slope of fluctuations (mean, std).
#         - `flexibility`: List of flexibility priors (mean, std).
#         - `asperity`: Controls the roughness of the spatial power spectrum
#         (set to `Null` if not used).
#         - `non_parametric_kind`: Specifies the type of non-parametric model
#         used for fluctuations (e.g., "power").
#       - `prefix`: A prefix used for naming the diffuse spatial component.
#
#     - `plaw`: Power-law prior for the diffuse emission component.
#     Similar as for the diffuse component.


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


# %% [markdown]
# ## Minimization
#
# - **minimization**: Settings for the minimization algorithm used for likelihood
# estimation.
#   - `resume`: Whether to resume a previous minimization run.
#   - `n_total_iterations`: Total number of iterations for minimization.
#   - Additional parameters to control the sampling and KL-divergence
#   calculations.
#
# %%
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