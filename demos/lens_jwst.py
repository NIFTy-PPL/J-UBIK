import argparse
from sys import exit
from os.path import join

import jax.numpy as jnp
import nifty8.re as jft
import numpy as np
from astropy import units as u
from charm_lensing.lens_system import build_lens_system
from jax import config, random
from matplotlib.colors import LogNorm

import jubik0 as ju
from jubik0.grid import Grid

from jubik0.instruments.jwst.config_handler import (
    insert_spaces_in_lensing_new,
    load_yaml_and_save_info,
)
from jubik0.instruments.jwst.jwst_likelihoods import build_jwst_likelihoods
from jubik0.instruments.jwst.plotting.plotting import get_plot, plot_prior
from jubik0.likelihood import connect_likelihood_to_model
from jubik0.parse.grid import GridModel
from jubik0.minimization.minimization_from_samples import (
    minimization_from_initial_samples,
    KLSettings,
)

SKY_KEY = "sky"
SKY_UNIT = u.MJy / u.sr
config.update("jax_enable_x64", True)

parser = argparse.ArgumentParser()
parser.add_argument(
    "config",
    help="Config File",
    type=str,
    nargs="?",
    const=1,
    default="./demos/configs/spt0418.yaml",
)
args = parser.parse_args()
config_path = args.config

cfg, results_directory = load_yaml_and_save_info(config_path)

if cfg["cpu"]:
    from jax import devices

    config.update("jax_default_device", devices("cpu")[0])

if cfg["no_interactive_plotting"]:
    import matplotlib

    matplotlib.use("Agg")

grid = Grid.from_grid_model(GridModel.from_yaml_dict(cfg["sky"]["grid"]))

# insert_ubik_energy_in_lensing(cfg, zsource=4.2)
insert_spaces_in_lensing_new(cfg["sky"])
lens_system = build_lens_system(cfg["sky"])


if cfg["nonparametric_lens"]:
    sky_model = lens_system.get_forward_model_full()
    sky_model_parametric = lens_system.get_forward_model_parametric_source()
    parametric_lens_flag = False
else:
    sky_model = lens_system.get_forward_model_parametric()
    sky_model_parametric = lens_system.get_forward_model_parametric_source(
        parametric_lens=True
    )
    parametric_lens_flag = True
sky_model = jft.Model(jft.wrap_left(sky_model, SKY_KEY), domain=sky_model.domain)
sky_model_parametric = jft.Model(
    jft.wrap_left(sky_model_parametric, SKY_KEY), domain=sky_model_parametric.domain
)

# # For testing
# sky_model = build_nifty_mf_from_grid(
#     grid,
#     'test',
#     cfg['sky']['model']['source']['light']['multifrequency']['nifty_mf'],
#     reference_bin=grid_model.color_reference_bin,
# )

likelihood_raw, filter_projector, data_dict = build_jwst_likelihoods(
    cfg, grid, sky_model, sky_key=SKY_KEY, sky_unit=SKY_UNIT
)


sky_model_with_keys = jft.Model(
    lambda x: filter_projector(sky_model(x)), init=sky_model.init
)
likelihood = connect_likelihood_to_model(likelihood_raw, sky_model_with_keys)

smwk_parametric = jft.Model(
    lambda x: filter_projector(sky_model_parametric(x)), init=sky_model_parametric.init
)
likelihood_parametric = connect_likelihood_to_model(likelihood_raw, smwk_parametric)

plot_source, plot_residual, plot_lens = get_plot(
    results_directory,
    grid,
    lens_system,
    filter_projector,
    data_dict,
    sky_model_with_keys,
    parametric_lens_flag,
)

_, plot_residual_parametric, plot_lens_parametric = get_plot(
    join(results_directory, "parametric"),
    grid,
    lens_system,
    filter_projector,
    data_dict,
    smwk_parametric,
    parametric_lens_flag,
    True,
)


if cfg.get("prior_samples"):
    plot_prior(
        config_path,
        likelihood,
        filter_projector,
        # plot_source, plot_lens, plot_color,
        plot_residuals=True,
        plot_source=False,
        plot_lens=False,
        data_dict=data_dict,
        parametric_flag=parametric_lens_flag,
        sky_key=SKY_KEY,
    )


def plot(samples: jft.Samples, state: jft.OptimizeVIState):
    print(f"Plotting: {state.nit}")
    if cfg["plot_results"]:
        plot_residual(samples, state)
        plot_lens(samples, state)
        plot_source(samples, state)


def plot_parametric(samples: jft.Samples, state: jft.OptimizeVIState):
    print(f"Plotting: {state.nit}")
    if cfg["plot_results"]:
        plot_residual_parametric(samples, state)
        plot_lens_parametric(samples, state)


cfg_mini = ju.get_config(config_path)["minimization"]
n_dof = ju.get_n_constrained_dof(likelihood)
mini_parser = ju.MinimizationParser(cfg_mini, n_dof, verbose=False)

kl_settings_parametric = KLSettings(
    random_key=random.PRNGKey(cfg_mini.get("key", 42)),
    outputdir=join(results_directory, "parametric"),
    minimization=mini_parser,
    n_total_iterations=3,
    callback=plot_parametric,
    constants=[p for p in likelihood_parametric.domain.tree if "nifty_mf" in p],
    point_estimates=[p for p in likelihood_parametric.domain.tree if "nifty_mf" in p],
    resume=True,  # cfg_mini.get("resume", False),
)

kl_settings = KLSettings(
    random_key=random.PRNGKey(cfg_mini.get("key", 42)),
    outputdir=results_directory,
    minimization=mini_parser,
    n_total_iterations=cfg_mini["n_total_iterations"],
    callback=plot,
    resume=cfg_mini.get("resume", False),
)


samples_parametric, state_parametric = minimization_from_initial_samples(
    likelihood_parametric, kl_settings_parametric, None
)

samples, state = minimization_from_initial_samples(
    likelihood, kl_settings, samples_parametric
)
