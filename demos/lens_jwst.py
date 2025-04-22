import argparse
from sys import exit
from os.path import join

import jax
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
    copy_and_replace_light_model,
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
parametric_lens_config = copy_and_replace_light_model(
    cfg["sky"], model_name="model_fixing_pointing"
)
lens_system_fixpointing = build_lens_system(parametric_lens_config)
lens_system = build_lens_system(cfg["sky"])


if cfg["nonparametric_lens"]:
    sky_model = lens_system.get_forward_model_full()
    sky_model_fixpointing = (
        lens_system_fixpointing.get_forward_model_parametric_source()
    )
    parametric_lens_flag = False
else:
    sky_model = lens_system.get_forward_model_parametric()
    sky_model_fixpointing = lens_system_fixpointing.get_forward_model_parametric_source(
        parametric_lens=True
    )
    parametric_lens_flag = True


sky_model = jft.Model(jft.wrap_left(sky_model, SKY_KEY), domain=sky_model.domain)
likelihood_raw, filter_projector, data_dict = build_jwst_likelihoods(
    cfg, grid, sky_model, sky_key=SKY_KEY, sky_unit=SKY_UNIT
)

sky_model_with_keys = jft.Model(
    lambda x: filter_projector(sky_model(x)), init=sky_model.init
)
likelihood = connect_likelihood_to_model(likelihood_raw, sky_model_with_keys)

fixpointing_model = jft.Model(
    lambda x: filter_projector(jft.wrap_left(sky_model_fixpointing, SKY_KEY)(x)),
    init=sky_model_fixpointing.init,
)
likelihood_fixpointing = connect_likelihood_to_model(likelihood_raw, fixpointing_model)


plot_source, plot_residual, plot_lens = get_plot(
    results_directory,
    grid,
    lens_system,
    filter_projector,
    data_dict,
    sky_model_with_keys,
    parametric_lens_flag,
)

_, plot_residual_fixpointing, _ = get_plot(
    join(results_directory, "fixpointing"),
    grid,
    lens_system_fixpointing,
    filter_projector,
    data_dict,
    fixpointing_model,
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
        sky_key=SKY_KEY,
    )


def plot(samples: jft.Samples, state: jft.OptimizeVIState):
    print(f"Plotting: {state.nit}")
    if cfg["plot_results"]:
        plot_residual(samples, state)
        plot_lens(samples, state)
        plot_source(samples, state)


def plot_fixpointing(samples: jft.Samples, state: jft.OptimizeVIState):
    print(f"Plotting: {state.nit}")
    if cfg["plot_results"]:
        plot_residual_fixpointing(samples, state)


cfg_mini = ju.get_config(config_path)["minimization"]
mini_parser_fixpointing = ju.MinimizationParser(
    cfg_mini, ju.get_n_constrained_dof(likelihood_fixpointing), verbose=False
)
mini_parser_full = ju.MinimizationParser(
    cfg_mini, ju.get_n_constrained_dof(likelihood), verbose=False
)

kl_settings_fixpointing = KLSettings(
    random_key=random.PRNGKey(cfg_mini.get("key", 42)),
    outputdir=join(results_directory, "fixpointing"),
    minimization=mini_parser_fixpointing,
    callback=plot_fixpointing,
    kl_jit=False,
    residual_jit=True,
    resume=cfg_mini["fixpointing"].get("resume", False),
    n_total_iterations=cfg_mini["fixpointing"]["n_total_iterations"],
)
kl_settings = KLSettings(
    random_key=random.PRNGKey(cfg_mini.get("key", 42)),
    outputdir=results_directory,
    minimization=mini_parser_full,
    n_total_iterations=cfg_mini["n_total_iterations"],
    callback=plot,
    resume=cfg_mini.get("resume", False),
)

jft.logger.info("Fix pointing reconstruction")
samples_fixpointing, state_imaging = minimization_from_initial_samples(
    likelihood_fixpointing, kl_settings_fixpointing, None
)
jax.clear_caches()

# jft.logger.info("Full reconstruction")
samples, state = minimization_from_initial_samples(
    likelihood,
    kl_settings,
    samples_fixpointing,
    not_take_starting_pos_keys=sky_model_with_keys.domain.keys(),
)
