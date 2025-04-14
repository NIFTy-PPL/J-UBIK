import argparse
from sys import exit

import jax.numpy as jnp
import nifty8.re as jft
import numpy as np
from astropy import units as u
from charm_lensing.lens_system import build_lens_system
from jax import config, random
from matplotlib.colors import LogNorm

import jubik0 as ju
from jubik0.grid import Grid

# from charm_lensing.physical_models.multifrequency_models.nifty_mf import build_nifty_mf_from_grid
from jubik0.instruments.jwst.config_handler import (
    insert_spaces_in_lensing_new,
    load_yaml_and_save_info,
)
from jubik0.instruments.jwst.jwst_likelihoods import build_jwst_likelihoods
from jubik0.instruments.jwst.plotting.plotting import get_plot, plot_prior
from jubik0.likelihood import connect_likelihood_to_model
from jubik0.parse.grid import GridModel

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
    parametric_flag = False
else:
    sky_model = lens_system.get_forward_model_parametric()
    sky_model_parametric = lens_system.get_forward_model_parametric_source(
        parametric_lens=True
    )
    parametric_flag = True
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

likelihood, filter_projector, data_dict = build_jwst_likelihoods(
    cfg, grid, sky_model, sky_key=SKY_KEY, sky_unit=SKY_UNIT
)


sky_model_with_keys = jft.Model(
    lambda x: filter_projector(sky_model(x)), init=sky_model.init
)
# likelihood = connect_likelihood_to_model(likelihood, sky_model_with_keys)

smwk_parametric = jft.Model(
    lambda x: filter_projector(sky_model_parametric(x)), init=sky_model_parametric.init
)
likelihood = likelihood_parametric = connect_likelihood_to_model(
    likelihood, smwk_parametric
)

plot_source, plot_residual, plot_lens, plot_color = get_plot(
    results_directory,
    grid,
    lens_system,
    filter_projector,
    data_dict,
    sky_model,
    sky_model_with_keys,
    cfg,
    parametric_flag,
    sky_key=SKY_KEY,
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
        parametric_flag=parametric_flag,
        sky_key=SKY_KEY,
    )


def plot(samples: jft.Samples, state: jft.OptimizeVIState):
    print(f"Plotting: {state.nit}")
    if cfg["plot_results"]:
        plot_source(samples, state)
        plot_residual(samples, state)
        # plot_color(samples, state)
        plot_lens(samples, state, parametric=parametric_flag)


cfg_mini = ju.get_config(config_path)["minimization"]
n_dof = ju.get_n_constrained_dof(likelihood)
minpars = ju.MinimizationParser(cfg_mini, n_dof, verbose=False)
key = random.PRNGKey(cfg_mini.get("key", 42))
key, rec_key = random.split(key, 2)
pos_init = 0.1 * jft.Vector(jft.random_like(rec_key, likelihood.domain))

print(f"Results: {results_directory}")
samples, state = jft.optimize_kl(
    likelihood_parametric,
    pos_init,
    key=rec_key,
    callback=plot,
    odir=results_directory,
    n_total_iterations=cfg_mini["n_total_iterations"],
    n_samples=minpars.n_samples,
    sample_mode=minpars.sample_mode,
    draw_linear_kwargs=minpars.draw_linear_kwargs,
    nonlinearly_update_kwargs=minpars.nonlinearly_update_kwargs,
    kl_kwargs=minpars.kl_kwargs,
    resume=cfg_mini.get("resume", False),
)
