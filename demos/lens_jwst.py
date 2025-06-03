import argparse
from sys import exit
from os.path import join

import nifty8.re as jft
from astropy import units as u
from charm_lensing.lens_system import build_lens_system
from jax import config, random

import jubik0 as ju
from jubik0.grid import Grid

from jubik0.instruments.jwst.config_handler import (
    insert_spaces_in_lensing_new,
    load_yaml_and_save_info,
)
from jubik0.instruments.jwst.jwst_likelihoods import build_jwst_likelihoods
from jubik0.instruments.jwst.minimization.mask_hot_pixels import (
    HotPixelMasking,
    minimize_with_hot_pixel_masking,
)
from jubik0.instruments.jwst.minimization.alignment_process import (
    alignment_minimization_process,
)
from jubik0.instruments.jwst.plotting.plotting import (
    get_plot,
    plot_prior,
    build_plot_alignment_residuals,
)
from jubik0.instruments.jwst.parse.plotting import FieldPlottingConfig
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
    parametric_lens_flag = False
else:
    sky_model = lens_system.get_forward_model_parametric()
    parametric_lens_flag = True


sky_model = jft.Model(jft.wrap_left(sky_model, SKY_KEY), domain=sky_model.domain)
likelihood_products = build_jwst_likelihoods(cfg, grid, sky_model, sky_unit=SKY_UNIT)

sky_model_with_keys = jft.Model(
    lambda x: likelihood_products.target.filter_projector(sky_model(x)),
    init=sky_model.init,
)
likelihood_target = connect_likelihood_to_model(
    likelihood_products.target.likelihood, sky_model_with_keys
)

plot_source, plot_residual, plot_lens = get_plot(
    results_directory,
    grid,
    lens_system,
    likelihood_products.target.filter_projector,
    likelihood_products.target.plotting,
    sky_model_with_keys,
    parametric_lens_flag,
    plotting_cfg=cfg["jwst_lens_plotting"],
)


def plot_target(samples: jft.Samples, state: jft.OptimizeVIState):
    print(f"Plotting: {state.nit}")
    if cfg["jwst_lens_plotting"].get("plot_results", True):
        plot_residual(samples, state)
        plot_lens(samples, state)
        plot_source(samples, state)


if cfg["jwst_lens_plotting"].get("prior_samples"):
    plot_prior(
        config_path,
        likelihood_target,
        likelihood_products.target.filter_projector,
        plot_residuals=True,
        plot_source=False,
        plot_lens=False,
        data_dict=likelihood_products.target.plotting,
        sky_key=SKY_KEY,
    )


if likelihood_products.alignment is not None:
    samples_fixpointing = alignment_minimization_process(
        config_path=config_path,
        results_directory=results_directory,
        alignment=likelihood_products.alignment,
    )

    plot_alignment_residuals = build_plot_alignment_residuals(
        results_directory,
        likelihood_products.alignment.plotting.psf,
        FieldPlottingConfig(vmin=1e-4, norm="log"),
    )


def callback(samples, state):
    plot_target(samples, state)
    # plot_alignment_residuals(samples, state)


cfg_mini = ju.get_config(config_path)["minimization"]
mini_parser_full = ju.MinimizationParser(
    cfg_mini, ju.get_n_constrained_dof(likelihood_target), verbose=False
)
kl_settings = KLSettings(
    random_key=random.PRNGKey(cfg_mini.get("key", 42)),
    outputdir=results_directory,
    minimization=mini_parser_full,
    n_total_iterations=cfg_mini["n_total_iterations"],
    callback=callback,
    resume=cfg_mini.get("resume", False),
    # point_estimates=[
    #     k
    #     for k in samples_fixpointing.pos.tree.keys()
    #     if k in likelihood_target.domain.tree
    # ],
    # constants=[
    #     k
    #     for k in samples_fixpointing.pos.tree.keys()
    #     if k in likelihood_target.domain.tree
    # ],
)

jft.logger.info("Full reconstruction")

samples, state = minimize_with_hot_pixel_masking(
    likelihood=likelihood_products.target,
    kl_settings=kl_settings,
    masking=HotPixelMasking.from_yaml_dict(
        cfg["telescope"],
        sky_with_filter=sky_model_with_keys,
    ),
    starting_samples=samples_fixpointing,
)
