from typing import Union
from os.path import join

import matplotlib.pyplot as plt
import nifty8.re as jft
import numpy as np
from jax import random

from ....grid import Grid
from ....parse.grid import GridModel
from ..filter_projector import FilterProjector
from .plotting_base import find_closest_factors, get_position_or_samples_of_model
from .residuals import build_plot_sky_residuals
from ..parse.plotting import (
    LensSystemPlottingConfig,
    ResidualPlottingConfig,
    FieldPlottingConfig,
    MultiFrequencyPlottingConfig,
)
from .plotting_lens_system import build_plot_lens_system
from .plot_source import build_plot_source


def get_plot(
    results_directory: str,
    grid: Grid,
    lens_system,
    filter_projector: FilterProjector,
    data_dict: dict,
    sky_model_with_keys: jft.Model,
    parametric_lens: bool,
    parametric_source: bool = False,
    max_residuals: int = 4,
    residual_ylen_offset: int = 0,
):
    from charm_lensing.lens_system import LensSystem

    lens_system: LensSystem = lens_system

    source_plotting_config = MultiFrequencyPlottingConfig(
        combined=FieldPlottingConfig(vmin=1e-4, norm="log"),
        reference=FieldPlottingConfig(vmax=5, norm=None),
        alpha=FieldPlottingConfig(norm=None),
    )
    lens_light_plotting_config = MultiFrequencyPlottingConfig(
        combined=FieldPlottingConfig(vmin=1e-4, vmax=1e2, norm="log"),
        reference=FieldPlottingConfig(vmax=5, norm=None),
        alpha=FieldPlottingConfig(norm=None),
    )

    plotting_config_lens_system = LensSystemPlottingConfig(
        source=source_plotting_config,
        lens_light=lens_light_plotting_config,
        share_source_vmin_vmax=False,
    )

    plot_lens = build_plot_lens_system(
        results_directory,
        plotting_config=plotting_config_lens_system,
        lens_system=lens_system,
        grid=grid,
        parametric_lens=parametric_lens,
        parametric_source=parametric_source,
    )

    plot_source = build_plot_source(
        results_directory,
        plotting_config=source_plotting_config,
        lens_system=lens_system,
        grid=grid,
    )

    residual_plotting_config = ResidualPlottingConfig(
        sky=lens_light_plotting_config.combined,
        data=FieldPlottingConfig(norm="log", vmin=1e-3),
        display_pointing=False,
        xmax_residuals=max_residuals,
        ylen_offset=residual_ylen_offset,
    )
    plot_residual = build_plot_sky_residuals(
        results_directory=results_directory,
        filter_projector=filter_projector,
        data_dict=data_dict,
        sky_model_with_filters=sky_model_with_keys,
        plotting_config=residual_plotting_config,
    )

    return plot_source, plot_residual, plot_lens


def plot_prior(
    config_path: str,
    likelihood: jft.Likelihood,
    filter_projector: FilterProjector,
    plot_residuals: bool,
    plot_source: Union[bool, callable],
    plot_lens: Union[bool, callable],
    data_dict: dict,
    parametric_flag: bool,
    sky_key: str = "sky",
    residual_ylen_offset: int = 0,
):
    import yaml
    from charm_lensing.lens_system import build_lens_system

    from jubik0.instruments.jwst.config_handler import insert_spaces_in_lensing_new

    test_key, _ = random.split(random.PRNGKey(42), 2)

    cfg = yaml.load(open(config_path, "r"), Loader=yaml.SafeLoader)
    insert_spaces_in_lensing_new(cfg["sky"])

    lens_system = build_lens_system(cfg["sky"])
    if cfg["nonparametric_lens"]:
        sky_model = lens_system.get_forward_model_full()
        parametric_flag = False
    else:
        sky_model = lens_system.get_forward_model_parametric()
        parametric_flag = True
    sky_model = jft.Model(jft.wrap_left(sky_model, sky_key), domain=sky_model.domain)
    sky_model_with_keys = jft.Model(
        lambda x: filter_projector(sky_model(x)), init=sky_model.init
    )

    grid = Grid.from_grid_model(GridModel.from_yaml_dict(cfg["sky"]["grid"]))
    results_directory = cfg["files"]["res_dir"]
    ll_alpha, ll_nonpar, sl_alpha, sl_nonpar = get_alpha_nonpar(lens_system)

    if plot_source and not callable(plot_source):
        plot_source = build_plot_source(
            results_directory,
            plotting_config=dict(
                norm_source="log",
                norm_source_parametric="log",
                norm_source_nonparametric="log",
                extent=lens_system.source_plane_model.space.extend().extent,
            ),
            grid=grid,
            source_light_model=lens_system.source_plane_model.light_model,
            source_light_alpha=sl_alpha,
            source_light_parametric=lens_system.source_plane_model.light_model.parametric,
            source_light_nonparametric=sl_nonpar,
            attach_name="",
        )
    elif not callable(plot_source):

        def plot_source(x):
            return True

    if plot_lens and not callable(plot_lens):
        plot_lens = build_plot_lens_system(
            results_directory,
            plotting_config=dict(
                norm_source="log",
                norm_lens="log",
                # norm_source_alpha=LogNorm,
                norm_source_nonparametric="log",
                # norm_mass=LogNorm,
            ),
            lens_system=lens_system,
            grid=grid,
            lens_light_alpha_nonparametric=(ll_alpha, ll_nonpar),
            source_light_alpha_nonparametric=(sl_alpha, sl_nonpar),
        )
    elif not callable(plot_lens):

        def plot_lens(x, v, parametric):
            return True

    if not isinstance(filter_projector.domain, jft.ShapeWithDtype):
        sky_model_new = jft.Model(
            lambda x: sky_model(x)[sky_key], domain=sky_model.domain
        )
    else:
        sky_model_new = sky_model

    if plot_residuals:

        def filter_data(datas: dict):
            filters = list()

            for kk, vv in datas.items():
                f = kk.split("_")[0]
                if f not in filters:
                    filters.append(f)
                    yield kk, vv

        prior_dict = {kk: vv for kk, vv in filter_data(data_dict)}
        residual_plotting_config = ResidualPlottingConfig(
            sky=FieldPlottingConfig(norm="log"),
            data=FieldPlottingConfig(norm="log"),
            display_pointing=False,
            xmax_residuals=cfg.get("max_residuals", 4),
            ylen_offset=residual_ylen_offset,
        )
        plot_prior_residuals = build_plot_sky_residuals(
            results_directory="",
            data_dict=prior_dict,
            filter_projector=filter_projector,
            sky_model_with_key=sky_model_with_keys,
            small_sky_model=sky_model_new,
            plotting_config=residual_plotting_config,
        )
    else:

        def plot_prior_residuals(x):
            return True

    nsamples = cfg.get("prior_samples") if cfg.get("prior_samples") else 3
    for ii in range(nsamples):
        test_key, _ = random.split(test_key, 2)
        position = likelihood.init(test_key)
        while isinstance(position, jft.Vector):
            position = position.tree

        plot_prior_residuals(position)
        plot_source(position)
        # plot_color(position)
        plot_lens(position, None, parametric=parametric_flag)


def build_plot_model_samples(
    results_directory: str,
    model_name: str,
    model: jft.Model,
    mapping_axis: int | None = None,
    plotting_config: dict = {},
):
    sky_directory = join(results_directory, model_name)
    makedirs(sky_directory, exist_ok=True)

    norm = plotting_config.get("norm", Normalize)
    sky_min = plotting_config.get("min", 5e-4)
    extent = plotting_config.get("extent")

    def plot_sky_samples(samples: jft.Samples, x: jft.OptimizeVIState):
        samps_big = [model(si) for si in samples]
        mean, std = jft.mean_and_std(samps_big)
        vmin = np.max((mean.min(), sky_min))
        vmax = mean.max()

        if mapping_axis is None:
            ylen, xlen = find_closest_factors(len(samples) + 2)
        else:
            ylen, xlen = model.target[mapping_axis], len(samples) + 2
        fig, axes = plt.subplots(ylen, xlen, figsize=(2 * xlen, 1.5 * ylen), dpi=300)

        if mapping_axis is None:
            axes = [axes]

        for axi in axes:
            for ax, fld in zip(axi.flatten(), samps_big):
                im = ax.imshow(
                    fld,
                    origin="lower",
                    extent=extent,
                    norm=norm(vmin=vmin, vmax=vmax),
                    interpolation="None",
                )
                fig.colorbar(im, ax=ax, shrink=0.7)

        fig.tight_layout()
        fig.savefig(join(sky_directory, f"{x.nit:02d}.png"), dpi=300)
        plt.close()

    return plot_sky_samples


def rgb_plotting(
    lens_system,
    samples: jft.Samples,
    three_filter_names: tuple[str] = ("f1000w", "f770w", "f560w"),
):
    sl = lens_system.source_plane_model.light_model
    ms, mss = jft.mean_and_std([sl(si) for si in samples])

    from astropy import cosmology, units

    sextent = np.array(lens_system.source_plane_model.space.extend().extent)
    extent_kpc = (
        np.tan(sextent * units.arcsec)
        * cosmology.Planck13.angular_diameter_distance(4.2).to(units.kpc)
    ).value

    # f0, f1, f2 = 'f560w', 'f444w', 'f356w'
    # f0, f1, f2 = 'f1000w', 'f770w', 'f560w'
    f0, f1, f2 = three_filter_names

    rgb = np.zeros((384, 384, 3))
    rgb[:, :, 0] = ms[0]
    rgb[:, :, 1] = ms[1]
    rgb[:, :, 2] = ms[2]

    rgb = rgb / np.max(rgb)
    rgb = np.sqrt(rgb)

    import matplotlib.font_manager as fm
    from charm_lensing.plotting import display_scalebar, display_text

    fig, ax = plt.subplots(1, 1, figsize=(11.5, 10))
    ax.imshow(rgb, origin="lower", extent=extent_kpc, interpolation="None")
    display_scalebar(
        ax, dict(size=5, unit="kpc", fontproperties=fm.FontProperties(size=24))
    )
    display_text(
        ax,
        text=dict(s=f0, color="red", fontproperties=fm.FontProperties(size=30)),
        keyword="top_right",
        y_offset_ticker=0,
    )
    display_text(
        ax,
        text=dict(s=f1, color="green", fontproperties=fm.FontProperties(size=30)),
        keyword="top_right",
        y_offset_ticker=1,
    )
    display_text(
        ax,
        text=dict(s=f2, color="blue", fontproperties=fm.FontProperties(size=30)),
        keyword="top_right",
        y_offset_ticker=2,
    )
    ax.set_xlim(-5.5, 6)
    ax.set_ylim(-4, 6)
    plt.axis("off")  # Turn off axis
    plt.tight_layout()
    plt.savefig(f"{f0}_{f1}_{f2}_source.png")
    plt.close()
