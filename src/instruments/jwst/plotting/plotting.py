from .jwst_plotting import (
    get_alpha_nonpar,
    build_plot_lens_system,
    build_color_components_plotting,
    build_plot_sky_residuals,
    build_plot_source,
    FieldPlottingConfig,
    ResidualPlottingConfig,
)
from ..filter_projector import FilterProjector
from ..grid import Grid

import nifty8.re as jft

from charm_lensing.lens_system import LensSystem

from jax import random
from matplotlib.colors import LogNorm

from typing import Union


def get_plot(
    results_directory: str,
    grid: Grid,
    lens_system: LensSystem,
    filter_projector: FilterProjector,
    data_dict: dict,
    sky_model: jft.Model,
    sky_model_with_keys: jft.Model,
    cfg: dict,
    parametric_flag: bool,
    sky_key: str = 'sky',
    residual_ylen_offset: int = 0,
):
    if isinstance(filter_projector.domain, jft.ShapeWithDtype):
        sky_model_new = sky_model
    else:
        sky_model_new = jft.Model(lambda x: sky_model(x)[sky_key],
                                  domain=sky_model.domain)

    ll_alpha, ll_nonpar, sl_alpha, sl_nonpar = get_alpha_nonpar(lens_system)

    plot_lens = build_plot_lens_system(
        results_directory,
        plotting_config=dict(
            norm_source=LogNorm,
            norm_lens=LogNorm,
            # norm_source_alpha=LogNorm,
            norm_source_nonparametric=LogNorm,
            # norm_mass=LogNorm,
        ),
        lens_system=lens_system,
        grid=grid,
        lens_light_alpha_nonparametric=(ll_alpha, ll_nonpar),
        source_light_alpha_nonparametric=(sl_alpha, sl_nonpar),
    )

    residual_plotting_config = ResidualPlottingConfig(
        sky=FieldPlottingConfig(norm=LogNorm),
        data=FieldPlottingConfig(norm=LogNorm),
        display_pointing=False,
        xmax_residuals=cfg.get('max_residuals', 4),
        ylen_offset=residual_ylen_offset,
    )
    plot_residual = build_plot_sky_residuals(
        results_directory=results_directory,
        filter_projector=filter_projector,
        data_dict=data_dict,
        sky_model_with_key=sky_model_with_keys,
        small_sky_model=sky_model_new,
        plotting_config=residual_plotting_config,
    )

    plot_color = build_color_components_plotting(
        lens_system.source_plane_model.light_model.nonparametric, results_directory, substring='source')

    plot_source = build_plot_source(
        results_directory,
        plotting_config=dict(
            norm_source=LogNorm,
            norm_source_parametric=LogNorm,
            norm_source_nonparametric=LogNorm,
            extent=lens_system.source_plane_model.space.extend().extent,
        ),
        grid=grid,
        source_light_model=lens_system.source_plane_model.light_model,
        source_light_alpha=sl_alpha,
        source_light_parametric=lens_system.source_plane_model.light_model.parametric,
        source_light_nonparametric=sl_nonpar,
        attach_name=''
    )

    return plot_source, plot_residual, plot_lens, plot_color


def plot_prior(
    config_path: str,
    likelihood: jft.Likelihood,
    filter_projector: FilterProjector,
    plot_residuals: bool,
    plot_source: Union[bool, callable],
    plot_lens: Union[bool, callable],

    data_dict: dict,
    parametric_flag: bool,
    sky_key: str = 'sky',
    residual_ylen_offset: int = 0,
):
    from charm_lensing.lens_system import build_lens_system
    from jubik0.instruments.jwst.config_handler import (
        insert_spaces_in_lensing_new)
    import yaml
    test_key, _ = random.split(random.PRNGKey(42), 2)

    cfg = yaml.load(open(config_path, 'r'), Loader=yaml.SafeLoader)
    insert_spaces_in_lensing_new(cfg['sky'])

    lens_system = build_lens_system(cfg['sky'])
    if cfg['nonparametric_lens']:
        sky_model = lens_system.get_forward_model_full()
        parametric_flag = False
    else:
        sky_model = lens_system.get_forward_model_parametric()
        parametric_flag = True
    sky_model = jft.Model(jft.wrap_left(sky_model, sky_key),
                          domain=sky_model.domain)
    sky_model_with_keys = jft.Model(
        lambda x: filter_projector(sky_model(x)),
        init=sky_model.init
    )

    from jubik0.instruments.jwst.parse.grid import yaml_to_grid_model
    from jubik0.instruments.jwst.grid import Grid
    grid = Grid.from_grid_model(yaml_to_grid_model(cfg['sky']['grid']))
    results_directory = cfg['files']['res_dir']
    ll_alpha, ll_nonpar, sl_alpha, sl_nonpar = get_alpha_nonpar(lens_system)

    if plot_source and not callable(plot_source):
        plot_source = build_plot_source(
            results_directory,
            plotting_config=dict(
                norm_source=LogNorm,
                norm_source_parametric=LogNorm,
                norm_source_nonparametric=LogNorm,
                extent=lens_system.source_plane_model.space.extend().extent,
            ),
            grid=grid,
            source_light_model=lens_system.source_plane_model.light_model,
            source_light_alpha=sl_alpha,
            source_light_parametric=lens_system.source_plane_model.light_model.parametric,
            source_light_nonparametric=sl_nonpar,
            attach_name=''
        )
    elif not callable(plot_source):
        def plot_source(x): return True

    if plot_lens and not callable(plot_lens):
        plot_lens = build_plot_lens_system(
            results_directory,
            plotting_config=dict(
                norm_source=LogNorm,
                norm_lens=LogNorm,
                # norm_source_alpha=LogNorm,
                norm_source_nonparametric=LogNorm,
                # norm_mass=LogNorm,
            ),
            lens_system=lens_system,
            grid=grid,
            lens_light_alpha_nonparametric=(ll_alpha, ll_nonpar),
            source_light_alpha_nonparametric=(sl_alpha, sl_nonpar),
        )
    elif not callable(plot_lens):
        def plot_lens(x, v, parametric): return True

    if not isinstance(filter_projector.domain, jft.ShapeWithDtype):
        sky_model_new = jft.Model(lambda x: sky_model(x)[sky_key],
                                  domain=sky_model.domain)
    else:
        sky_model_new = sky_model

    if plot_residuals:
        def filter_data(datas: dict):
            filters = list()

            for kk, vv in datas.items():
                f = kk.split('_')[0]
                if f not in filters:
                    filters.append(f)
                    yield kk, vv

        prior_dict = {kk: vv for kk, vv in filter_data(data_dict)}
        residual_plotting_config = ResidualPlottingConfig(
            sky=FieldPlottingConfig(norm=LogNorm),
            data=FieldPlottingConfig(norm=LogNorm),
            display_pointing=False,
            xmax_residuals=cfg.get('max_residuals', 4),
            ylen_offset=residual_ylen_offset,
        )
        plot_prior_residuals = build_plot_sky_residuals(
            results_directory='',
            data_dict=prior_dict,
            filter_projector=filter_projector,
            sky_model_with_key=sky_model_with_keys,
            small_sky_model=sky_model_new,
            plotting_config=residual_plotting_config,
        )
    else:
        def plot_prior_residuals(x): return True

    nsamples = cfg.get('prior_samples') if cfg.get('prior_samples') else 3
    for ii in range(nsamples):
        test_key, _ = random.split(test_key, 2)
        position = likelihood.init(test_key)
        while isinstance(position, jft.Vector):
            position = position.tree

        plot_prior_residuals(position)
        plot_source(position)
        # plot_color(position)
        plot_lens(position, None, parametric=parametric_flag)
