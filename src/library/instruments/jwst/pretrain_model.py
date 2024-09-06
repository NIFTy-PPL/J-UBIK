import yaml
import os
import pickle
from os.path import join
from typing import Optional

from numpy.typing import ArrayLike
from jax import random

import nifty8.re as jft
import jubik0 as ju
from jubik0.jwst.config_handler import insert_spaces_in_lensing
from charm_lensing.lens_system import build_lens_system, LensSystem


def get_pretrain_samples_and_lens_system(pretrain_path: str) -> tuple[jft.Samples, LensSystem]:

    cfg = yaml.load(open(pretrain_path, 'r'), Loader=yaml.SafeLoader)

    insert_spaces_in_lensing(cfg)
    lens_system = build_lens_system(cfg['lensing'])

    odir = cfg['files']['res_dir']
    LAST_FILENAME = "last.pkl"
    last_fn = os.path.join(odir, LAST_FILENAME) if odir is not None else None

    with open(last_fn, "rb") as f:
        samples, opt_vi_st = pickle.load(f)

    return samples, lens_system


def get_pretrain_data(pretrain_path: str, full_mass=False):
    '''Returns the
        - lens_mass (mean, std)
        - lens_light (mean, std)
        - source_light (mean, std)
    for the last iteration and model of the pretrain_path.'''

    cfg = yaml.load(open(pretrain_path, 'r'), Loader=yaml.SafeLoader)

    insert_spaces_in_lensing(cfg)
    lens_system = build_lens_system(cfg['lensing'])

    odir = cfg['files']['res_dir']
    LAST_FILENAME = "last.pkl"
    last_fn = os.path.join(odir, LAST_FILENAME) if odir is not None else None

    with open(last_fn, "rb") as f:
        samples, opt_vi_st = pickle.load(f)

    mass = lens_system.lens_plane_model.convergence_model if full_mass else lens_system.lens_plane_model.convergence_model.parametric()
    lens_mass = jft.mean_and_std([mass(s) for s in samples])

    lens_light = lens_system.lens_plane_model.light_model
    lens_light = jft.mean_and_std([lens_light(s) for s in samples])

    source_light = lens_system.source_plane_model.light_model
    source_light = jft.mean_and_std([source_light(s) for s in samples])

    return lens_mass, lens_light, source_light


def build_plot_pretrain(
    outdir: str,
    data_std: ArrayLike,
    model: jft.Model,
    plotting_config: dict = {},
):
    from jubik0.jwst.jwst_plotting import (_plot_data_data_model_residuals,
                                           _get_model_samples_or_position)

    import matplotlib.pyplot as plt
    import numpy as np

    ylen = data_std[0].shape[0] if len(data_std[0].shape) == 3 else None

    def plot_pretrain(
        samples: jft.Samples, state_or_none: Optional[jft.OptimizeVIState]
    ):
        model_mean = jft.mean(_get_model_samples_or_position(samples, model))

        if ylen is None:
            fig, axes = plt.subplots(1, 3, figsize=(12, 3), dpi=300)
            ims = np.zeros_like(axes)
            ims = _plot_data_data_model_residuals(
                ims,
                axes,
                data_key=0,
                data=data_std[0],
                data_model=model_mean,
                std=data_std[1],
                plotting_config=plotting_config,
            )

        else:
            fig, axes = plt.subplots(ylen, 3, figsize=(12, 3*ylen), dpi=300)
            ims = np.zeros_like(axes)
            for ii in range(ylen):
                ims[ii] = _plot_data_data_model_residuals(
                    ims[ii],
                    axes[ii],
                    data_key=ii,
                    data=data_std[0][ii],
                    data_model=model_mean[ii],
                    std=data_std[1][ii],
                    plotting_config=plotting_config,
                )

        for ax, im in zip(axes.flatten(), ims.flatten()):
            if not isinstance(im, int):
                fig.colorbar(im, ax=ax, shrink=0.7)
        fig.tight_layout()

        if state_or_none is None:
            plt.show()
        else:
            plt.savefig(join(outdir, f'{state_or_none.nit:02d}.png'), dpi=300)
            plt.close()

    return plot_pretrain


def pretrain_model(
    res_dir: str,
    model_name: str,
    cfg_mini: dict,
    data_std: ArrayLike,
    model: jft.Model,
    plotting_config: dict = {}
):
    likelihood = ju.library.likelihood.build_gaussian_likelihood(
        data=data_std[0], std=data_std[1])
    likelihood = likelihood.amend(
        model, domain=jft.Vector(model.domain))

    n_dof = ju.calculate_n_constrained_dof(likelihood)
    minpars = ju.MinimizationParser(cfg_mini, n_dof, verbose=False)
    key = random.PRNGKey(cfg_mini.get('key', 42))
    key, rec_key = random.split(key, 2)
    pos_init = 0.1 * jft.Vector(jft.random_like(rec_key, likelihood.domain))

    model_dir = join(res_dir, model_name)
    plot = build_plot_pretrain(model_dir, data_std, model, plotting_config)
    print(f'Results: {model_dir}')
    samples, state = jft.optimize_kl(
        likelihood,
        pos_init,
        key=rec_key,
        callback=plot,
        odir=model_dir,
        n_total_iterations=cfg_mini['pretraining_steps'],
        n_samples=minpars.n_samples,
        sample_mode=minpars.sample_mode,
        draw_linear_kwargs=minpars.draw_linear_kwargs,
        nonlinearly_update_kwargs=minpars.nonlinearly_update_kwargs,
        kl_kwargs=minpars.kl_kwargs,
        resume=cfg_mini.get('resume', False),
    )
    return samples


def pretrain_lens_system(cfg: dict, lens_system: LensSystem):
    from matplotlib.colors import LogNorm

    if ((cfg['minimization']['pretraining_steps'] is None) or
            (cfg['minimization']['pretraining_steps'] == 0)):
        return None

    pretrain_res_dir = join(cfg['files']['res_dir'], 'pretrain')
    lens_mass, lens_light, source_light = get_pretrain_data(
        cfg['files']['pretrain_path'])

    source_light_samples = pretrain_model(
        res_dir=pretrain_res_dir,
        model_name='source_light',
        cfg_mini=cfg["minimization"],
        data_std=source_light,
        model=lens_system.source_plane_model.light_model,
        plotting_config=dict(norm=LogNorm),
    )

    lens_light_samples = pretrain_model(
        res_dir=pretrain_res_dir,
        model_name='lens_light',
        cfg_mini=cfg["minimization"],
        data_std=lens_light,
        model=lens_system.lens_plane_model.light_model,
        plotting_config=dict(norm=LogNorm),
    )

    lens_mass_samples = pretrain_model(
        res_dir=pretrain_res_dir,
        model_name='lens_mass',
        cfg_mini=cfg["minimization"],
        data_std=lens_mass,
        model=lens_system.lens_plane_model.convergence_model,
        plotting_config=dict(norm=LogNorm),
    )

    source_light_mean, lens_light_mean, lens_mass_mean = (
        jft.mean(source_light_samples),
        jft.mean(lens_light_samples),
        jft.mean(lens_mass_samples)
    )

    while isinstance(source_light_mean, jft.Vector):
        source_light_mean = source_light_mean.tree

    while isinstance(lens_light_mean, jft.Vector):
        lens_light_mean = lens_light_mean.tree

    while isinstance(lens_mass_mean, jft.Vector):
        lens_mass_mean = lens_mass_mean.tree

    return source_light_mean | lens_light_mean | lens_mass_mean
