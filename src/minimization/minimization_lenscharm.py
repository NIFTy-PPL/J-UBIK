from .minimization_parser import MinimizationParser
from ..likelihood import get_n_constrained_dof

import nifty8.re as jft
from jax import random

from os.path import join
from dataclasses import dataclass
from typing import Callable


@dataclass
class LensCharmMinimization:
    key: random.PRNGKey
    outputdir: str

    likelihood_para: jft.Likelihood
    mini_cfg_para: dict
    plot_full: Callable
    resume_para: bool

    likelihood_full: jft.Likelihood
    mini_cfg_full: dict
    plot_para: Callable
    resume_full: bool


def lenscharm_minimization(
    lenscharm_minimization_settings: LensCharmMinimization
):
    key = lenscharm_minimization_settings.key
    outputdir = lenscharm_minimization_settings.outputdir
    likelihood_para = lenscharm_minimization_settings.likelihood_para
    likelihood_full = lenscharm_minimization_settings.likelihood_full
    mini_cfg_para = lenscharm_minimization_settings.mini_cfg_para
    mini_cfg_full = lenscharm_minimization_settings.mini_cfg_full
    plot_full = lenscharm_minimization_settings.plot_full
    plot_para = lenscharm_minimization_settings.plot_para
    resume_full = lenscharm_minimization_settings.resume_full
    resume_para = lenscharm_minimization_settings.resume_para

    key, init_par_key, init_ful_key, mini_par_key, mini_ful_key = random.split(
        key, 6)

    # Initialize seed, and initial position
    initial_position = jft.random_like(
        init_par_key, likelihood_para.domain
    ) * 0.1

    n_dof_para = get_n_constrained_dof(likelihood_para)
    mini_parser_para = MinimizationParser(mini_cfg_para, n_dof_para)

    # Minimze only parametric
    samples, state = jft.optimize_kl(
        likelihood_para,
        initial_position,
        key=mini_par_key,
        callback=plot_para,
        odir=join(outputdir, 'save_parametric'),
        n_total_iterations=mini_cfg_para['n_total_iterations'],

        n_samples=mini_parser_para.n_samples,
        sample_mode=mini_parser_para.sample_mode,
        draw_linear_kwargs=mini_parser_para.draw_linear_kwargs,
        nonlinearly_update_kwargs=mini_parser_para.nonlinearly_update_kwargs,
        kl_kwargs=mini_parser_para.kl_kwargs,
        resume=resume_para,
    )

    print()
    print("*" * 80)
    print("Switching to full model")
    print("*" * 80)
    print()
    initial_position = (jft.random_like(
        init_ful_key, likelihood_full.domain) * 0.1).tree
    for key in samples.pos.tree.keys():
        initial_position[key] = samples.pos[key]
    initial_position = jft.Vector(initial_position)

    # Minimization config
    n_dof_full = get_n_constrained_dof(likelihood_full)
    mini_parser_full = MinimizationParser(mini_cfg_full, n_dof_full)

    # Minimze only non-parametric
    samples, state = jft.optimize_kl(
        likelihood_full,
        initial_position,
        key=mini_ful_key,
        callback=plot_full,
        odir=join(outputdir, 'save_full'),
        n_total_iterations=mini_cfg_full['n_total_iterations'],
        n_samples=mini_parser_full.n_samples,
        sample_mode=mini_parser_full.sample_mode,
        draw_linear_kwargs=mini_parser_full.draw_linear_kwargs,
        nonlinearly_update_kwargs=mini_parser_full.nonlinearly_update_kwargs,
        kl_kwargs=mini_parser_full.kl_kwargs,
        resume=resume_full,
    )
