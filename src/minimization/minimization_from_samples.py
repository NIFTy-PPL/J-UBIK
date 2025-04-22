from ..minimization_parser import MinimizationParser

import nifty8.re as jft
from jax import random

import os
import pickle
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class KLSettings:
    random_key: random.PRNGKey
    outputdir: str
    minimization: MinimizationParser
    n_total_iterations: int
    callback: Optional[Callable[jft.Samples, jft.OptimizeVIState]] = None
    sample_multiply: Optional[float] = 0.1
    constants: tuple[str] = ()
    point_estimates: tuple[str] = ()
    kl_jit: bool = True
    residual_jit: bool = True
    resume: bool = False
    resume_from_pickle_path: str | None = None


def _initial_position(
    init_key: random.PRNGKey,
    domain,
    kl_settings: KLSettings,
    starting_samples: Optional[jft.Samples] = None,
    not_take_starting_pos_keys: tuple[str] = (),
):
    position_rescaling = kl_settings.sample_multiply
    initial_position = jft.random_like(init_key, domain) * position_rescaling
    while isinstance(initial_position, jft.Vector):
        initial_position = initial_position.tree

    if kl_settings.resume_from_pickle_path is not None:
        if starting_samples is not None:
            jft.logger.warning(
                "Warning: Overwriting starting position from path: "
                f"{kl_settings.resume_from_pickle_path}."
            )
        else:
            jft.logger.info(f"Loading result: {kl_settings.resume_from_pickle_path}")

        with open(kl_settings.resume_from_pickle_path, "rb") as f:
            starting_samples, _ = pickle.load(f)

    if starting_samples is not None:
        starting_pos = starting_samples.pos
        while isinstance(starting_pos, jft.Vector):
            starting_pos = starting_pos.tree

        for key in initial_position.keys():
            if key in starting_pos and not (key in not_take_starting_pos_keys):
                jft.logger.info(f"Taking {key}: starting_pos[key]")
                initial_position[key] = starting_pos[key]

    initial_position, opt_vi_state = jft.Vector(initial_position), None

    LAST_FILENAME = "last.pkl"
    resume = False
    if kl_settings.resume:
        if isinstance(kl_settings.resume, str):
            last_pkl = kl_settings.resume
        else:
            last_pkl = os.path.join(kl_settings.outputdir, LAST_FILENAME)

        if os.path.isfile(last_pkl):
            with open(last_pkl, "rb") as f:
                initial_position, opt_vi_state = pickle.load(f)

    return initial_position, opt_vi_state


def minimization_from_initial_samples(
    likelihood: jft.Likelihood,
    kl_settings: KLSettings,
    starting_samples: Optional[jft.Samples] = None,
    not_take_starting_pos_keys: tuple[str] = (),
):
    """This function executes a KL minimization specified by the
    `KLSettings`. Optionally, one can start the reconstruction from
    a set of `starting_samples` of a subdomain of the likelihood.

    Parameters
    ----------
    random_key:
        Random random_key used for sampling.
    likelihood: jft.Likelihood
        The likelihood to be minimized.
    kl_settings:
        - random_key
        - outputdir
        - minimization: MinimizationParser
        - callback
        - sample_multiply
    starting_samples: Optional
        This can be a set of samples from a subdomain of likelihood. The mean
        will be taken in order to start the minimization.
    """
    jft.logger.info(f"Results: {kl_settings.outputdir}")

    init_key, mini_key = random.split(kl_settings.random_key, 2)

    initial_position, opt_vi_state = _initial_position(
        init_key,
        likelihood.domain,
        position_rescaling=kl_settings.sample_multiply,
        starting_samples=starting_samples,
        not_take_starting_pos_keys=not_take_starting_pos_keys,
        resume_from_pickle_path=kl_settings.resume_from_pickle_path,
    )

    # Minimze only parametric
    minimization: MinimizationParser = kl_settings.minimization
    samples, state = jft.optimize_kl(
        likelihood,
        initial_position,
        key=mini_key,
        callback=kl_settings.callback,
        odir=kl_settings.outputdir,
        n_total_iterations=kl_settings.n_total_iterations,
        n_samples=minimization.n_samples,
        sample_mode=minimization.sample_mode,
        draw_linear_kwargs=minimization.draw_linear_kwargs,
        nonlinearly_update_kwargs=minimization.nonlinearly_update_kwargs,
        kl_kwargs=minimization.kl_kwargs,
        constants=kl_settings.constants,
        point_estimates=kl_settings.point_estimates,
        resume=False,
        kl_jit=kl_settings.kl_jit,
        residual_jit=kl_settings.residual_jit,
        _optimize_vi_state=opt_vi_state,
    )
    return samples, state
