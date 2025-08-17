from ..minimization_parser import MinimizationParser

import nifty.re as jft
from jax import random

import os
import pickle
from dataclasses import dataclass
from typing import Callable, Optional, Iterable


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


def _initial_position(
    init_key: random.PRNGKey,
    domain,
    kl_settings: KLSettings,
    starting_samples_or_position: jft.Samples | jft.Vector | None = None,
    not_take_starting_pos_keys: tuple[str] = (),
):
    position_rescaling = kl_settings.sample_multiply
    initial_position = jft.random_like(init_key, domain) * position_rescaling
    while isinstance(initial_position, jft.Vector):
        initial_position = initial_position.tree

    if starting_samples_or_position is not None:
        if isinstance(starting_samples_or_position, jft.Samples):
            starting_pos = starting_samples_or_position.pos
        else:
            starting_pos = starting_samples_or_position

        while isinstance(starting_pos, jft.Vector):
            starting_pos: dict = starting_pos.tree

        for key in initial_position.keys():
            if key in starting_pos and not (key in not_take_starting_pos_keys):
                jft.logger.info(f"Taking {key}")
                initial_position[key] = starting_pos[key]

    initial_position, opt_vi_state = jft.Vector(initial_position), None

    LAST_FILENAME = "last.pkl"
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
    starting_samples_or_position: jft.Samples | jft.Vector | None = None,
    not_take_starting_pos_keys: Iterable[str] = (),
):
    """This function executes a KL minimization specified by the
    `KLSettings`. Optionally, one can start the reconstruction from
    a set of `starting_samples_or_position` of a subdomain of the likelihood.

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
    starting_samples_or_position: Optional
        This can be a set of samples from a subdomain of likelihood. The mean
        will be taken as the starting position of the minimization.
    not_take_starting_pos_keys: Optional
        This will hold the keys that should not be taken in the starting position.
    """
    jft.logger.info(f"Results: {kl_settings.outputdir}")

    init_key, mini_key = random.split(kl_settings.random_key, 2)

    initial_position, opt_vi_state = _initial_position(
        init_key,
        likelihood.domain,
        kl_settings=kl_settings,
        starting_samples_or_position=starting_samples_or_position,
        not_take_starting_pos_keys=not_take_starting_pos_keys,
    )

    resume = (
        os.path.join(os.getcwd(), kl_settings.outputdir, "last.pkl")
        if kl_settings.resume
        else False
    )

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
        resume=resume,
        kl_jit=kl_settings.kl_jit,
        residual_jit=kl_settings.residual_jit,
        _optimize_vi_state=opt_vi_state,
    )
    return samples, state


def get_full_position_from_partial(
    partial_samples_or_position: jft.Samples | jft.Vector,
    new_full_position: jft.Vector,
    discard_keys: tuple[str] = (),
) -> jft.Vector:
    
    if partial_samples_or_position is not None:
        if isinstance(partial_samples_or_position, jft.Samples):
            partial_position = partial_samples_or_position.pos
        else:
            partial_position = partial_samples_or_position

    while isinstance(partial_position, jft.Vector):
        partial_position = partial_position.tree

    while isinstance(new_full_position, jft.Vector):
        new_full_position = new_full_position.tree

    updated_position = dict(new_full_position)

    for key in updated_position:
        if key in partial_position.keys() and key not in discard_keys:
            jft.logger.info(f"Re-using {key} position.")
            updated_position[key] = partial_position[key]

    return jft.Vector(updated_position)