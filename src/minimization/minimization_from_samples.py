from .minimization_parser import MinimizationParser

import nifty8.re as jft
from jax import random

from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class KLSettings:
    random_key: random.PRNGKey
    outputdir: str
    minimization: MinimizationParser
    callback: Optional[Callable[jft.Samples, jft.OptimizeVIState]] = None
    sample_multiply: Optional[float] = 0.1


def _initial_position(
    init_key: random.PRNGKey,
    domain,
    sample_multiply,
    starting_samples: Optional[jft.Samples] = None,
):
    initial_position = jft.random_like(init_key, domain) * sample_multiply

    if starting_samples is not None:
        starting_pos = starting_samples.pos
        while isinstance(starting_pos, jft.Vector):
            starting_pos = starting_pos.tree
        while isinstance(initial_position, jft.Vector):
            initial_position = initial_position.tree

        for key in starting_pos.keys():
            initial_position[key] = starting_pos[key]

    return jft.Vector(initial_position)


def minimization_from_initial_samples(
    likelihood: jft.Likelihood,
    kl_settings: KLSettings,
    starting_samples: Optional[jft.Samples] = None
):
    '''This function executes a KL minimization specified by the
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
    '''
    init_key, mini_key = random.split(kl_settings.random_key, 2)

    initial_position = _initial_position(
        init_key,
        likelihood.domain,
        sample_multiply=kl_settings.sample_multiply,
        starting_samples=starting_samples,
    )

    # Minimze only parametric
    minimization = kl_settings.minimization
    samples, state = jft.optimize_kl(
        likelihood,
        initial_position,
        key=mini_key,
        callback=kl_settings.callback,
        odir=kl_settings.outputdir,
        n_total_iterations=minimization.n_total_iterations,

        n_samples=minimization.n_samples,
        sample_mode=minimization.sample_mode,
        draw_linear_kwargs=minimization.draw_linear_kwargs,
        nonlinearly_update_kwargs=minimization.nonlinearly_update_kwargs,
        kl_kwargs=minimization.kl_kwargs,
        resume=minimization.resume,
    )
    return samples, state
