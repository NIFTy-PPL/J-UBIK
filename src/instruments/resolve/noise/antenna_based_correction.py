from .log_inverse_noise_correction import LogInverseNoiseCovariance

import nifty8.re as jft

import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike


def build_antenna_based_noise_correction(
    base_shape: tuple[int],
    antenna_to_baseline: ArrayLike,
    alpha_antenna: float,
    scale_antenna: float,
    weight: ArrayLike,
    prefix: str = "antennabased_noise_correction",
) -> LogInverseNoiseCovariance:
    pass

    sigma_antenna = jft.InvGammaPrior(
        a=alpha_antenna, scale=scale_antenna, name=f"{prefix}_sigma", shape=base_shape
    )

    sigma_weight = np.sqrt(weight) ** -1

    def log_inverse_covariance(x):
        variance = (sigma_antenna(x)[antenna_to_baseline] + sigma_weight) ** 2
        return -jnp.log(variance)

    log_inverse_covariance_model = jft.Model(
        log_inverse_covariance, domain=sigma_antenna.domain
    )

    return LogInverseNoiseCovariance(
        log_inverse_covariance_model=log_inverse_covariance_model,
        correction_model=sigma_antenna,
    )


def get_baselines(obs):
    """Get the baselines corresponding to the antennas of the observation."""
    antenna_pairs = np.array((obs.ant1, obs.ant2)).T
    baselines = np.unique(antenna_pairs, axis=0)

    caster = np.zeros(obs.vis.shape)
    for ii in range(len(baselines)):
        tmp = np.argwhere(np.all(antenna_pairs == baselines[ii], axis=1))
        caster[:, tmp, :] = ii
    caster = caster.astype(int)
    return (baselines.shape[0],), caster
