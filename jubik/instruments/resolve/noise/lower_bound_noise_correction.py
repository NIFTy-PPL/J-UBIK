from .inverse_noise_correction import InverseStandardDeviation

import nifty.re as jft

from jax import numpy as jnp
from numpy.typing import NDArray


def build_lower_bound_noise_correction(
    alpha: float,
    scale: float,
    weight: NDArray,
    prefix: str = "noise_correction",
) -> InverseStandardDeviation:
    sigma_min = jft.InvGammaPrior(a=alpha, scale=scale, name=f"{prefix}_sigma_min")
    sigma_weight = jnp.sqrt(weight) ** -1

    def log_inverse_covariance(x):
        variance = (sigma_weight + sigma_min(x)) ** 2
        return -jnp.log(variance)  # log(1/variance)

    inverse_covariance_model = jft.Model(
        log_inverse_covariance, domain=sigma_min.domain
    )

    return InverseStandardDeviation(
        inverse_std_model=inverse_covariance_model, correction_model=sigma_min
    )
