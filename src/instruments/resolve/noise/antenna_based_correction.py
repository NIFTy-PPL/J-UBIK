from .inverse_noise_correction import InverseStandardDeviation

import nifty.re as jft

import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike


def build_interferometric_noise_correction(
    correction_factor_shape: tuple[int],
    correction_to_weight_mask: ArrayLike,
    alpha_correction: float,
    scale_correction: float,
    weight: ArrayLike,
    prefix: str = "noise_correction",
) -> InverseStandardDeviation:
    """Build an interferometric noise correction.

    Parameters
    ----------
    correction_factor_shape: tuple[int]
        The shape of the correction factor. Could be baselines, antennas, etc.
        But basically this is determined by the `correction_to_weight_mask`.
    correction_to_weight_mask: ArrayLike
        The mapping from correction factors to the weights. This is done by a mask.
    alpha_correction: float
        InverseGamma alpha
    scale_correction: float
        InverseGamma scale
    weight: ArrayLike
        The weight array.
    prefix: str = "noise_correction"
    """
    sigma_baseline = jft.InvGammaPrior(
        a=alpha_correction,
        scale=scale_correction,
        name=f"{prefix}_sigma",
        shape=correction_factor_shape,
    )

    sqrt_weight = np.sqrt(weight)

    def inverse_std(x):
        correction = 1 / sigma_baseline(x)
        return sqrt_weight * jnp.take(correction, correction_to_weight_mask)

    return InverseStandardDeviation(
        inverse_std_model=jft.Model(inverse_std, domain=sigma_baseline.domain),
        correction_model=sigma_baseline,
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
