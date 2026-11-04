import nifty.re as jft

import numpy as np

from ..data.observation import Observation
from ..parse.noise import NoiseCorrection
from .inverse_noise_correction import InverseStandardDeviation


def build_interferometric_noise_correction(
    observation: Observation,
    settings: NoiseCorrection,
    prefix: str = "noise_correction",
) -> InverseStandardDeviation:
    """Build the inverse noise standard deviation with inferred correction factors.

    The noise std of a visibility on the baseline of antennas a and b is
    ``sigma_baseline[a, b] * s[a] * s[b] / sqrt(weight)``. Each factor is an
    inverse gamma prior and enters only if its settings are given.

    Parameters
    ----------
    observation: Observation
        Provides the weights and the antennas of each row.
    settings: NoiseCorrection
        Inverse gamma `alpha` and `scale` of the baseline factors and of the
        antenna factors, each optional.
    prefix: str = "noise_correction"
        The baseline factors are named `{prefix}_sigma`, the antenna factors
        `{prefix}_antenna_sigma`.
    """
    sqrt_weight = np.sqrt(observation.weight.asnumpy())
    domain = {}

    sigma_baseline = None
    if settings.baseline is not None:
        shape, caster = get_baselines(observation)
        sigma_baseline = jft.InvGammaPrior(
            a=settings.baseline.alpha,
            scale=settings.baseline.scale,
            name=f"{prefix}_sigma",
            shape=shape,
        )
        domain |= sigma_baseline.domain

    sigma_antenna = None
    if settings.antenna is not None:
        shape, index1, index2 = get_antennas(observation)
        sigma_antenna = jft.InvGammaPrior(
            a=settings.antenna.alpha,
            scale=settings.antenna.scale,
            name=f"{prefix}_antenna_sigma",
            shape=shape,
        )
        domain |= sigma_antenna.domain

    def inverse_std(x):
        res = sqrt_weight
        if sigma_baseline is not None:
            res = res / sigma_baseline(x)[caster]
        if sigma_antenna is not None:
            s = sigma_antenna(x)
            res = res / (s[index1] * s[index2])[None, :, None]
        return res

    return InverseStandardDeviation(
        inverse_std_model=jft.Model(inverse_std, domain=domain),
        correction_model=sigma_baseline,
        antenna_correction_model=sigma_antenna,
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


def get_antennas(obs):
    """Get the antennas with data and, per row, the index of `ant1` and `ant2`
    among them, in the order of `np.unique`."""
    antennas, index = np.unique(
        np.concatenate((obs.ant1, obs.ant2)), return_inverse=True
    )
    index = index.reshape(-1)
    return (antennas.size,), index[: obs.ant1.size], index[obs.ant1.size :]
