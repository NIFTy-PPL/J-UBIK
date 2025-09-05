import numpy as np

from ...parse.data.data_modify.masking import CorruptedWeights
from ..antenna_positions import AntennaPositions
from ..observation import Observation


def mask_corrupted_weights(obs: Observation, setting: CorruptedWeights | None):
    """Mask visibilities and weights according to setting.min, setting.max."""

    if setting is None:
        return obs

    mask_bigger = obs.weight.asnumpy() < setting.max
    mask_smaller = obs.weight.asnumpy() > setting.min
    mask = ~(~mask_bigger + ~mask_smaller).any(axis=(0, 2))

    visibilities = obs.vis.asnumpy()[:, mask, :]
    weights = obs.weight.asnumpy()[:, mask, :]
    uvw = obs.antenna_positions.uvw[mask, :]

    return Observation(
        AntennaPositions(
            uvw,
            obs.antenna_positions.ant1,
            obs.antenna_positions.ant2,
            obs.antenna_positions.time,
        ),
        visibilities,
        weights,
        obs.legacy_polarization,
        obs.freq,
        auxiliary_tables=obs._auxiliary_tables,
    )
