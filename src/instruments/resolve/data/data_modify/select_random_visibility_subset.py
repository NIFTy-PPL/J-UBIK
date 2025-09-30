from ..observation import Observation
from ..antenna_positions import AntennaPositions

import numpy as np


def select_random_visibility_subset(
    obs: Observation,
    percentage: float | None,
):
    """Restrict observation to a fraction (percentage) of the data points for
    testing purposes.

    Parameters
    ----------
    obs: Observation
        The observation to restrict.
    percentage: float
        The percentage of data points to be taken.
    """
    if percentage is None:
        return obs

    length = obs.uvw.shape[0]
    mask = np.sort(
        np.random.choice(
            np.arange(0, length), size=int(length * percentage), replace=False
        )
    )
    new_vis = obs.vis.asnumpy()[:, mask, :]
    new_weight = obs.weight.asnumpy()[:, mask, :]
    antenna_position = [
        a[mask, ...] if a is not None else None for a in obs.antenna_positions.to_list()
    ]
    antenna_position = AntennaPositions.from_list(antenna_position)

    return Observation(
        antenna_position,
        new_vis,
        new_weight,
        obs.legacy_polarization,
        obs.freq,
        obs._auxiliary_tables,
    )
