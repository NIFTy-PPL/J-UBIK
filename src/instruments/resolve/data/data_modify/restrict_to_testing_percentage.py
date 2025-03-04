from ..observation import Observation
from ..antenna_positions import AntennaPositions

import numpy as np


def restrict_to_testing_percentage(
    obs: Observation,
    percentage: float,
):
    '''Restrict observation to a fraction (percentage) of the data points for
    testing purposes.

    Parameters
    ----------
    obs: Observation
        The observation to restrict.
    percentage: float
        The percentage of data points to be taken.
    '''
    length = obs.uvw.shape[0]
    mask = np.sort(np.random.choice(np.arange(0, length),
                                    size=int(length*percentage),
                                    replace=False))
    new_vis = obs.vis.val[:, mask, :]
    new_weight = obs.weight.val[:, mask, :]
    antenna_position = [a[mask, ...] if a is not None else None
                        for a in obs.antenna_positions.to_list()]
    antenna_position = AntennaPositions.from_list(antenna_position)

    return Observation(
        antenna_position,
        new_vis,
        new_weight,
        obs.polarization,
        obs.freq,
        obs._auxiliary_tables)
