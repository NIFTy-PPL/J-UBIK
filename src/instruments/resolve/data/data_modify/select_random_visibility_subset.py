from __future__ import annotations

from pathlib import Path

from ..observation import Observation
from ..antenna_positions import AntennaPositions
from ...parse.data.data_modify.select_subset import SelectSubset

import numpy as np


def select_random_visibility_subset(
    obs: Observation,
    select_subset: SelectSubset | None,
):
    """Restrict observation to a fraction (percentage) of the data points for
    testing purposes. Optionally saves/loads the mask to/from a file.

    Parameters
    ----------
    obs: Observation
        The observation to restrict.
    select_subset: SelectSubset | None
        Configuration for subset selection (percentage and optional mask_path).
    """
    if select_subset is None:
        return obs

    length = obs.uvw.shape[0]

    if select_subset.mask_path is not None:
        mask_file = Path(select_subset.mask_path)
        if mask_file.exists():
            mask = np.load(mask_file)
        else:
            mask = _generate_mask(length, select_subset.percentage)
            np.save(mask_file, mask)
    else:
        mask = _generate_mask(length, select_subset.percentage)

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


def _generate_mask(length: int, percentage: float) -> np.ndarray:
    rng = np.random.Generator(np.random.PCG64(seed=42))
    return np.sort(
        rng.choice(np.arange(0, length), size=int(length * percentage), replace=False)
    )
