"""Visibility-subset selection for resolve observations."""

from __future__ import annotations

from pathlib import Path

from ..observation import Observation
from ..antenna_positions import AntennaPositions
from ...parse.data.data_modify.visibility_subset import SelectSubset

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
        # `np.save` appends the suffix, hence the existence check has to look
        # for the suffixed file as well.
        if mask_file.suffix != ".npy":
            mask_file = mask_file.with_name(mask_file.name + ".npy")
        if mask_file.exists():
            mask = _validate_mask(np.load(mask_file), length, select_subset)
        else:
            mask = _generate_mask(length, select_subset.percentage)
            mask_file.parent.mkdir(parents=True, exist_ok=True)
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


def _n_selected(length: int, percentage: float | None) -> int:
    """Number of rows that correspond to `percentage` of `length` rows."""
    if percentage is None:
        raise ValueError(
            "`SelectSubset.percentage` is None: cannot generate a visibility "
            "subset mask. Either set a percentage or point `mask_path` to an "
            "existing mask file."
        )
    if not 0.0 < percentage <= 1.0:
        raise ValueError(
            f"`SelectSubset.percentage` must be a fraction in (0, 1], got "
            f"{percentage}. (Percent values such as 25 instead of 0.25 are "
            "not supported.)"
        )
    n_selected = int(length * percentage)
    if n_selected == 0:
        raise ValueError(
            f"`SelectSubset.percentage`={percentage} selects 0 of {length} "
            "rows. Increase the percentage."
        )
    return n_selected


def _validate_mask(
    mask: np.ndarray, length: int, select_subset: SelectSubset
) -> np.ndarray:
    """Check that a mask loaded from disk fits the given observation."""
    if mask.size > 0 and (mask.min() < 0 or mask.max() >= length):
        raise ValueError(
            f"The mask stored in {select_subset.mask_path} indexes rows "
            f"[{mask.min()}, {mask.max()}], which does not fit an observation "
            f"with {length} rows. It was probably generated for a different "
            "data set."
        )
    if select_subset.percentage is not None:
        expected = _n_selected(length, select_subset.percentage)
        if mask.size != expected:
            raise ValueError(
                f"The mask stored in {select_subset.mask_path} selects "
                f"{mask.size} rows, but percentage="
                f"{select_subset.percentage} of {length} rows corresponds to "
                f"{expected} rows. It was probably generated for a different "
                "data set."
            )
    return mask


def _generate_mask(length: int, percentage: float | None) -> np.ndarray:
    rng = np.random.Generator(np.random.PCG64(seed=42))
    return np.sort(
        rng.choice(
            np.arange(0, length), size=_n_selected(length, percentage), replace=False
        )
    )
