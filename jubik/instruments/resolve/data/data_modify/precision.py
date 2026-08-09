import numpy as np

from ...data.observation import Observation


def to_double_precision(observation: Observation):
    return Observation(
        observation._antpos,
        observation._vis.astype(np.complex128, casting="same_kind", copy=False),
        observation._weight.astype(np.float64, casting="same_kind", copy=False),
        observation._polarization,
        observation._freq,
        observation._auxiliary_tables,
    )


def to_single_precision(observation: Observation):
    with np.errstate(over="ignore"):
        weight = observation._weight.astype(
            np.float32, casting="same_kind", copy=False
        )
    _check_weight_cast(observation._weight, weight)
    return Observation(
        observation._antpos,
        observation._vis.astype(np.complex64, casting="same_kind", copy=False),
        weight,
        observation._polarization,
        observation._freq,
        observation._auxiliary_tables,
    )


def _check_weight_cast(weight, single_weight):
    """Ensure that the cast to single precision preserves the flags (weight ==
    0) and stays finite."""
    if not np.all(np.isfinite(single_weight)):
        raise ValueError(
            "Casting the weights to single precision overflows: the largest "
            f"weight is {np.max(weight)}, but float32 only reaches "
            f"{np.finfo(np.float32).max}. Rescale the weights before casting."
        )
    if np.any((weight > 0.0) & (single_weight == 0.0)):
        smallest = np.min(weight[weight > 0.0])
        raise ValueError(
            "Casting the weights to single precision underflows: the smallest "
            f"non-zero weight is {smallest}, which becomes 0.0 in float32 and "
            "would silently flag the corresponding visibilities. Rescale the "
            "weights before casting."
        )
