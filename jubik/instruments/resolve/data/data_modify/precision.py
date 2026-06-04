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
    return Observation(
        observation._antpos,
        observation._vis.astype(np.complex64, casting="same_kind", copy=False),
        observation._weight.astype(np.float32, casting="same_kind", copy=False),
        observation._polarization,
        observation._freq,
        observation._auxiliary_tables,
    )
