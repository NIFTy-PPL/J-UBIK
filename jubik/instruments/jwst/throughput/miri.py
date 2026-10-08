# SPDX-License-Identifier: BSD-2-Clause
# Authors: Julian Rüstig

# Copyright(C) 2026 Max-Planck-Society

# %

"""MIRI filter weights from a half-power tophat.

No MIRI throughput curves are packaged yet, so ``miri_filter_weights`` uses a
unit tophat over the half-power range in ``miri_filters`` and logs a warning.
When STScI curves get packaged they replace the tophat here, nothing else
changes.
"""

import numpy as np
from nifty.re import logger

from ....color import Color
from ....sky_filter import FilterWeights
from ..data.jwst_information import miri_filters
from .weights import MAX_MISSING, throughput_weights

__all__ = ["miri_filter_weights"]


def miri_filter_weights(
    spectral: Color, filter_name: str, max_missing: float = MAX_MISSING
) -> FilterWeights:
    """Weights of a MIRI filter on the sky channels, from its half-power range.

    Parameters
    ----------
    spectral : Color
        Sky channel bounds.
    filter_name : str
        MIRI filter name, case-insensitive, e.g. "F560W".
    max_missing : float
        See `throughput_weights`.

    Returns
    -------
    FilterWeights
        Photon-weighted integrals of a unit tophat over the half-power range,
        logged as a warning.

    Raises
    ------
    KeyError
        Not a MIRI filter.
    """
    name = filter_name.upper()
    if name not in miri_filters:
        raise KeyError(f"{name}: not a MIRI filter")
    _, _, _, blue, red = miri_filters[name]
    logger.warning(f"{name}: no throughput curve packaged, half-power tophat")
    return throughput_weights(
        spectral, np.array([blue, red]), np.ones(2), max_missing, name
    )
