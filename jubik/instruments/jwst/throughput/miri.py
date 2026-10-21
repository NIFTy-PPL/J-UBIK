# SPDX-License-Identifier: BSD-2-Clause
# Authors: Julian Rüstig

# Copyright(C) 2026 Max-Planck-Society

# %

"""MIRI throughput curves: a half-power tophat for now.

No MIRI throughput curves are wired up yet, so ``miri_throughput`` returns a
unit tophat over the half-power range in ``miri_filters`` and logs a warning.
When STScI curves get wired up they replace the tophat here, nothing else
changes.
"""

import numpy as np
from astropy import units as u
from nifty.re import logger

from ..data.jwst_information import miri_filters
from .weights import Throughput

__all__ = ["miri_throughput"]


def miri_throughput(filter_name: str) -> Throughput:
    """A MIRI filter's curve: a unit tophat over its half-power range.

    Parameters
    ----------
    filter_name : str
        MIRI filter name, case-insensitive, e.g. "F560W".

    Returns
    -------
    Throughput
        The two tophat corners with throughput one, logged as a warning.

    Raises
    ------
    KeyError
        Not a MIRI filter.
    """
    name = filter_name.upper()
    if name not in miri_filters:
        raise KeyError(f"{name}: not a MIRI filter")
    _, _, _, blue, red = miri_filters[name]
    logger.warning(f"{name}: no throughput curve available, half-power tophat")
    return Throughput([blue, red] * u.um, np.ones(2))
