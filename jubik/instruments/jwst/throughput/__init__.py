# SPDX-License-Identifier: BSD-2-Clause
# Authors: Julian Rüstig

# Copyright(C) 2026 Max-Planck-Society

# %

"""JWST filter weights on a sky grid.

``jwst_filter_weights(spectral, filter_name)`` is what ``build_jwst_likelihoods``
calls for every filter in the config: the ``FilterWeights`` that
``jubik.sky_filter.SkyFilter`` takes. It only decides which instrument the
filter belongs to. ``nircam`` reads the packaged STScI curve, ``miri`` builds a
half-power tophat, and both hand their curve to ``weights.throughput_weights``,
the photon-counting F_nu integral every JWST filter goes through.
"""

from ....color import Color
from ....sky_filter import FilterWeights
from ..data.jwst_information import miri_filters, nircam_filters
from .miri import miri_filter_weights
from .nircam import nircam_filter_weights
from .weights import MAX_MISSING, throughput_weights

__all__ = [
    "MAX_MISSING",
    "jwst_filter_weights",
    "miri_filter_weights",
    "nircam_filter_weights",
    "throughput_weights",
]


def jwst_filter_weights(
    spectral: Color, filter_name: str, max_missing: float = MAX_MISSING
) -> FilterWeights:
    """Weights of a JWST filter on the sky channels.

    Parameters
    ----------
    spectral : Color
        Sky channel bounds.
    filter_name : str
        NIRCam or MIRI filter name, case-insensitive, e.g. "F444W".
    max_missing : float
        See `throughput_weights`.

    Returns
    -------
    FilterWeights
        `nircam_filter_weights` for a NIRCam filter, `miri_filter_weights` for
        a MIRI filter.

    Raises
    ------
    KeyError
        The name is neither a NIRCam nor a MIRI filter.
    """
    name = filter_name.upper()
    if name in nircam_filters:
        return nircam_filter_weights(spectral, name, max_missing)
    if name in miri_filters:
        return miri_filter_weights(spectral, name, max_missing)
    raise KeyError(f"{name}: not a NIRCam or MIRI filter")
