# SPDX-License-Identifier: BSD-2-Clause
# Authors: Julian Rüstig

# Copyright(C) 2026 Max-Planck-Society

# %

"""JWST filter weights on a sky grid.

``jwst_filter_weights(spectral, filter_name, throughput_dir)`` is what
``build_jwst_likelihoods`` calls for every filter in the config: the
``FilterWeights`` that ``jubik.sky_filter.SkyFilter`` takes. It only decides
which instrument the filter belongs to. ``nircam`` reads the STScI curve from
``throughput_dir`` (the config's ``files.throughputs``, downloaded once with
``python -m jubik.instruments.jwst.throughput.nircam --download <dir>``),
``miri`` builds a half-power tophat, and both hand their curve to
``weights.throughput_weights``, the photon-counting F_nu integral every JWST
filter goes through.
"""

from pathlib import Path

from ....color import Color
from ....sky_filter import FilterWeights
from ..data.jwst_information import miri_filters, nircam_filters
from .miri import miri_filter_weights
from .nircam import download_nircam_throughputs, nircam_filter_weights
from .weights import MAX_MISSING, throughput_weights

__all__ = [
    "MAX_MISSING",
    "download_nircam_throughputs",
    "jwst_filter_weights",
    "miri_filter_weights",
    "nircam_filter_weights",
    "throughput_weights",
]


def jwst_filter_weights(
    spectral: Color,
    filter_name: str,
    throughput_dir: str | Path | None = None,
    max_missing: float = MAX_MISSING,
) -> FilterWeights:
    """Weights of a JWST filter on the sky channels.

    Parameters
    ----------
    spectral : Color
        Sky channel bounds.
    filter_name : str
        NIRCam or MIRI filter name, case-insensitive, e.g. "F444W".
    throughput_dir : str | Path | None
        Directory of the downloaded NIRCam curves; required for a NIRCam
        filter, unused for MIRI.
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
    ValueError
        A NIRCam filter without ``throughput_dir``.
    """
    name = filter_name.upper()
    if name in nircam_filters:
        if throughput_dir is None:
            raise ValueError(
                f"{name}: NIRCam filters need throughput_dir, the directory of "
                "STScI curves (config files.throughputs); download them with "
                "python -m jubik.instruments.jwst.throughput.nircam --download <dir>"
            )
        return nircam_filter_weights(spectral, name, throughput_dir, max_missing)
    if name in miri_filters:
        return miri_filter_weights(spectral, name, max_missing)
    raise KeyError(f"{name}: not a NIRCam or MIRI filter")
