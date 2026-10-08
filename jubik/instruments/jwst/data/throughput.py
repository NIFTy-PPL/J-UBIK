# SPDX-License-Identifier: BSD-2-Clause
# Authors: Julian Rüstig

# Copyright(C) 2026 Max-Planck-Society

# %

"""JWST filter weights on a sky grid.

``jwst_filter_weights(spectral, filter_name)`` is what ``build_jwst_likelihoods``
calls for every filter in the config: the ``FilterWeights`` that
``jubik.sky_filter.SkyFilter`` takes, for photon-counted data calibrated in
F_nu. NIRCam filters use the packaged STScI mean system throughput curves. MIRI
filters, for which no curves are packaged, use a half-power tophat from
``JWST_FILTERS`` and log a warning. ``throughput_weights`` is the integral
behind both, usable for any sampled curve.

The NIRCam curves of release ``nircam_throughputs_4Nov2022_v5`` ship with the
package as ``nircam_throughputs_v5.npz``; ``THROUGHPUT_VERSION`` reports the
release. Repack a new STScI release with

    python -m jubik.instruments.jwst.data.throughput --update <mean_throughputs_dir> --version <tag>
"""

import argparse
from functools import cache
from importlib.resources import files
from pathlib import Path

import numpy as np
from astropy import units as u
from nifty.re import logger

from ....color import Color
from ....sky_filter import FilterWeights
from .jwst_information import miri_filters, nircam_filters

__all__ = [
    "MAX_MISSING",
    "jwst_filter_weights",
    "pack_throughputs",
    "throughput_weights",
]

# --------------------------------------------------------------------------- #
# Public: a JWST filter's weights, the integral behind them, and the release
# --------------------------------------------------------------------------- #

MAX_MISSING = 0.01  # passband fraction allowed outside sky coverage


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
        From the packaged system throughput curve for a NIRCam filter; from a
        unit tophat over the half-power range in ``JWST_FILTERS`` for a MIRI
        filter, logged as a warning.

    Raises
    ------
    KeyError
        The name is neither a NIRCam nor a MIRI filter, or a NIRCam filter
        without a packaged curve.
    """
    name = filter_name.upper()
    if name in nircam_filters:
        packaged = _packaged_curves()
        if f"{name}_lam_um" not in packaged:
            raise KeyError(f"{name}: NIRCam filter without a packaged throughput curve")
        wavelength_um, throughput = packaged[f"{name}_lam_um"], packaged[f"{name}_T"]
    elif name in miri_filters:
        _, _, _, blue, red = miri_filters[name]
        logger.warning(f"{name}: no throughput curve packaged, half-power tophat")
        wavelength_um, throughput = np.array([blue, red]), np.ones(2)
    else:
        raise KeyError(f"{name}: not a NIRCam or MIRI filter")
    return throughput_weights(spectral, wavelength_um, throughput, max_missing, name)


def throughput_weights(
    spectral: Color,
    wavelength_um: np.ndarray,
    throughput: np.ndarray,
    max_missing: float = MAX_MISSING,
    name: str = "band",
) -> FilterWeights:
    """Weights of one sampled throughput curve on the sky channels.

    The curve is interpolated linearly between its samples and is zero outside
    the first and last one; the samples are points along the curve, not bin
    edges (a tophat over [lo, hi] is ``[lo, hi]`` with throughput ``[1, 1]``).
    It must be the full system throughput (optics, filter, detector), since a
    wavelength-dependent factor changes the weights; a constant factor does
    not.

    The weight of sky channel i is the integral of ``T(lam) / lam`` over the
    channel, normalised so the weights sum to one. This encodes three
    assumptions about the instrument:

    1. A photon-counting detector: the rate is ``int F_nu T / (h nu) dnu``.
    2. A sky cube in F_nu, constant within each sky channel. With
       ``dnu / nu = dlam / lam`` channel i then contributes
       ``F_nu,i int_i T / lam dlam``.
    3. Data calibrated against a flat-F_nu reference (the JWST ``photom``
       step), so a calibrated value is the photon-weighted mean of F_nu over
       the band. The normalisation reproduces that; a flat sky passes through
       unchanged.

    Parameters
    ----------
    spectral : Color
        Sky channel bounds.
    wavelength_um : np.ndarray
        Wavelengths of the curve samples, microns, strictly ascending, at
        least two.
    throughput : np.ndarray
        Throughput at each sample, dimensionless, non-negative, not all zero.
    max_missing : float
        Largest fraction of the passband allowed outside the sky channels
        before raising. Smaller non-zero fractions log a warning.
    name : str
        Band name for messages.

    Returns
    -------
    FilterWeights
        1-D weights over the contiguous channel range the curve touches.

    Raises
    ------
    ValueError
        The curve is malformed, or more than ``max_missing`` of it lies
        outside the sky channels, including gaps between them.
    """
    wavelength_um = np.asarray(wavelength_um, float)
    throughput = np.asarray(throughput, float)
    if (
        wavelength_um.ndim != 1
        or wavelength_um.shape != throughput.shape
        or wavelength_um.size < 2
    ):
        raise ValueError(
            f"{name}: wavelength_um {wavelength_um.shape} and throughput "
            f"{throughput.shape} must be 1-D, same length, at least two samples"
        )
    if np.any(np.diff(wavelength_um) <= 0):
        raise ValueError(f"{name}: wavelength_um must be strictly ascending")
    if np.any(throughput < 0) or not np.any(throughput > 0):
        raise ValueError(f"{name}: throughput must be non-negative and not all zero")
    channel_bounds_um = spectral.to(u.um, equivalencies=u.spectral()).value
    channel_bounds_um = np.sort(np.atleast_2d(channel_bounds_um), axis=1)
    per_channel = _channel_integrals(
        channel_bounds_um, wavelength_um, throughput, max_missing, name
    )
    nonzero = np.flatnonzero(per_channel)
    channels = slice(int(nonzero[0]), int(nonzero[-1]) + 1)
    return FilterWeights(channels, per_channel[channels])


def pack_throughputs(source_dir: Path, version: str, out_path: Path) -> int:
    """Pack STScI mean system throughput files into one npz.

    Parameters
    ----------
    source_dir : Path
        Directory of ``<filter>_mean_system_throughput.txt`` files with the
        header line "Microns Throughput".
    version : str
        Release tag stored under the key "version".
    out_path : Path
        The npz to write.

    Returns
    -------
    int
        Number of packed curves.

    Raises
    ------
    FileNotFoundError
        ``source_dir`` holds no ``<filter>_mean_system_throughput.txt`` files.
    """
    arrays = {"version": np.array(version)}
    for path in sorted(Path(source_dir).glob(f"*{_STSCI_FILE_SUFFIX}")):
        name = path.name.removesuffix(_STSCI_FILE_SUFFIX).upper()
        wavelength_um, throughput = np.loadtxt(path, skiprows=1, unpack=True)
        arrays[f"{name}_lam_um"] = wavelength_um.astype(np.float64)
        arrays[f"{name}_T"] = throughput.astype(np.float32)
    n_curves = (len(arrays) - 1) // 2
    if n_curves == 0:
        raise FileNotFoundError(f"no *{_STSCI_FILE_SUFFIX} files in {source_dir}")
    np.savez_compressed(out_path, **arrays)
    return n_curves


# Module-level __getattr__ (PEP 562): serves THROUGHPUT_VERSION lazily, so
# importing never touches the npz.
def __getattr__(name: str) -> str:
    """Release tag of the packaged curves, read on first access."""
    if name == "THROUGHPUT_VERSION":
        return str(_packaged_curves()["version"])
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# --------------------------------------------------------------------------- #
# Internals: the packaged npz and the channel integrals
# --------------------------------------------------------------------------- #

_N_FINE = 20001  # integration grid over the curve support
_TOL = 1e-9  # missing fraction treated as zero (integration round-off)

# Rename together with a new throughput release.
_PACKAGED_NPZ = "nircam_throughputs_v5.npz"
_STSCI_FILE_SUFFIX = "_mean_system_throughput.txt"


@cache
def _packaged_curves() -> dict[str, np.ndarray]:
    """All arrays of the packaged npz, loaded once.

    Returns
    -------
    dict[str, np.ndarray]
        ``version`` plus ``<FILTER>_lam_um`` and ``<FILTER>_T`` per curve.
    """
    with (
        (files("jubik.instruments.jwst.data") / _PACKAGED_NPZ).open("rb") as f,
        np.load(f) as npz,
    ):
        return {k: npz[k] for k in npz.files}


def _channel_integrals(
    channel_bounds_um: np.ndarray,
    wavelength_um: np.ndarray,
    throughput: np.ndarray,
    max_missing: float,
    name: str,
) -> np.ndarray:
    """Integrals of T(lam) / lam over every channel, normalised to 1.

    Parameters
    ----------
    channel_bounds_um : np.ndarray
        (n_ch, 2) sky channel bounds, microns, low edge first.
    wavelength_um : np.ndarray
        Curve samples, microns, ascending.
    throughput : np.ndarray
        Throughput at each sample.
    max_missing : float
        See `throughput_weights`.
    name : str
        Band name for messages.

    Returns
    -------
    np.ndarray
        (n_ch,) integrals, summing to one.

    Raises
    ------
    ValueError
        More than ``max_missing`` of the weighted passband lies outside the
        sky channels, including gaps between them.
    """
    # channel edges inside the support join the fine grid, so the per-channel
    # integrals partition the total exactly
    inner = channel_bounds_um.ravel()
    inner = inner[(inner > wavelength_um[0]) & (inner < wavelength_um[-1])]
    fine_um = np.union1d(
        np.linspace(wavelength_um[0], wavelength_um[-1], _N_FINE), inner
    )
    integrand = np.interp(fine_um, wavelength_um, throughput) / fine_um
    total = np.trapezoid(integrand, fine_um)
    per_channel = np.zeros(len(channel_bounds_um))
    for i, (lo, hi) in enumerate(channel_bounds_um):
        inside = (lo <= fine_um) & (fine_um <= hi)
        if inside.sum() > 1:
            per_channel[i] = np.trapezoid(integrand[inside], fine_um[inside])
    missing = 1.0 - per_channel.sum() / total
    missing = 0.0 if missing < _TOL else missing
    if missing > max_missing:
        raise ValueError(
            f"{name}: {100 * missing:.2f}% of the passband lies outside the sky "
            f"channels (allowed {100 * max_missing:.2f}%)"
        )
    if missing > 0:
        logger.warning(
            f"{name}: {100 * missing:.3f}% of the passband lies outside the sky "
            "channels, the output is renormalised onto the covered part"
        )
    return per_channel / per_channel.sum()


# --------------------------------------------------------------------------- #
# Command line: repack a new STScI release
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Repack the NIRCam mean system throughputs shipped with jubik."
    )
    parser.add_argument(
        "--update",
        type=Path,
        required=True,
        metavar="MEAN_THROUGHPUTS_DIR",
        help="directory of STScI *_mean_system_throughput.txt files",
    )
    parser.add_argument(
        "--version",
        required=True,
        help="release tag, e.g. nircam_throughputs_4Nov2022_v5",
    )
    args = parser.parse_args()
    out_path = Path(__file__).with_name(_PACKAGED_NPZ)
    n = pack_throughputs(args.update, args.version, out_path)
    print(f"wrote {n} curves ({args.version}) to {out_path}")
