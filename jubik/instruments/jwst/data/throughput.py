# SPDX-License-Identifier: BSD-2-Clause
# Authors: Julian Rüstig

# Copyright(C) 2026 Max-Planck-Society

# %

"""JWST throughput curves and the sky-channel weights they give.

`jwst_transmission` loads a filter's curve: NIRCam mean system throughputs ship
with the package (nircam_throughputs_v5.npz); filters without a packaged curve
fall back to a half-power tophat from JWST_FILTERS. `throughput_weights` turns
a curve into the `FilterWeights` a `jubik.sky_filter.SkyFilter` consumes, for
photon-counted data calibrated in F_nu. Repack a new STScI release with

    python -m jubik.instruments.jwst.data.throughput --update <mean_throughputs_dir> --version <tag>
"""

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache
from importlib.resources import files
from pathlib import Path

import numpy as np
from astropy import units as u
from nifty.re import logger

from ....color import Color
from ....sky_filter import FilterWeights
from .jwst_information import JWST_FILTERS

__all__ = [
    "MAX_MISSING",
    "Transmission",
    "jwst_transmission",
    "pack_throughputs",
    "throughput_weights",
]

MAX_MISSING = 0.01  # passband fraction allowed outside sky coverage
_N_FINE = 20001  # integration grid over the curve support
_TOL = 1e-9  # missing fraction treated as zero (integration round-off)

# Rename together with a new throughput release.
_NPZ = "nircam_throughputs_v5.npz"
_SUFFIX = "_mean_system_throughput.txt"


@dataclass(frozen=True)
class Transmission:
    """Sampled throughput curve of one output bin.

    The curve is interpolated linearly between consecutive samples and is zero
    outside the first and last one. The samples are points along the curve,
    not bin edges; a tophat over [lo, hi] is the two samples ``[lo, hi]`` with
    throughput ``[1, 1]``, a measured filter is a few hundred samples. The
    curve must be the full system throughput (optics, filter, detector), since
    a wavelength-dependent factor changes the weights; a constant factor does
    not.

    Parameters
    ----------
    wavelength_um : np.ndarray
        Wavelengths of the samples, microns, strictly ascending, at least two.
    throughput : np.ndarray
        Throughput at each sample, dimensionless, non-negative, not all zero.
    """

    wavelength_um: np.ndarray
    throughput: np.ndarray

    def __post_init__(self) -> None:
        wavelength_um = np.asarray(self.wavelength_um, float)
        throughput = np.asarray(self.throughput, float)
        if (
            wavelength_um.ndim != 1
            or wavelength_um.shape != throughput.shape
            or wavelength_um.size < 2
        ):
            raise ValueError(
                f"wavelength_um {wavelength_um.shape} and throughput "
                f"{throughput.shape} must be 1-D, same length"
            )
        if np.any(np.diff(wavelength_um) <= 0):
            raise ValueError("wavelength_um must be strictly ascending")
        if np.any(throughput < 0) or not np.any(throughput > 0):
            raise ValueError("throughput must be non-negative and not all zero")
        object.__setattr__(self, "wavelength_um", wavelength_um)
        object.__setattr__(self, "throughput", throughput)


Band = Transmission | Sequence[Transmission]


def throughput_weights(
    spectral: Color,
    band: Band,
    max_missing: float = MAX_MISSING,
    name: str = "band",
) -> FilterWeights:
    """Weights of a band on the sky channels, from its throughput curve(s).

    The weight of sky channel i for output bin b is the integral of
    ``T_b(lam) / lam`` over channel i, normalised so every output bin's weights
    sum to one. This encodes three assumptions about the instrument:

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
    band : Transmission | Sequence[Transmission]
        One curve (an imaging filter, one output plane) or one per output bin
        (a spectrograph's data bins, one output cube in this order).
    max_missing : float
        Largest fraction of an output bin's passband allowed outside the sky
        channels before raising. Smaller non-zero fractions log a warning.
    name : str
        Band name for messages.

    Returns
    -------
    FilterWeights
        1-D weights for one curve, (n_out, n_channels) for a sequence, over the
        contiguous channel range the curves touch.

    Raises
    ------
    ValueError
        A curve exceeds ``max_missing``, or the band has no curves.
    """
    channel_bounds_um = spectral.to(u.um, equivalencies=u.spectral()).value
    channel_bounds_um = np.sort(np.atleast_2d(channel_bounds_um), axis=1)
    curves = [band] if isinstance(band, Transmission) else list(band)
    if not curves:
        raise ValueError(f"{name}: band has no transmission curves")
    rows = np.stack(
        [
            _channel_integrals(
                channel_bounds_um,
                transmission,
                max_missing,
                name if len(curves) == 1 else f"{name}[{bin_index}]",
            )
            for bin_index, transmission in enumerate(curves)
        ]
    )
    nonzero = np.flatnonzero(rows.any(axis=0))
    channels = slice(int(nonzero[0]), int(nonzero[-1]) + 1)
    weights = rows[:, channels]
    return FilterWeights(
        channels, weights[0] if isinstance(band, Transmission) else weights
    )


def _channel_integrals(
    channel_bounds_um: np.ndarray,
    transmission: Transmission,
    max_missing: float,
    name: str,
) -> np.ndarray:
    """Integrals of T(lam) / lam over every channel, normalised to 1.

    Parameters
    ----------
    channel_bounds_um : np.ndarray
        (n_ch, 2) sky channel bounds, microns, low edge first.
    transmission : Transmission
        Curve of one output bin.
    max_missing : float
        See `throughput_weights`.
    name : str
        Band name for messages.

    Returns
    -------
    np.ndarray
        (n_ch,) integrals.

    Raises
    ------
    ValueError
        More than ``max_missing`` of the weighted passband lies outside the
        sky channels, including gaps between them.
    """
    wavelength_um = transmission.wavelength_um
    throughput = transmission.throughput
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


@cache
def _curves() -> dict[str, np.ndarray]:
    """All arrays of the packaged npz, loaded once."""
    with (
        (files("jubik.instruments.jwst.data") / _NPZ).open("rb") as f,
        np.load(f) as npz,
    ):
        return {k: npz[k] for k in npz.files}


def __getattr__(name: str) -> str:
    # THROUGHPUT_VERSION is read lazily so importing never touches the npz.
    if name == "THROUGHPUT_VERSION":
        return str(_curves()["version"])
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def jwst_transmission(filter_name: str) -> Transmission:
    """Transmission curve of a JWST filter.

    Parameters
    ----------
    filter_name : str
        JWST filter name, case-insensitive, e.g. "F444W".

    Returns
    -------
    Transmission
        The packaged throughput curve, or a half-power tophat from
        JWST_FILTERS when no curve is packaged.

    Raises
    ------
    KeyError
        The filter has neither a packaged curve nor a JWST_FILTERS entry.
    """
    name = filter_name.upper()
    curves = _curves()
    if f"{name}_lam_um" in curves:
        return Transmission(curves[f"{name}_lam_um"], curves[f"{name}_T"])
    if name not in JWST_FILTERS:
        raise KeyError(f"{name}: no throughput curve packaged and not in JWST_FILTERS")
    _, _, _, blue, red = JWST_FILTERS[name]
    logger.warning(f"{name}: no throughput curve packaged, half-power tophat")
    return Transmission(np.array([blue, red]), np.ones(2))


def pack_throughputs(source: Path, version: str, out: Path) -> int:
    """Pack STScI mean system throughput files into one npz.

    Parameters
    ----------
    source : Path
        Directory of ``<filter>_mean_system_throughput.txt`` files with the
        header line "Microns Throughput".
    version : str
        Release tag stored under the key "version".
    out : Path
        The npz to write.

    Returns
    -------
    int
        Number of packed curves.
    """
    arrays = {"version": np.array(version)}
    for path in sorted(Path(source).glob(f"*{_SUFFIX}")):
        name = path.name.removesuffix(_SUFFIX).upper()
        lam, T = np.loadtxt(path, skiprows=1, unpack=True)
        arrays[f"{name}_lam_um"] = lam.astype(np.float64)
        arrays[f"{name}_T"] = T.astype(np.float32)
    n_curves = (len(arrays) - 1) // 2
    if n_curves == 0:
        raise FileNotFoundError(f"no *{_SUFFIX} files in {source}")
    np.savez_compressed(out, **arrays)
    return n_curves


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
    out = Path(__file__).with_name(_NPZ)
    n = pack_throughputs(args.update, args.version, out)
    print(f"wrote {n} curves ({args.version}) to {out}")
