# SPDX-License-Identifier: BSD-2-Clause
# Authors: Julian Rüstig

# Copyright(C) 2026 Max-Planck-Society

# %

"""The weights of one sampled throughput curve on a sky grid.

``throughput_weights(spectral, wavelength_um, throughput)`` turns a curve into
the ``FilterWeights`` that ``jubik.sky_filter.SkyFilter`` takes. It is the
integral every JWST filter goes through; ``nircam`` and ``miri`` only decide
which curve to hand it. It knows no instrument, but it does assume a
photon-counting detector, an F_nu sky and a flat-F_nu calibration, see its
docstring. An instrument that breaks one of those writes its own.
"""

import numpy as np
from astropy import units as u
from nifty.re import logger

from ....color import Color
from ....sky_filter import FilterWeights

__all__ = ["MAX_MISSING", "throughput_weights"]

# --------------------------------------------------------------------------- #
# Public: the integral and the coverage policy
# --------------------------------------------------------------------------- #

MAX_MISSING = 0.01  # passband fraction allowed outside sky coverage


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


# --------------------------------------------------------------------------- #
# Internals: the channel integrals
# --------------------------------------------------------------------------- #

_N_FINE = 20001  # integration grid over the curve support
_TOL = 1e-9  # missing fraction treated as zero (integration round-off)


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
