# SPDX-License-Identifier: BSD-2-Clause
# Authors: Julian Rüstig

# Copyright(C) 2026 Max-Planck-Society

# %

"""The weights of one sampled throughput curve on a sky grid.

``Throughput(wavelength, throughput)`` is a sampled curve, what ``nircam``
and ``miri`` provide. ``throughput_weights(spectral, curve)`` turns it into
the ``FilterWeights`` that ``jubik.sky_filter.SkyFilter`` takes. It is the
integral every JWST filter goes through. It knows no instrument, but it does
assume a photon-counting detector, an F_nu sky and a flat-F_nu calibration,
see its docstring. An instrument that breaks one of those writes its own.
"""

from dataclasses import dataclass

import numpy as np
from astropy import units as u
from nifty.re import logger

from ....color import Color
from ....sky_filter import FilterWeights

__all__ = ["MAX_MISSING", "Throughput", "throughput_weights"]

MAX_MISSING = 0.01  # passband fraction allowed outside sky coverage


@dataclass(frozen=True)
class Throughput:
    """One sampled system throughput curve.

    The curve is interpolated linearly between its samples and is zero outside
    the first and last one; the samples are points along the curve, not bin
    edges (a tophat over [lo, hi] is ``[lo, hi]`` with throughput ``[1, 1]``).
    It must be the full system throughput (optics, filter, detector), since a
    wavelength-dependent factor changes the weights; a constant factor does
    not.

    Parameters
    ----------
    wavelength : u.Quantity
        Wavelengths of the samples, a length unit, finite, positive, strictly
        ascending, at least two.
    throughput : np.ndarray
        Throughput at each sample, dimensionless, finite, non-negative, not
        all zero.
    """

    wavelength: u.Quantity
    throughput: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.wavelength, u.Quantity):
            raise TypeError("wavelength must be a Quantity with a length unit")
        wavelength_um = self.wavelength.to_value(u.um)
        throughput = np.asarray(self.throughput, float)
        if (
            wavelength_um.ndim != 1
            or wavelength_um.shape != throughput.shape
            or wavelength_um.size < 2
        ):
            raise ValueError(
                f"wavelength {wavelength_um.shape} and throughput {throughput.shape} "
                "must be 1-D, same length, at least two samples"
            )
        # comparisons below are all False for NaN, so check finiteness first
        if not (np.isfinite(wavelength_um).all() and np.isfinite(throughput).all()):
            raise ValueError("wavelength and throughput must be finite")
        if wavelength_um[0] <= 0 or np.any(np.diff(wavelength_um) <= 0):
            raise ValueError("wavelength must be positive, strictly ascending")
        if np.any(throughput < 0) or not np.any(throughput > 0):
            raise ValueError("throughput must be non-negative and not all zero")
        object.__setattr__(self, "throughput", throughput)


def throughput_weights(
    spectral: Color,
    curve: Throughput,
    *,
    max_missing: float = MAX_MISSING,
    name: str = "band",
) -> FilterWeights:
    """Weights of one sampled throughput curve on the sky channels.

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
        Sky channel bounds, non-overlapping.
    curve : Throughput
        The filter's system throughput curve.
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
        The sky channels overlap, or more than ``max_missing`` of the curve
        lies outside the sky channels, including gaps between them.
    """
    channel_bounds_um = _channel_bounds_um(spectral, name)
    per_channel, missing = _channel_integrals(
        channel_bounds_um, curve.wavelength.to_value(u.um), curve.throughput
    )
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
    nonzero = np.flatnonzero(per_channel)
    channels = slice(int(nonzero[0]), int(nonzero[-1]) + 1)
    return FilterWeights(channels, per_channel[channels] / per_channel.sum())


# --------------------------------------------------------------------------- #
# Internals: the closed-form integral
# --------------------------------------------------------------------------- #


def _channel_bounds_um(spectral: Color, name: str) -> np.ndarray:
    """(n_ch, 2) channel bounds in microns, low edge first, in grid order.

    Raises
    ------
    ValueError
        Two channels overlap. The integral sums per-channel contributions, so
        an overlap would count flux twice and hide missing coverage.
    """
    bounds = spectral.to(u.um, equivalencies=u.spectral()).value
    bounds = np.sort(np.atleast_2d(bounds), axis=1)
    by_lo = bounds[np.argsort(bounds[:, 0])]
    overlap = np.flatnonzero(by_lo[1:, 0] < by_lo[:-1, 1])
    if overlap.size:
        i = overlap[0]
        raise ValueError(
            f"{name}: sky channels overlap, {by_lo[i]} um and {by_lo[i + 1]} um"
        )
    return bounds


def _channel_integrals(
    channel_bounds_um: np.ndarray,
    wavelength_um: np.ndarray,
    throughput: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Integral of T(lam) / lam over every channel, and the fraction missed.

    The curve is linear between samples, so between two consecutive
    breakpoints (curve samples and channel edges) ``T / lam = a / lam + b``
    and the integral is ``a ln(hi / lo) + b (hi - lo)``. Every such segment
    lies inside exactly one channel or outside all of them.

    Parameters
    ----------
    channel_bounds_um : np.ndarray
        (n_ch, 2) non-overlapping sky channel bounds, microns, low edge first.
    wavelength_um : np.ndarray
        Curve samples, microns, positive, ascending.
    throughput : np.ndarray
        Throughput at each sample.

    Returns
    -------
    tuple[np.ndarray, float]
        ``(per_channel, missing)``: the (n_ch,) integrals and the fraction of
        the total integral that no channel covers, exactly zero under full
        coverage.
    """
    edges = channel_bounds_um.ravel()
    edges = edges[(edges > wavelength_um[0]) & (edges < wavelength_um[-1])]
    lam = np.union1d(wavelength_um, edges)
    t = np.interp(lam, wavelength_um, throughput)
    lo, hi = lam[:-1], lam[1:]
    b = (t[1:] - t[:-1]) / (hi - lo)
    a = t[:-1] - b * lo
    segment = a * np.log(hi / lo) + b * (hi - lo)

    per_channel = np.zeros(len(channel_bounds_um))
    covered = np.zeros(segment.size, bool)
    for i, (ch_lo, ch_hi) in enumerate(channel_bounds_um):
        inside = (lo >= ch_lo) & (hi <= ch_hi)
        per_channel[i] = segment[inside].sum()
        covered |= inside
    missing = segment[~covered].sum() / segment.sum()
    return per_channel, missing
