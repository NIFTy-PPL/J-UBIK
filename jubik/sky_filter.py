# SPDX-License-Identifier: BSD-2-Clause
# Authors: Julian Rüstig

# Copyright(C) 2026 Max-Planck-Society

# %

"""SkyFilter: the sky owns the spectral channels, every band selects from them.

The sky is one F_nu cube on the channels of ``grid.spectral``. An instrument
provides one Transmission curve T(lam) per band and nothing else. The weight
of a band on channel i is the integral of T(lam) / lam over that channel
(photon-counted F_nu data), normalised so the weights sum to one. The band
image is the weighted sum of the sky channels, in the units of the sky.
Instrument-agnostic: curves are passed in, never loaded here.
"""

from dataclasses import dataclass

import jax.numpy as jnp
import nifty.re as jft
import numpy as np
from astropy import units as u
from jax import Array
from jax.typing import DTypeLike
from nifty.re import logger

from .color import Color
from .grid import Grid

SKY_KEY = "sky"
MAX_MISSING = 0.01  # passband fraction allowed outside sky coverage
N_FINE = 20001  # integration grid over the curve support
_TOL = 1e-9  # missing fraction treated as zero (integration round-off)


@dataclass(frozen=True)
class Transmission:
    """Throughput curve T(lam) of one band, zero outside its samples.

    Parameters
    ----------
    lam_um : wavelength samples, microns, ascending.
    T : throughput at lam_um, dimensionless, non-negative. Overall scale is
        irrelevant, the weights are normalised.
    """

    lam_um: np.ndarray
    T: np.ndarray

    def __post_init__(self) -> None:
        lam = np.asarray(self.lam_um, float)
        T = np.asarray(self.T, float)
        if lam.ndim != 1 or lam.shape != T.shape or lam.size < 2:
            raise ValueError(
                f"lam_um {lam.shape} and T {T.shape} must be 1-D, same length"
            )
        if np.any(np.diff(lam) <= 0):
            raise ValueError("lam_um must be strictly ascending")
        if np.any(T < 0) or not np.any(T > 0):
            raise ValueError("T must be non-negative and not all zero")
        object.__setattr__(self, "lam_um", lam)
        object.__setattr__(self, "T", T)

    @classmethod
    def tophat(cls, lo_um: float, hi_um: float) -> "Transmission":
        """Unit box over [lo_um, hi_um]. Fallback for a band without a measured curve."""
        return cls(np.array([lo_um, hi_um], float), np.array([1.0, 1.0]))


@dataclass(frozen=True)
class BandWeights:
    """Weights of one band on the sky channels.

    Parameters
    ----------
    sl : contiguous sky channel slice the band draws from.
    w : (n_sl,) weights on those channels, sum to 1.
    """

    sl: slice
    w: np.ndarray


def _bounds_um(spectral: Color) -> np.ndarray:
    """(n_ch, 2) channel bounds in microns, each row ascending."""
    b = spectral.to(u.um, equivalencies=u.spectral()).value
    return np.sort(np.atleast_2d(b), axis=1)


def band_weights(
    spectral: Color,
    transmission: Transmission,
    max_missing: float = MAX_MISSING,
    name: str = "band",
) -> BandWeights:
    """Integrate T(lam) / lam over every sky channel.

    Parameters
    ----------
    spectral : sky channel bounds.
    transmission : the band's curve.
    max_missing : largest fraction of the passband allowed outside the sky
        channels before raising. Smaller non-zero fractions log a warning.
    name : band name for messages.

    Returns
    -------
    BandWeights with normalised weights over the channels the curve touches.

    Raises
    ------
    ValueError
        More than ``max_missing`` of the weighted passband lies outside the
        sky channels, including gaps between them.
    """
    bounds = _bounds_um(spectral)
    lam, T_curve = transmission.lam_um, transmission.T
    # channel edges inside the support join the fine grid, so the per-channel
    # integrals partition the total exactly
    inner = bounds.ravel()
    inner = inner[(inner > lam[0]) & (inner < lam[-1])]
    fine = np.union1d(np.linspace(lam[0], lam[-1], N_FINE), inner)
    g = np.interp(fine, lam, T_curve) / fine
    total = np.trapezoid(g, fine)
    per_ch = np.zeros(len(bounds))
    for i, (lo, hi) in enumerate(bounds):
        m = (lo <= fine) & (fine <= hi)
        if m.sum() > 1:
            per_ch[i] = np.trapezoid(g[m], fine[m])
    missing = 1.0 - per_ch.sum() / total
    missing = 0.0 if missing < _TOL else missing
    if missing > max_missing:
        raise ValueError(
            f"{name}: {100 * missing:.2f}% of the passband lies outside the sky "
            f"channels (allowed {100 * max_missing:.2f}%)"
        )
    if missing > 0:
        logger.warning(
            f"{name}: {100 * missing:.3f}% of the passband lies outside the sky "
            "channels, the band image is renormalised onto the covered part"
        )
    nz = np.flatnonzero(per_ch)
    sl = slice(int(nz[0]), int(nz[-1]) + 1)
    return BandWeights(sl, per_ch[sl] / per_ch[sl].sum())


class SkyFilter(jft.Model):
    """{sky_key: (n_ch, ny, nx)} -> {band: (ny, nx)}, one band-averaged plane per band.

    Parameters
    ----------
    grid : sky grid; its spectral axis defines the channels.
    bands : band name -> transmission curve.
    sky_key : key of the sky cube in the input.
    dtype : dtype of the sky cube and the weights.
    max_missing : see `band_weights`.
    """

    def __init__(
        self,
        grid: Grid,
        bands: dict[str, Transmission],
        sky_key: str = SKY_KEY,
        dtype: DTypeLike = jnp.float32,
        max_missing: float = MAX_MISSING,
    ) -> None:
        self.grid = grid
        self.sky_key = sky_key
        self.weights = {
            k: band_weights(grid.spectral, t, max_missing, name=k)
            for k, t in bands.items()
        }
        self._w = {k: jnp.asarray(bw.w, dtype=dtype) for k, bw in self.weights.items()}
        n_ch = _bounds_um(grid.spectral).shape[0]
        shape = (n_ch, *grid.spatial.shape_yx)
        super().__init__(domain={sky_key: jft.ShapeWithDtype(shape, dtype)})

    def __call__(self, x: jft.Vector | dict[str, Array]) -> dict[str, Array]:
        """Weighted sum of the sky channels for every band.

        Parameters
        ----------
        x : input dict holding the sky cube under ``sky_key``.

        Returns
        -------
        dict band name -> (ny, nx) plane.
        """
        sky = x[self.sky_key]
        return {
            k: jnp.tensordot(self._w[k], sky[bw.sl], axes=1)
            for k, bw in self.weights.items()
        }
