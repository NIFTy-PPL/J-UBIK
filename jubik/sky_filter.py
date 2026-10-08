# SPDX-License-Identifier: BSD-2-Clause
# Authors: Julian Rüstig

# Copyright(C) 2026 Max-Planck-Society

# %

"""SkyFilter: the sky owns the spectral channels, every band re-bins from them.

The sky is one F_nu cube on the channels of ``grid.spectral``. An instrument
provides one Transmission curve T(lam) per output bin and nothing else. The
weight of an output bin on sky channel i is the integral of T(lam) / lam over
that channel (photon-counted F_nu data), normalised so the weights sum to one.
A band is one output bin (an imaging filter, one plane) or a sequence of them
(the spectral bins of a spectrograph, one cube); the band image is the
weighted sum of the sky channels, in the units of the sky. Instrument-agnostic:
curves are passed in, never built here.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import NamedTuple

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
    """Throughput curve T(lam) of one output bin, zero outside its samples.

    Parameters
    ----------
    lam_um : wavelength samples, microns, strictly ascending, at least two.
    T : throughput at lam_um, dimensionless, non-negative, not all zero.
        Overall scale is irrelevant, the weights are normalised.
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


Band = Transmission | Sequence[Transmission]


def _channel_integrals(
    bounds: np.ndarray, t: Transmission, max_missing: float, name: str
) -> np.ndarray:
    """(n_ch,) integrals of T(lam) / lam over every channel, normalised to 1.

    Raises
    ------
    ValueError
        More than ``max_missing`` of the weighted passband lies outside the
        sky channels, including gaps between them.
    """
    lam, T_curve = t.lam_um, t.T
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
            "channels, the output is renormalised onto the covered part"
        )
    return per_ch / per_ch.sum()


class BandWeights(NamedTuple):
    """Weights of one band on the sky channels.

    sl : contiguous sky channel slice the band draws from.
    W : (n_out, n_sl) weights of every output bin on those channels, rows
        sum to 1.
    """

    sl: slice
    W: np.ndarray

    @classmethod
    def from_band(
        cls,
        spectral: Color,
        band: Band,
        max_missing: float = MAX_MISSING,
        name: str = "band",
    ) -> "BandWeights":
        """Integrate every curve of a band over the sky channels.

        Parameters
        ----------
        spectral : sky channel bounds.
        band : one Transmission, or one per output bin of the band.
        max_missing : largest fraction of an output bin's passband allowed
            outside the sky channels before raising. Smaller non-zero
            fractions log a warning.
        name : band name for messages.

        Returns
        -------
        BandWeights over the contiguous channel range the band's curves touch.

        Raises
        ------
        ValueError
            A curve exceeds ``max_missing``, or the band has no curves.
        """
        bounds = spectral.to(u.um, equivalencies=u.spectral()).value
        bounds = np.sort(np.atleast_2d(bounds), axis=1)
        curves = [band] if isinstance(band, Transmission) else list(band)
        if not curves:
            raise ValueError(f"{name}: band has no transmission curves")
        rows = np.stack(
            [
                _channel_integrals(
                    bounds, t, max_missing, name if len(curves) == 1 else f"{name}[{b}]"
                )
                for b, t in enumerate(curves)
            ]
        )
        nz = np.flatnonzero(rows.any(axis=0))
        sl = slice(int(nz[0]), int(nz[-1]) + 1)
        return cls(sl, rows[:, sl])


class SkyFilter(jft.Model):
    """{sky_key: (n_ch, ny, nx)} -> {band: (ny, nx) plane or (n_out, ny, nx) cube}.

    A band given as one Transmission outputs a plane, a band given as a
    sequence outputs one cube bin per curve, in order. Every likelihood that
    lives on a band's spectral binning reads its sky under the band's key.

    Parameters
    ----------
    grid : sky grid; its spectral axis defines the channels.
    bands : band name -> one Transmission, or one per output bin.
    sky_key : key of the sky cube in the input.
    dtype : dtype of the sky cube and the weights.
    max_missing : see `BandWeights.from_band`.
    """

    def __init__(
        self,
        grid: Grid,
        bands: dict[str, Band],
        sky_key: str = SKY_KEY,
        dtype: DTypeLike = jnp.float32,
        max_missing: float = MAX_MISSING,
    ) -> None:
        self.grid = grid
        self.sky_key = sky_key
        self.weights = {
            k: BandWeights.from_band(grid.spectral, b, max_missing, name=k)
            for k, b in bands.items()
        }
        self._plane = {k: isinstance(b, Transmission) for k, b in bands.items()}
        self._W = {k: jnp.asarray(bw.W, dtype=dtype) for k, bw in self.weights.items()}
        n_ch = np.atleast_1d(grid.spectral.center).size
        shape = (n_ch, *grid.spatial.shape_yx)
        super().__init__(domain={sky_key: jft.ShapeWithDtype(shape, dtype)})

    def __call__(self, x: jft.Vector | dict[str, Array]) -> dict[str, Array]:
        """Re-bin the sky cube onto every band.

        Parameters
        ----------
        x : input dict holding the sky cube under ``sky_key``.

        Returns
        -------
        dict band name -> (ny, nx) plane or (n_out, ny, nx) cube.
        """
        sky = x[self.sky_key]
        out = {}
        for k, bw in self.weights.items():
            y = jnp.tensordot(self._W[k], sky[bw.sl], axes=(1, 0))
            out[k] = y[0] if self._plane[k] else y
        return out
