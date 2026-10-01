# SPDX-License-Identifier: BSD-2-Clause
# Authors: Julian Rüstig

# Copyright(C) 2026 Max-Planck-Society

# %

"""SkyProjection: the sky owns the spectral channels, every band selects from them.

The sky is one cube on the channels of ``grid.spectral``. A band is a set of
output bins, each with a transmission curve T(lam); its projector weight on a
channel is the integral of T(lam) / lam over that channel (photon-counted F_nu
data), rows normalised to one. An imaging filter (FilterBand) gives one plane,
an IFU band (IfuBand) an identity sub-cube over the channels it overlaps.
Instrument-agnostic: curves are passed in, never loaded here.
"""

from collections.abc import Sequence
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


@dataclass(frozen=True)
class Transmission:
    """Curve T(lam), zero outside its support.

    Parameters
    ----------
    lam_um : wavelength samples, microns, ascending.
    T : transmission at lam_um.
    """

    lam_um: np.ndarray
    T: np.ndarray

    @classmethod
    def tophat(cls, lo_um: float, hi_um: float) -> "Transmission":
        """Unit box over [lo_um, hi_um]."""
        return cls(np.array([lo_um, hi_um], float), np.array([1.0, 1.0]))


@dataclass(frozen=True)
class FilterBand:
    """Imaging filter: one output plane, weighted by its transmission curve.

    Parameters
    ----------
    key : band key of the output.
    transmission : the filter curve.
    """

    key: str
    transmission: Transmission


@dataclass(frozen=True)
class IfuBand:
    """IFU band: the sky channels overlapping [lo_um, hi_um], unweighted.

    Parameters
    ----------
    key : band key of the output.
    lo_um, hi_um : wavelength range, microns; None means unbounded.
    """

    key: str
    lo_um: float | None = None
    hi_um: float | None = None


@dataclass(frozen=True)
class BandSelection:
    """One band's projector onto the sky channels.

    Parameters
    ----------
    key : band key.
    sl : contiguous sky channel slice.
    W : (n_out, n_sl) weights, rows sum to 1.
    plane : True when the output is squeezed to (ny, nx).
    """

    key: str
    sl: slice
    W: np.ndarray
    plane: bool


def _bounds_um(spectral: Color) -> np.ndarray:
    """(n_ch, 2) channel bounds in microns, each row ascending."""
    b = spectral.to(u.um, equivalencies=u.spectral()).value
    return np.sort(np.atleast_2d(b), axis=1)


def filter_selection(spectral: Color, band: FilterBand) -> BandSelection:
    """Weights of a filter curve over the sky channels.

    Parameters
    ----------
    spectral : sky channel bounds.
    band : the filter.

    Returns
    -------
    BandSelection with a single normalised row and ``plane=True``.

    Raises
    ------
    ValueError
        More than MAX_MISSING of the weighted passband falls outside the sky
        channels, including gaps between them.
    """
    bounds = _bounds_um(spectral)
    lam = np.asarray(band.transmission.lam_um, float)
    T_curve = np.asarray(band.transmission.T, float)
    fine = np.linspace(lam[0], lam[-1], N_FINE)
    T = np.interp(fine, lam, T_curve, left=0.0, right=0.0)
    g = T / fine
    total = np.trapezoid(g, fine)
    per_ch = np.array(
        [
            np.trapezoid(np.where((lo <= fine) & (fine < hi), g, 0.0), fine)
            for lo, hi in bounds
        ]
    )
    missing = 1.0 - per_ch.sum() / total
    if missing > MAX_MISSING:
        raise ValueError(
            f"{band.key}: {100 * missing:.2f}% of the passband lies outside the sky "
            f"channels (allowed {100 * MAX_MISSING:.2f}%)"
        )
    nz = np.flatnonzero(per_ch)
    sl = slice(int(nz[0]), int(nz[-1]) + 1)
    if len(nz) == 1 and np.ptp(bounds[nz[0]]) > np.ptp(fine[T > 0]):
        logger.warning(f"{band.key}: single sky channel wider than passband")
    W = (per_ch[sl] / per_ch[sl].sum())[None, :]
    return BandSelection(band.key, sl, W, plane=True)


def ifu_selection(spectral: Color, band: IfuBand) -> BandSelection:
    """Identity selection of the sky channels overlapping an IFU band.

    Parameters
    ----------
    spectral : sky channel bounds.
    band : the IFU band.

    Returns
    -------
    BandSelection with ``W = eye(n_sel)`` and ``plane=False``.
    """
    bounds = _bounds_um(spectral)
    lo = -np.inf if band.lo_um is None else band.lo_um
    hi = np.inf if band.hi_um is None else band.hi_um
    idx = np.flatnonzero((bounds[:, 1] > lo) & (bounds[:, 0] < hi))
    if idx.size == 0:
        raise ValueError(f"{band.key}: no sky channel overlaps [{lo}, {hi}] um")
    if idx[-1] - idx[0] + 1 != idx.size:
        raise ValueError(f"{band.key}: overlapping sky channels are not contiguous")
    sl = slice(int(idx[0]), int(idx[-1]) + 1)
    return BandSelection(band.key, sl, np.eye(idx.size), plane=False)


def select(spectral: Color, band: FilterBand | IfuBand) -> BandSelection:
    """Dispatch to filter_selection or ifu_selection."""
    if isinstance(band, FilterBand):
        return filter_selection(spectral, band)
    if isinstance(band, IfuBand):
        return ifu_selection(spectral, band)
    raise TypeError(f"unknown band type {type(band).__name__}")


class SkyProjection(jft.Model):
    """{sky_key: (n_ch, ny, nx)} -> {band key: (ny, nx) plane or (n_sel, ny, nx) sub-cube}.

    Parameters
    ----------
    grid : sky grid; its spectral axis defines the channels.
    bands : bands to project onto, keys unique.
    sky_key : key of the sky cube in the input.
    dtype : dtype of the sky cube and the weights.
    """

    def __init__(
        self,
        grid: Grid,
        bands: Sequence[FilterBand | IfuBand],
        sky_key: str = SKY_KEY,
        dtype: DTypeLike = jnp.float32,
    ) -> None:
        keys = [b.key for b in bands]
        if len(set(keys)) != len(keys):
            raise ValueError(f"duplicate band keys in {keys}")
        self.grid = grid
        self.sky_key = sky_key
        self.selections = {b.key: select(grid.spectral, b) for b in bands}
        self._bounds_um = _bounds_um(grid.spectral)
        # identity rows reduce to a slice, so IFU sub-cubes skip the matmul
        self._W = {
            k: None
            if s.W.shape[0] == s.W.shape[1]
            and np.array_equal(s.W, np.eye(s.W.shape[0]))
            else jnp.asarray(s.W, dtype=dtype)
            for k, s in self.selections.items()
        }
        n_ch = self._bounds_um.shape[0]
        shape = (n_ch, *grid.spatial.shape_yx)
        super().__init__(domain={sky_key: jft.ShapeWithDtype(shape, dtype)})

    def band_grid(self, key: str) -> Grid:
        """Sky grid restricted to the channels of one band.

        Parameters
        ----------
        key : band key.

        Returns
        -------
        Grid with the sky's spatial axis and the band's channel bounds.
        """
        sl = self.selections[key].sl
        return Grid(
            spatial=self.grid.spatial, spectral=Color(self._bounds_um[sl] * u.um)
        )

    def __call__(self, x: jft.Vector | dict[str, Array]) -> dict[str, Array]:
        """Project the sky cube onto every band.

        Parameters
        ----------
        x : input dict holding the sky cube under ``sky_key``.

        Returns
        -------
        dict band key -> (ny, nx) plane or (n_sel, ny, nx) sub-cube.
        """
        sky = x[self.sky_key]
        out = {}
        for key, sel in self.selections.items():
            W = self._W[key]
            y = sky[sel.sl] if W is None else jnp.tensordot(W, sky[sel.sl], axes=(1, 0))
            out[key] = y[0] if sel.plane else y
        return out
