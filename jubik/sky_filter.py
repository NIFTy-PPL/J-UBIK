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

__all__ = ["MAX_MISSING", "SKY_KEY", "Band", "SkyFilter", "Transmission"]

SKY_KEY = "sky"
MAX_MISSING = 0.01  # passband fraction allowed outside sky coverage
_N_FINE = 20001  # integration grid over the curve support
_TOL = 1e-9  # missing fraction treated as zero (integration round-off)


@dataclass(frozen=True)
class Transmission:
    """Throughput curve T(lam) of one output bin, zero outside its samples.

    Parameters
    ----------
    wavelength_um : np.ndarray
        Wavelength samples, microns, strictly ascending, at least two.
    throughput : np.ndarray
        Throughput at `wavelength_um`, dimensionless, non-negative, not all zero.
        Overall scale is irrelevant, the weights are normalised.
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


class _BandWeights(NamedTuple):
    """Weights of one band on the sky channels.

    Parameters
    ----------
    channels : slice
        Contiguous sky channel slice the band draws from.
    weights : np.ndarray
        (n_out, n_channels) weights of every output bin on those channels, rows
        sum to 1.
    """

    channels: slice
    weights: np.ndarray

    @staticmethod
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
            See `_BandWeights.from_band`.
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

    @classmethod
    def from_band(
        cls,
        spectral: Color,
        band: Band,
        max_missing: float = MAX_MISSING,
        name: str = "band",
    ) -> "_BandWeights":
        """Integrate every curve of a band over the sky channels.

        Parameters
        ----------
        spectral : Color
            Sky channel bounds.
        band : Transmission | Sequence[Transmission]
            One Transmission, or one per output bin of the band.
        max_missing : float
            Largest fraction of an output bin's passband allowed outside the sky
            channels before raising. Smaller non-zero fractions log a warning.
        name : str
            Band name for messages.

        Returns
        -------
        _BandWeights
            Weights over the contiguous channel range the band's curves touch.

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
                cls._channel_integrals(
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
        return cls(channels, rows[:, channels])


class SkyFilter(jft.Model):
    """{sky_key: (n_ch, ny, nx)} -> {band: (ny, nx) plane or (n_out, ny, nx) cube}.

    A band given as one Transmission outputs a plane, a band given as a
    sequence outputs one cube bin per curve, in order. Every likelihood that
    lives on a band's spectral binning reads its sky under the band's key.

    Parameters
    ----------
    grid : Grid
        Sky grid; its spectral axis defines the channels.
    bands : dict[str, Band]
        Band name -> one Transmission, or one per output bin.
    sky_key : str
        Key of the sky cube in the input.
    dtype : DTypeLike
        Dtype of the sky cube.
    max_missing : float
        See `_BandWeights.from_band`.
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
            key: _BandWeights.from_band(grid.spectral, band, max_missing, name=key)
            for key, band in bands.items()
        }
        self._is_plane = {
            key: isinstance(band, Transmission) for key, band in bands.items()
        }
        n_ch = np.atleast_1d(grid.spectral.center).size
        shape = (n_ch, *grid.spatial.shape_yx)
        super().__init__(domain={sky_key: jft.ShapeWithDtype(shape, dtype)})

    def __call__(self, x: jft.Vector | dict[str, Array]) -> dict[str, Array]:
        """Re-bin the sky cube onto every band.

        Parameters
        ----------
        x : jft.Vector | dict[str, Array]
            Input dict holding the sky cube under ``sky_key``.

        Returns
        -------
        dict[str, Array]
            Band name -> (ny, nx) plane or (n_out, ny, nx) cube.
        """
        sky = x[self.sky_key]
        out = {}
        for key, band_weights in self.weights.items():
            weights = jnp.asarray(band_weights.weights, dtype=sky.dtype)
            y = jnp.tensordot(weights, sky[band_weights.channels], axes=(1, 0))
            out[key] = y[0] if self._is_plane[key] else y
        return out
