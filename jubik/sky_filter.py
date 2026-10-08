# SPDX-License-Identifier: BSD-2-Clause
# Authors: Julian Rüstig

# Copyright(C) 2026 Max-Planck-Society

# %

"""Re-bin the sky cube onto the spectral bins an instrument measures.

The sky model produces one cube with the spectral channels of ``grid.spectral``.
Instruments do not see those channels. An imaging filter sees one broad band,
weighted by its throughput curve; a spectrograph sees its own bins. SkyFilter
turns the sky cube into those instrument bins, so a likelihood can be built on
the data's own spectral binning.

Two public objects:

``Transmission(wavelength_um, throughput)``
    The throughput curve of one instrument bin: wavelength samples in microns
    and the dimensionless throughput at each sample, down to zero at both ends.
    This is all the instrument has to provide.

``SkyFilter(grid, bands)``
    A ``jft.Model`` from ``{sky_key: (n_ch, ny, nx)}`` to one array per band.
    ``bands`` maps a band name to one Transmission (an imaging filter, output
    ``(ny, nx)``) or to a sequence of them, one per data bin (a spectrograph,
    output ``(n_out, ny, nx)`` in the order given). Each output is the weighted
    sum of the sky channels, with the weight of channel i being the integral of
    ``throughput / wavelength`` over that channel (photon-counted F_nu data),
    normalised to one. The output is therefore the band-averaged sky in the
    units of the sky.

What a consumer on the data side does:

1. Build a Transmission per band from its own calibration data, for example
   ``jwst_transmission("F444W")`` in ``jubik.instruments.jwst.data.throughput``.
2. Hand them to ``SkyFilter(grid, {"F444W": curve, ...})``. Construction checks
   that the sky channels cover every passband; a gap above ``max_missing``
   raises, a smaller one warns.
3. Build the likelihood on ``sky_filter.target["F444W"]``, i.e. read the sky
   under the band name, and connect it with
   ``connect_likelihood_to_model(likelihood, sky_filter)``. Everything spatial
   (PSF, pointing, pixel integration, unit conversion) stays in the instrument
   response that follows.

``sky_filter.weights[name]`` exposes which sky channels feed a band and with
which weights, for plotting and bookkeeping.
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

# --------------------------------------------------------------------------- #
# Public: the contract an instrument fills and the model that consumes it
# --------------------------------------------------------------------------- #

SKY_KEY = "sky"
MAX_MISSING = 0.01  # passband fraction allowed outside sky coverage


@dataclass(frozen=True)
class Transmission:
    """Sampled throughput curve of one output bin.

    The curve is interpolated linearly between consecutive samples and is zero
    outside the first and last one. The samples are points along the curve,
    not bin edges; a tophat over [lo, hi] is the two samples ``[lo, hi]`` with
    throughput ``[1, 1]``, a measured filter is a few hundred samples.

    Parameters
    ----------
    wavelength_um : np.ndarray
        Wavelengths of the samples, microns, strictly ascending, at least two.
    throughput : np.ndarray
        Throughput at each sample, dimensionless, non-negative, not all zero.
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


# --------------------------------------------------------------------------- #
# Internals: integration of a curve over the sky channels
# --------------------------------------------------------------------------- #

_N_FINE = 20001  # integration grid over the curve support
_TOL = 1e-9  # missing fraction treated as zero (integration round-off)


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
