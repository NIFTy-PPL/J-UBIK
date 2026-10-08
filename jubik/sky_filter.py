# SPDX-License-Identifier: BSD-2-Clause
# Authors: Julian Rüstig

# Copyright(C) 2026 Max-Planck-Society

# %

"""Re-bin the sky cube onto the spectral bins an instrument measures.

The sky model produces one cube with the spectral channels of ``grid.spectral``.
Instruments do not see those channels. An imaging filter sees one broad band, a
spectrograph its own bins. SkyFilter turns the sky cube into those instrument
bins by a fixed linear weighting of the sky channels, so a likelihood can be
built on the data's own spectral binning.

Two public objects:

``FilterWeights(channels, weights)``
    What an instrument provides per band: the contiguous slice of sky channels
    the band draws from and the weight of each. A 1-D ``weights`` is one output
    plane (an imaging filter), a 2-D ``(n_out, n_channels)`` one output bin per
    row (a spectrograph). How the weights follow from a throughput curve is the
    instrument's business, since detector type, sky units and calibration
    convention all enter there; ``jubik.instruments.jwst.data.throughput`` does
    it for photon-counted F_nu data.

``SkyFilter(grid, bands)``
    A ``jft.Model`` from ``{sky_key: (n_ch, ny, nx)}`` to one array per band,
    ``(ny, nx)`` for 1-D weights and ``(n_out, ny, nx)`` for 2-D. Weights that
    sum to one give the band-averaged sky in the units of the sky.

What a consumer on the data side does:

1. Compute a FilterWeights per band on ``grid.spectral``, for example
   ``throughput_weights(grid.spectral, jwst_transmission("F444W"))``.
2. Hand them to ``SkyFilter(grid, {"F444W": weights, ...})``.
3. Build the likelihood on ``sky_filter.target["F444W"]``, i.e. read the sky
   under the band name, and connect it with
   ``connect_likelihood_to_model(likelihood, sky_filter)``. Everything spatial
   (PSF, pointing, pixel integration, unit conversion) stays in the instrument
   response that follows.

``sky_filter.bands[name]`` is the FilterWeights of a band, for plotting and
bookkeeping.
"""

from dataclasses import dataclass

import jax.numpy as jnp
import nifty.re as jft
import numpy as np
from jax import Array
from jax.typing import DTypeLike

from .grid import Grid

__all__ = ["SKY_KEY", "FilterWeights", "SkyFilter"]

SKY_KEY = "sky"


@dataclass(frozen=True)
class FilterWeights:
    """Weights of one band on the sky channels.

    Parameters
    ----------
    channels : slice
        Contiguous sky channels the band draws from, explicit start and stop,
        unit step.
    weights : np.ndarray
        Weight of each of those channels: (n_channels,) for one output plane,
        (n_out, n_channels) for one output bin per row. Finite.
    """

    channels: slice
    weights: np.ndarray

    def __post_init__(self) -> None:
        channels = self.channels
        if (
            not isinstance(channels, slice)
            or channels.start is None
            or channels.stop is None
            or channels.step not in (None, 1)
            or channels.start < 0
            or channels.stop <= channels.start
        ):
            raise ValueError(
                f"channels must be a non-empty forward slice with explicit bounds, "
                f"got {channels}"
            )
        weights = np.asarray(self.weights, float)
        n_channels = channels.stop - channels.start
        if weights.ndim not in (1, 2) or weights.shape[-1] != n_channels:
            raise ValueError(
                f"weights {weights.shape} must be (n_channels,) or (n_out, n_channels) "
                f"with n_channels = {n_channels}"
            )
        if not np.all(np.isfinite(weights)):
            raise ValueError("weights must be finite")
        object.__setattr__(
            self, "channels", slice(int(channels.start), int(channels.stop))
        )
        object.__setattr__(self, "weights", weights)


class SkyFilter(jft.Model):
    """{sky_key: (n_ch, ny, nx)} -> {band: (ny, nx) plane or (n_out, ny, nx) cube}.

    Every likelihood that lives on a band's spectral binning reads its sky
    under the band's name.

    Parameters
    ----------
    grid : Grid
        Sky grid; its spectral axis defines the channels.
    bands : dict[str, FilterWeights]
        Band name -> the band's weights on the sky channels.
    sky_key : str
        Key of the sky cube in the input.
    dtype : DTypeLike
        Dtype of the sky cube.

    Raises
    ------
    ValueError
        A band's channels reach beyond the sky channels.
    """

    def __init__(
        self,
        grid: Grid,
        bands: dict[str, FilterWeights],
        sky_key: str = SKY_KEY,
        dtype: DTypeLike = jnp.float32,
    ) -> None:
        n_ch = np.atleast_1d(grid.spectral.center).size
        for name, band in bands.items():
            if band.channels.stop > n_ch:
                raise ValueError(
                    f"{name}: channels {band.channels} reach beyond the {n_ch} sky channels"
                )
        self.grid = grid
        self.sky_key = sky_key
        self.bands = dict(bands)
        shape = (n_ch, *grid.spatial.shape_yx)
        super().__init__(domain={sky_key: jft.ShapeWithDtype(shape, dtype)})

    def __call__(self, x: jft.Vector | dict[str, Array]) -> dict[str, Array]:
        """Weighted sum of the sky channels for every band.

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
        for name, band in self.bands.items():
            weights = jnp.asarray(band.weights, dtype=sky.dtype)
            out[name] = jnp.tensordot(
                weights, sky[band.channels], axes=(weights.ndim - 1, 0)
            )
        return out
