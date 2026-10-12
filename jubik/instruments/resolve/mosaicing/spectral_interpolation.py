"""Interpolation of canonical radio skies onto observation frequencies."""

import numpy as np
import jax.numpy as jnp
import nifty.re as jft
from astropy import units as u

from ....color import get_2d_binbounds
from ....grid import Grid


class SpectralInterpolator(jft.Model):
    """Linear brightness interpolation along axis 2 of a canonical sky.

    With gap_mode="nearest" (default), connected bins interpolate between
    their centers and gaps use the nearest center (ties choose the lower
    frequency). With gap_mode="linear", interpolation bridges gaps. Outer
    edges always use nearest endpoint brightness. A single bin
    broadcasts, including grids with infinite single-bin bounds.
    """

    def __init__(
        self,
        sky_grid: Grid,
        target_frequencies: u.Quantity,
        dtype,
        gap_mode: str = "nearest",
    ):
        if gap_mode not in ("nearest", "linear"):
            raise ValueError("gap_mode must be 'nearest' or 'linear'.")

        self.gap_mode = gap_mode
        bounds = get_2d_binbounds(sky_grid.spectral, u.Hz)
        target = np.asarray(target_frequencies.to_value(u.Hz), dtype=float)

        if target.ndim != 1 or not target.size or not np.all(np.isfinite(target)):
            raise ValueError("Target frequencies must be a nonempty finite 1-D array.")
        if not len(bounds) or np.any(bounds[:, 0] >= bounds[:, 1]):
            raise ValueError("Sky spectral bins must have positive widths.")
        if np.any(bounds[:-1, 1] > bounds[1:, 0]):
            raise ValueError("Sky spectral bins must be ordered and nonoverlapping.")

        # Use arithmetic bin centers. For a single bin, its center is irrelevant:
        # every target selects that bin, so use 0 to support infinite bounds.
        centers = bounds.mean(axis=1) if len(bounds) > 1 else np.zeros(1)

        if not np.all(np.isfinite(centers)) or np.any(np.diff(centers) <= 0):
            raise ValueError("Multiple sky bins must have finite increasing centers.")

        # Find the nearest source center for each target.
        # Equal-distance ties select the lower-frequency center.
        nearest = np.abs(target[:, None] - centers).argmin(axis=1)
        # Initialize nearest-center fallback: identical source indices and zero weight.
        # The loop replaces this mapping for targets handled by interpolation.
        # Unchanged entries return nearest brightness on either side of the grid
        # and, in gap_mode="nearest", inside gaps.
        left = nearest.copy()
        right = nearest.copy()
        weights = np.zeros(target.shape)

        # In nearest mode, split at gaps so interpolation cannot bridge them.
        # In linear mode, process all source centers together, including across gaps.
        splits = (
            np.flatnonzero(bounds[:-1, 1] != bounds[1:, 0]) + 1
            if gap_mode == "nearest"
            else np.array([], dtype=int)
        )
        # Process each source-index group.
        for indices in np.split(np.arange(len(bounds)), splits):
            # Groups with one center retain the nearest fallback.
            if len(indices) < 2:
                continue
            # Select targets within this group's outer bounds: [lower, upper).
            selected = (target >= bounds[indices[0], 0]) & (
                target < bounds[indices[-1], 1]
            )
            # Positions of selected targets in the full target array.
            positions = np.flatnonzero(selected)
            # Find insertion positions within this group's centers.
            hi = np.searchsorted(centers[indices], target[selected], side="right")
            # Restrict positions so (hi - 1, hi) is a valid pair at both endpoints.
            hi = np.clip(hi, 1, len(indices) - 1)
            # Convert local pair positions into global source-bin indices.
            lo_idx, hi_idx = indices[hi - 1], indices[hi]
            # Store the source-bin pair for each selected target.
            left[positions], right[positions] = lo_idx, hi_idx
            # Fractional distance between the selected centers.
            # Clip to 0 below the first center and 1 above the last center,
            # giving constant endpoint brightness rather than extrapolation.
            # For the hi clipped case (a frequency lying before the first bin center
            # but inside the subgrid) the unclipped weight would be negative and is
            # clipped to zero; in this case only the left interpolation point will
            # contribute which is the nearest center
            weights[positions] = np.clip(
                (target[selected] - centers[lo_idx])
                / (centers[hi_idx] - centers[lo_idx]),
                0,
                1,
            )
        self.left = jnp.asarray(left)
        self.right = jnp.asarray(right)
        # Broadcast frequency weights over polarization, time, y, and x.
        self.weights = jnp.asarray(weights)[None, None, :, None, None]
        self.target_frequencies = target_frequencies.to(u.Hz).copy()
        super().__init__(domain=jft.ShapeWithDtype(sky_grid.shape, dtype))

    def __call__(self, sky):
        left = jnp.take(sky, self.left, axis=2)
        right = jnp.take(sky, self.right, axis=2)
        return left + self.weights * (right - left)
