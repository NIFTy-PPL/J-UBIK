# Copyright(C) 2026
# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause
# Authors: Matteo Guardiani

"""Normalized sky-space maps of the excitations of a `SpectralProductSky`.

An excitation field is a white latent ξ in harmonic space.
`to_excitation_map` synthesizes its active modes and divides by the
per-pixel prior standard deviation, so a prior sample is a white
unit-variance map. `with_excitation_map` writes an edited map back into a
latent position: exactly (Hartley transform) on
Cartesian grids, by least squares (ducc0 `pseudo_analysis`) on HEALPix grids,
where there are fewer coefficients than pixels.

See also
--------
M. Guardiani et al., "Latent-space field tension for astrophysical
component detection", A&A 703, A203 (2025),
`<https://doi.org/10.1051/0004-6361/202556103>`_. It introduces latent-space
tension: excitations that depart from their white prior flag a missing
sky component.
Consider citing this paper if you use these maps for model diagnostics.
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from ducc0.healpix import Healpix_Base
from ducc0.sht import pseudo_analysis
from nifty.re import Vector
from nifty.re.correlated_field import HEALPixGrid, RegularCartesianGrid, hartley

from .sht import _real_from_alm, get_healpix_sht


class ExcitationTransform:
    """Map view of the excitation field `key` on `grid`.

    Parameters
    ----------
    key: str
        Latent key of ξ. Its shape is the harmonic shape of `grid`, with
        optional leading axes (e.g. one ξ per deviation step), each mapped
        separately.
    grid: RegularCartesianGrid or HEALPixGrid
        NIFTy grid of the field, as returned by `make_grid`.
    zero_mode_active: bool
        Whether the field's amplitude keeps its k=0 (l=0) mode. The
        normalization counts the active modes only; `to_excitation_map` drops an
        inactive zero mode.
    nthreads: int
        ducc0 threads for the HEALPix transforms.

    Notes
    -----
    `with_excitation_map` keeps the zero-mode coefficient of the given position: a
    no-op on the map for an inactive zero mode, and for an active one it
    keeps the sky average (e.g. the sky-averaged spectral shape of the
    deviations) out of local edits.
    """

    def __init__(self, key, grid, *, zero_mode_active, nthreads=1):
        self.key = key
        self._zero = np.asarray(grid.harmonic_grid.power_distributor) == 0
        self._active = np.ones_like(self._zero) if zero_mode_active else ~self._zero
        # A white unit ξ synthesizes to a per-pixel variance equal to the
        # number of active modes (addition theorem; exact for full l-shells).
        self._norm = np.sqrt(self._active.sum())

        if isinstance(grid, RegularCartesianGrid):
            axes = tuple(range(-self._zero.ndim, 0))
            self._synthesis = partial(hartley, axes=axes)
            # The Hartley transform is its own inverse up to the pixel count.
            self._analysis = lambda m: np.asarray(hartley(m, axes)) / self._zero.size
        elif isinstance(grid, HEALPixGrid):
            nside, lmax, mmax = (
                grid.nside,
                grid.harmonic_grid.lmax,
                grid.harmonic_grid.mmax,
            )
            self._synthesis = jax.vmap(
                get_healpix_sht(nside, lmax, mmax, nthreads=nthreads)
            )
            self._analysis = partial(
                _healpix_analysis, nside=nside, lmax=lmax, mmax=mmax, nthreads=nthreads
            )
        else:
            raise TypeError(f"Unsupported grid type: {type(grid)!r}")

    def to_excitation_map(self, position):
        """Normalized map of ξ in `position`, shape `ξ.shape[:-n] + map shape`."""
        xi = jnp.where(self._active, _tree(position)[self.key], 0.0)
        lead = xi.shape[: xi.ndim - self._zero.ndim]
        xi = xi.reshape((-1,) + self._zero.shape)
        maps = self._synthesis(xi) / self._norm
        return maps.reshape(lead + maps.shape[1:])

    def with_excitation_map(self, position, map):
        """Copy of `position` whose ξ gives `map`, zero mode kept from `position`."""
        tree = dict(_tree(position))
        xi = self._analysis(np.asarray(map, dtype=float) * self._norm)
        tree[self.key] = jnp.asarray(np.where(self._zero, tree[self.key], xi))
        return Vector(tree) if isinstance(position, Vector) else tree


def _tree(position):
    return position.tree if isinstance(position, Vector) else position


def _healpix_analysis(maps, *, nside, lmax, mmax, nthreads):
    """Least-squares inverse of NIFTy's synthesis, on the last axis of `maps`."""
    lead, npix = maps.shape[:-1], maps.shape[-1]
    # NIFTy's synthesis is ducc0's times sqrt(4 pi), see `get_healpix_sht`.
    alm, istop, *_ = pseudo_analysis(
        map=maps.reshape(-1, 1, npix) / np.sqrt(4 * np.pi),
        lmax=lmax,
        mmax=mmax,
        spin=0,
        maxiter=100,
        epsilon=1e-13,
        nthreads=nthreads,
        **Healpix_Base(nside, "RING").sht_info(),
    )
    # LSMR stop codes: 0 zero map, 1 exact solution, 2 least-squares solution.
    if not np.isin(istop, (0, 1, 2)).all():
        raise RuntimeError(f"pseudo_analysis did not converge (istop={istop})")
    alm = alm[:, 0]
    L = lmax + 1
    x = np.empty((alm.shape[0], 2 * alm.shape[1] - L))
    for a, xi in zip(alm, x):
        # The inverse of `_alm_from_real` is its transpose `_real_from_alm`.
        _real_from_alm(a, xi, L)
    return x.reshape(lead + x.shape[1:])


def build_excitation_transforms(sky, *, nthreads=1):
    """`ExcitationTransform`s of the excitation fields of a `SpectralProductSky`.

    Returns a dict with `"spatial"`, and with several frequencies also
    `"spectral_index"` and, if the sky has deviations, `"deviations"` (one map
    per deviation step). Keys are read from the submodel domains.
    """
    grid = sky.spatial_amplitude.grid
    (key,) = sky.spatial_scaled_excitations.latent_space_xi.domain
    out = {
        "spatial": ExcitationTransform(
            key, grid, zero_mode_active=False, nthreads=nthreads
        )
    }
    if sky.log_spectral_behavior.relative_log_frequencies.shape[0] == 1:
        return out

    spectral = sky.log_spectral_behavior.spectral_scaled_excitations
    (key,) = spectral.latent_space_xi.domain
    out["spectral_index"] = ExcitationTransform(
        key, grid, zero_mode_active=False, nthreads=nthreads
    )
    if sky.spectral_index_deviations is not None:
        # The deviation domain also holds sigma; ξ is the field-shaped key.
        (key,) = (
            k
            for k, v in sky.spectral_index_deviations.domain.items()
            if v.shape[1:] == grid.harmonic_grid.shape
        )
        # `SpectralProductSky` keeps the k=0 mode of the deviation amplitude.
        out["deviations"] = ExcitationTransform(
            key, grid, zero_mode_active=True, nthreads=nthreads
        )
    return out
