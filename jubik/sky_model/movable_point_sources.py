# Copyright(C) 2026
# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause
# Authors: Matteo Guardiani

"""Movable point sources with inferred positions and power-law spectra.

A catalogue of ``n`` sources, each with a fixed anchor, a latent position
offset, a positive pivot flux and a spectrum, deposited onto a Cartesian grid
or a HEALPix RING map. The deposited flux is divided by the pixel solid angle,
so the model returns a surface brightness (flux per bin per steradian when the
pixel sizes are angles in radians).

See also
--------
M. Guardiani et al., "Latent-space field tension for astrophysical
component detection", A&A 703, A203 (2025),
`<https://doi.org/10.1051/0004-6361/202556103>`_. It describes this point-source
model and its sub-pixel localization.
Consider citing this paper if you use these point sources.
"""

from itertools import product

import jax.numpy as jnp
import nifty.re as jft
import numpy as np
from nifty.re.num.stats_distributions import normal_prior
from numpy.typing import ArrayLike

from .multifrequency.spectral_product_mf_point_sources import (
    MultiFrequencyInvGammaSky,
    PointSourceSpectralIndex,
    _point_source_spatial_shape,
)
from .multifrequency.spectral_product_utils.distribution_or_default import (
    build_distribution_or_default,
)
from .multifrequency.spectral_product_utils.frequency_deviations import (
    build_frequency_deviations_model_with_degeneracies,
)

FLUX_PRIORS = {
    "lognormal": (jft.lognormal_prior, jft.lognormal_invprior),
    "invgamma": (jft.invgamma_prior, jft.invgamma_invprior),
}


class MovablePointSources(jft.Model):
    """A fixed-size catalogue of point sources deposited onto a sky map.

    ``__call__`` returns the sky of shape ``(n_frequencies, *spatial)``. The
    per-source quantities are available through `positions`, `fluxes`
    (at the reference frequency) and `spectra` (per frequency bin).
    """

    def __init__(
        self,
        offsets: jft.Model,
        spectra: MultiFrequencyInvGammaSky,
        to_positions: callable,
        deposit: callable,
        flux_invprior: callable,
    ):
        self._offsets = offsets
        self.spectral_model = spectra
        """The spectral submodel; its `log_spectral_behavior` and
        `spectral_index_deviations` domains give the spectral latents to
        freeze."""
        self._to_positions = to_positions
        self._deposit = deposit
        self.flux_invprior = flux_invprior
        """Map pivot fluxes to the flux latents (latent 0 is the prior
        median)."""
        super().__init__(domain=offsets.domain | spectra.domain)

    def positions(self, p):
        """Source positions, shape ``(n, 2)``: (lon, lat) in radians for the
        spherical geometry, shape ``(n, ndim)`` coordinates in the units of
        `distances` for the Cartesian one."""
        return self._to_positions(self._offsets(p))

    def fluxes(self, p):
        """Flux in the reference bin, shape ``(n,)``."""
        return self.spectral_model.reference_frequency_distribution(p)

    def spectra(self, p):
        """Flux per model bin, shape ``(n_frequencies, n)``."""
        return self.spectral_model(p)

    def position_summary(self, samples):
        """Posterior mean position and the standard deviation of the
        offsets, both of the shape of `positions`.

        The mean is taken in each anchor's tangent plane, so it is safe
        across the longitude wrap.
        """
        offsets = jnp.stack([self._offsets(s) for s in samples])
        return self._to_positions(offsets.mean(axis=0)), offsets.std(axis=0)

    def __call__(self, p):
        return self._deposit(self.positions(p), self.spectra(p))


def build_movable_point_sources(
    prefix: str,
    anchors: ArrayLike,
    position_std: float,
    flux_prior: tuple,
    geometry: str,
    shape: tuple[int, ...],
    log_frequencies: ArrayLike,
    reference_frequency_index: int,
    spectral_settings: dict,
    distances: tuple[float, ...] | None = None,
) -> MovablePointSources:
    """Build an `MovablePointSources` model.

    Each source has a latent position offset ``Normal(0, position_std)``
    around its anchor, a pivot flux from `flux_prior`, a normal spectral
    index and optional Wiener spectral deviations pinned to zero at the
    reference frequency. Anchors are fixed: to add sources, append anchors
    and rebuild. Each per-source latent has one source axis, so the latents
    of existing sources carry over when that axis is zero-padded. The
    source axis is the first axis of the position offsets, flux and
    spectral-index latents and the last axis of the deviation excitations;
    the deviation amplitude is shared.

    Parameters
    ----------
    prefix: str
        Prefix for the latent names.
    anchors: ArrayLike
        Fixed reference point of each source. The inferred position is the
        anchor moved by the latent offset, so the anchor is the prior mean
        of the position and, on the sphere, the tangent point of the
        offset. Anchors never move during or between fits; latents of
        existing sources stay valid only because of that.
        Spherical: shape ``(n, 2)``, (lon, lat) in radians. Cartesian:
        shape ``(n, len(shape))``, a coordinate per spatial axis, in the
        units of `distances`, measured from the outer corner of pixel 0, so
        pixel ``i`` covers ``[i * d, (i + 1) * d)``.
    position_std: float
        Standard deviation of each offset component. Spherical: in radians,
        as gnomonic tangent-plane coordinates (east, north) at the anchor.
    flux_prior: tuple
        ``("lognormal", mean, std)`` or ``("invgamma", a, scale)``. The
        flux is integrated over one model bin.
    geometry: str
        ``"cartesian"`` or ``"spherical"``.
    shape: tuple[int, ...]
        Cartesian: the spatial shape. Spherical: ``(nside,)``.
    log_frequencies: ArrayLike
        Log energies of the model bins.
    reference_frequency_index: int
        Index of the pivot bin in `log_frequencies`.
    spectral_settings: dict
        ``mean``: normal prior (mean, std) of the per-source spectral index.
        ``deviations``: optional settings for
        `build_frequency_deviations_model_with_degeneracies`.
    distances: tuple[float, ...], optional
        Cartesian pixel sizes, one per spatial axis. Required for
        ``geometry="cartesian"``.
    """
    anchors = jnp.asarray(anchors, dtype=float)
    log_frequencies = np.asarray(log_frequencies, dtype=float)
    spatial_shape = _point_source_spatial_shape(shape, geometry)
    n_sources = anchors.shape[0]
    ndim = 2 if geometry == "spherical" else len(spatial_shape)
    if anchors.ndim != 2 or anchors.shape[1] != ndim:
        raise ValueError(f"anchors must have shape (n, {ndim})")

    if flux_prior[0] not in FLUX_PRIORS:
        raise ValueError(
            f"flux_prior must be one of {tuple(FLUX_PRIORS)}, got "
            f"{flux_prior[0]!r}"
        )
    prior, invprior = FLUX_PRIORS[flux_prior[0]]
    flux = build_distribution_or_default(
        flux_prior[1:], f"{prefix}_flux", prior, shape=(n_sources,)
    )
    offsets = build_distribution_or_default(
        (0.0, position_std),
        f"{prefix}_position_offsets",
        normal_prior,
        shape=anchors.shape,
    )

    spectral_index = PointSourceSpectralIndex(
        log_frequencies=log_frequencies,
        mean=build_distribution_or_default(
            spectral_settings["mean"],
            f"{prefix}_spectral_index_mean",
            normal_prior,
            shape=(n_sources,),
        ),
        reference_frequency_index=reference_frequency_index,
        spatial_shape=(n_sources,),
    )
    deviations = None
    if log_frequencies.size > 1:
        deviations = build_frequency_deviations_model_with_degeneracies(
            (n_sources,),
            log_frequencies,
            reference_frequency_index,
            spectral_settings.get("deviations"),
            prefix=f"{prefix}_spectral",
        )
    spectra = MultiFrequencyInvGammaSky(flux, spectral_index, deviations)

    if geometry == "spherical":
        nside = int(shape[0])
        omega = 4 * np.pi / spatial_shape[0]

        def to_positions(o):
            return gnomonic_to_lonlat(anchors, o)

        def deposit(positions, spectra):
            pix, wgt = healpix_bilinear_weights(
                nside, jnp.pi / 2 - positions[:, 1], positions[:, 0]
            )
            sky = jnp.zeros((spectra.shape[0],) + spatial_shape, spectra.dtype)
            sky = sky.at[:, pix.reshape(-1)].add(
                (spectra[:, :, None] * wgt).reshape(spectra.shape[0], -1)
            )
            return sky / omega

    else:
        if distances is None or len(distances) != len(spatial_shape):
            raise ValueError("cartesian geometry needs one distance per axis")
        distances = jnp.asarray(distances, dtype=float)
        omega = float(jnp.prod(distances))

        def to_positions(o):
            return anchors + o

        def deposit(positions, spectra):
            return deposit_bilinear_cartesian(
                positions / distances, spectra, spatial_shape
            ) / omega

    return MovablePointSources(
        offsets, spectra, to_positions, deposit, invprior(*flux_prior[1:])
    )


def deposit_bilinear_cartesian(coordinates, spectra, shape):
    """Deposit sources onto a grid with bilinear weights.

    Equivalent to integrating a one-pixel square centred on each source over
    the pixels (anomaly_detection's square sources). Flux falling outside
    the grid is dropped, so a source beyond the outer pixel centres loses
    flux instead of being piled up on the edge.

    Parameters
    ----------
    coordinates: ArrayLike
        Shape ``(n, ndim)``, in pixel units; pixel ``i`` covers ``[i, i+1)``.
    spectra: ArrayLike
        Shape ``(n_frequencies, n)``.
    shape: tuple[int, ...]
        Spatial grid shape.

    Returns
    -------
    sky: ArrayLike
        Shape ``(n_frequencies, *shape)``; the sum over pixels is the total
        deposited flux.
    """
    u = coordinates - 0.5  # in units where pixel centres are integers
    i0 = jnp.floor(u).astype(int)
    w1 = u - i0
    upper = jnp.array(shape) - 1
    sky = jnp.zeros((spectra.shape[0],) + tuple(shape), spectra.dtype)
    for corner in product((0, 1), repeat=len(shape)):
        idx = i0 + jnp.array(corner)
        w = jnp.prod(jnp.where(jnp.array(corner), w1, 1 - w1), axis=-1)
        w = jnp.where(jnp.all((idx >= 0) & (idx <= upper), axis=-1), w, 0.0)
        idx = jnp.clip(idx, 0, upper)
        sky = sky.at[(slice(None),) + tuple(idx.T)].add(spectra * w)
    return sky


def gnomonic_to_lonlat(anchors, offsets):
    """Map tangent-plane offsets (east, north) at anchors (lon, lat) onto the
    sphere with the gnomonic projection. All angles in radians.

    Returns (lon, lat) of shape ``(n, 2)``, with lon in ``(-pi, pi]``.
    """
    cos_lon, sin_lon = jnp.cos(anchors[:, 0]), jnp.sin(anchors[:, 0])
    cos_lat, sin_lat = jnp.cos(anchors[:, 1]), jnp.sin(anchors[:, 1])
    east, north = offsets[:, 0], offsets[:, 1]
    # anchor direction + east * e_lon + north * e_lat; the norm cancels below
    radial = cos_lat - north * sin_lat
    x = radial * cos_lon - east * sin_lon
    y = radial * sin_lon + east * cos_lon
    z = sin_lat + north * cos_lat
    lonlat = jnp.arctan2(y, x), jnp.arctan2(z, jnp.hypot(x, y))
    return jnp.stack(lonlat, axis=-1)


def healpix_bilinear_weights(nside, theta, phi):
    """Bilinear interpolation weights on a HEALPix RING map.

    JAX port of HEALPix C++ ``Healpix_Base::get_interpol`` (healpy's
    ``get_interp_weights``), with the ring geometry taken from
    ``ducc0.healpix.Healpix_Base.sht_info()``. For each point, two pixels in
    the ring above and two in the ring below, linear in longitude within a
    ring and in colatitude between rings. Above the first ring (below the
    last) the pole takes the place of the missing ring: its weight is shared
    by the four pixels of the first (last) ring, which surround the pole.

    Parameters
    ----------
    nside: int
    theta, phi: ArrayLike
        Colatitude in ``[0, pi]`` and longitude (any value) in radians.

    Returns
    -------
    pix, wgt: ArrayLike
        Shape ``theta.shape + (4,)``. Weights are nonnegative and sum to one.
    """
    from ducc0.healpix import Healpix_Base

    info = Healpix_Base(nside, "RING").sht_info()
    ring_theta = jnp.asarray(info["theta"])
    nphi = jnp.asarray(info["nphi"].astype(np.int64))
    phi0 = jnp.asarray(info["phi0"])
    start = jnp.asarray(info["ringstart"].astype(np.int64))
    nring, npix = ring_theta.size, 12 * nside**2

    def two_in_ring(r):
        t = (phi - phi0[r]) * nphi[r] / (2 * jnp.pi)
        i = jnp.floor(t)
        w = t - i
        i = i.astype(np.int64)
        pix = start[r][..., None] + jnp.stack([i, i + 1], axis=-1) % nphi[r][..., None]
        return pix, jnp.stack([1 - w, w], axis=-1)

    below = jnp.searchsorted(ring_theta, theta, side="right")
    ra, rb = jnp.clip(below - 1, 0, nring - 1), jnp.clip(below, 0, nring - 1)
    pa, wa = two_in_ring(ra)
    pb, wb = two_in_ring(rb)
    ta, tb = ring_theta[ra], ring_theta[rb]

    # Colatitude weight of the lower ring; at a cap the pole is the other end
    north, south = below == 0, below == nring
    upper = jnp.where(north, 0.0, ta)
    lower = jnp.where(south, jnp.pi, tb)
    wt = ((theta - upper) / jnp.where(north | south, lower - upper, tb - ta))[..., None]
    north, south = north[..., None], south[..., None]
    # At a cap the missing ring is the four polar pixels: the two found in the
    # nearest ring and the two half a turn away share the pole's weight.
    pole = jnp.where(north, 1 - wt, wt) / 4
    pa = jnp.where(north, (pb + 2) % 4, pa)
    pb = jnp.where(south, (pa + 2) % 4 + npix - 4, pb)  # npix - 4 = 0 mod 4
    wa = jnp.where(north, pole, wa * (1 - wt) + south * pole)
    wb = jnp.where(south, pole, wb * wt + north * pole)
    return jnp.concatenate([pa, pb], -1), jnp.concatenate([wa, wb], -1)
