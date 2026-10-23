# Copyright(C) 2026
# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause
# Authors: Matteo Guardiani

"""Point-source candidates from excitation maps on HEALPix, and their handoff.

Array functions for one detect-relax-handoff pass:

1. score the posterior-mean normalized ξ maps (`multifrequency_score`, or the
   spatial map alone at a single frequency) and pick candidates among its
   local maxima (`select_candidates`);
2. relax the excitations around the candidates toward clipped white noise
   (`relaxation_weights`, `relax_position`); the relaxed posterior mean is the
   warm start of the refit;
3. integrate the diffuse emission the relaxation removes
   (`aperture_labels`, `removed_emission`), fit it with the mean source
   spectrum (`pivot_flux`) and accept or reject, with the per-round limit
   (`flux_acceptance`). One relaxation per draw covers all candidates; the
   warm start relaxes only the accepted ones. `select_and_accept` runs
   selection and acceptance together and re-selects around flux-rejected
   candidates.

Maps are in HEALPix RING order, angles in radians, positions are
(lon, lat) pairs. The relaxation is a port of sptLensing's
`spt_lensing.inference.substructure` with great-circle distances.

See also
--------
M. Guardiani et al., "Latent-space field tension for astrophysical
component detection", A&A 703, A203 (2025),
`<https://doi.org/10.1051/0004-6361/202556103>`_. It detects point sources from
the tension in the excitation maps.
Consider citing this paper if you use this detection.
"""

from functools import lru_cache

import numpy as np
from ducc0.healpix import Healpix_Base
from scipy.sparse import csr_array
from scipy.stats import chi


def multifrequency_score(xi_maps, p=0.6827):
    """`sqrt(sum_fields ξ̄**2) / chi.ppf(p, d)` over the `d` maps on axis 0.

    `xi_maps` stacks the posterior-mean normalized maps of the spatial,
    spectral-index and deviation-step fields. The score has no sign.
    """
    xi_maps = np.asarray(xi_maps)
    return np.sqrt(np.sum(xi_maps**2, axis=0)) / chi.ppf(p, xi_maps.shape[0])


@lru_cache
def _healpix(nside):
    """Pixel-centre unit vectors and the 8 neighbours (-1 where missing)."""
    base = Healpix_Base(nside, "RING")
    pixels = np.arange(base.npix())
    return base.pix2vec(pixels), base.neighbors(pixels)


def _nside(npix):
    nside = int(round(np.sqrt(npix / 12)))
    if 12 * nside**2 != npix:
        raise ValueError(f"{npix} is not a HEALPix pixel count")
    return nside


def _lonlat_to_vec(lonlat):
    lon, lat = np.asarray(lonlat, dtype=float).reshape(-1, 2).T
    return np.stack(
        (np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)), axis=-1
    )


def _angle(a, b):
    """Great-circle distances between the unit vectors `a` (n, 3) and `b` (k, 3)."""
    return np.arccos(np.clip(a @ b.T, -1.0, 1.0))


def local_maxima(score):
    """Pixels larger than all their neighbours.

    Ties between neighbours go to the lower index; a plateau can still give
    several maxima (a positive `min_separation` in `select_candidates` usually
    merges them).
    """
    score = np.asarray(score)
    _, neighbours = _healpix(_nside(score.size))
    pixels = np.arange(score.size)[:, None]
    other = score[neighbours]
    higher = (score[:, None] > other) | (
        (score[:, None] == other) & (pixels < neighbours)
    )
    return np.flatnonzero(np.all(higher | (neighbours < 0), axis=1))


def select_candidates(
    score,
    *,
    threshold,
    min_separation=0.0,
    source_lonlat=(),
    exclusion_radius=0.0,
    excluded=None,
):
    """Candidate pixels among the local maxima of `score`, with the reason
    each one is kept or rejected.

    The maxima are taken by descending score, ties by pixel index. A maximum
    is rejected, in this order, as `"below_threshold"` (score not above
    `threshold`), `"existing_source"` (closer than `exclusion_radius` to a
    source at `source_lonlat`), `"exclusion_region"` (`excluded[pixel]`) or
    `"too_close"` (closer than `min_separation` to an accepted candidate;
    the default 0 turns this off, so a plateau can give neighbouring
    candidates). The others are `"accepted"`. The per-round limit comes after the flux
    checks, in `flux_acceptance`.

    Returns
    -------
    pixels, reasons: np.ndarray
        All local maxima in that order, and one reason string each.
    """
    if threshold < 0:
        raise ValueError("candidates are positive maxima: need threshold >= 0")
    score = np.asarray(score)
    vectors, _ = _healpix(_nside(score.size))
    pixels = local_maxima(score)
    pixels = pixels[np.lexsort((pixels, -score[pixels]))]
    # Only maxima above the threshold can be accepted, and they are few: with
    # many sources, a distance matrix over all maxima would not fit in memory.
    above = score[pixels] > threshold
    near_source = np.zeros(pixels.size, dtype=bool)
    near_source[above] = np.any(
        _angle(vectors[pixels[above]], _lonlat_to_vec(source_lonlat))
        < exclusion_radius,
        axis=1,
    )
    accepted, reasons = [], []
    for pixel, ok, near in zip(pixels, above, near_source):
        if not ok:
            reason = "below_threshold"
        elif near:
            reason = "existing_source"
        elif excluded is not None and excluded[pixel]:
            reason = "exclusion_region"
        elif accepted and _angle(vectors[[pixel]], vectors[accepted]).min() < (
            min_separation
        ):
            reason = "too_close"
        else:
            reason = "accepted"
            accepted.append(pixel)
        reasons.append(reason)
    return pixels, np.array(reasons, dtype=str)


def cosine_taper_weights(distance, core_radius, outer_radius):
    """One within `core_radius`, a cosine taper to zero at `outer_radius`."""
    if not 0.0 <= core_radius < outer_radius:
        raise ValueError("need 0 <= core_radius < outer_radius")
    phase = np.clip((distance - core_radius) / (outer_radius - core_radius), 0, 1)
    return 0.5 * (1.0 + np.cos(np.pi * phase))


def relaxation_weights(nside, pixels, core_radius, outer_radius):
    """Largest taper weight over the candidate `pixels`, per HEALPix pixel."""
    vectors, _ = _healpix(nside)
    weights = np.zeros(vectors.shape[0])
    # One candidate at a time: a pixels x candidates table does not fit at nside 512.
    for vector in vectors[np.asarray(pixels, dtype=int)]:
        distance = _angle(vectors, vector[None])[:, 0]
        weights = np.maximum(
            weights, cosine_taper_weights(distance, core_radius, outer_radius)
        )
    return weights


def relax_map(values, weights, rng, *, scale):
    """`(1 - w) * values + w * clip(scale * N(0, 1), -scale, scale)`.

    Noise is drawn only where `w > 0`, so values outside the taper support
    are kept exactly. `weights` broadcasts against the last axes of `values`.
    """
    values = np.asarray(values, dtype=float)
    weights = np.broadcast_to(weights, values.shape)
    support = weights > 0.0
    noise = np.zeros_like(values)
    noise[support] = np.clip(
        scale * rng.normal(size=np.count_nonzero(support)), -scale, scale
    )
    return (1.0 - weights) * values + weights * noise


def relax_position(position, transforms, weights, rng, *, scale):
    """Relax the excitation maps of all `transforms` in `position`.

    `transforms` are `ExcitationTransform`s (e.g. the values of
    `build_excitation_transforms`), every field that enters the score.
    """
    for transform in transforms:
        excitation = transform.to_excitation_map(position)
        relaxed = relax_map(excitation, weights, rng, scale=scale)
        position = transform.with_excitation_map(position, relaxed)
    return position


def aperture_labels(nside, pixels, outer_radius):
    """Index of the nearest candidate within `outer_radius` per pixel, else -1.

    Exactly equidistant pixels go to either candidate (floating-point rounding).
    """
    vectors, _ = _healpix(nside)
    labels = np.full(vectors.shape[0], -1)
    nearest = np.full(vectors.shape[0], float(outer_radius))
    # One candidate at a time, as in `relaxation_weights`; ties keep the first.
    for i, vector in enumerate(vectors[np.asarray(pixels, dtype=int)]):
        distance = _angle(vectors, vector[None])[:, 0]
        closer = distance < nearest
        labels[closer], nearest[closer] = i, distance[closer]
    return labels


def removed_emission(
    diffuse, positions, transforms, weights, labels, pixel_area, rng, *, scale, n_draws
):
    """Diffuse emission removed by relaxing each position `n_draws` times.

    Parameters
    ----------
    diffuse: callable
        Latent position to the diffuse sky, shape `(n_e, npix)`, per solid
        angle (e.g. a jitted `SpectralProductSky`).
    positions: iterable
        Posterior samples, e.g. a `jft.Samples`.
    transforms, weights, scale:
        As in `relax_position`.
    labels: np.ndarray
        Aperture assignment from `aperture_labels`.
    pixel_area: float
        Pixel solid angle.

    Returns
    -------
    fluxes: np.ndarray
        Aperture integrals `F_e`, shape `(n_samples, n_draws, n_candidates, n_e)`.
    mean, std: np.ndarray
        Removed emission over all (sample, draw) pairs, shape `(n_e, npix)`.
    """
    inside = np.flatnonzero(labels >= 0)
    apertures = csr_array(
        (np.full(inside.size, pixel_area), (inside, labels[inside])),
        shape=(labels.size, labels.max() + 1),
    )
    fluxes, total, total_sq, n = [], 0.0, 0.0, 0
    for position in positions:
        sky = np.asarray(diffuse(position))
        draws = []
        for _ in range(n_draws):
            relaxed = relax_position(position, transforms, weights, rng, scale=scale)
            removed = sky - np.asarray(diffuse(relaxed))
            draws.append((removed @ apertures).T)
            total, total_sq, n = total + removed, total_sq + removed**2, n + 1
        fluxes.append(draws)
    mean = total / n
    return np.array(fluxes), mean, np.sqrt(np.maximum(total_sq / n - mean**2, 0.0))


def pivot_flux(fluxes, relative_log_energies, spectral_index):
    """Least-squares pivot flux `sum_e F_e t_e / sum_e t_e**2` on the last axis,
    with the mean source spectrum `t_e = exp(spectral_index * relative_log_energies)`.
    """
    t = np.exp(spectral_index * np.asarray(relative_log_energies))
    return np.asarray(fluxes) @ t / (t @ t)


def flux_acceptance(flux, persistence, max_candidates=None):
    """Accept candidates by their pivot fluxes, `(n_samples, n_draws, n_candidates)`,
    with the candidates in score order.

    A candidate is rejected as `"non_positive_flux"` if the mean over all
    (sample, draw) pairs is not finite and positive, as `"low_persistence"`
    if fewer than a `persistence` fraction of samples have a positive flux
    averaged over their draws, and as `"round_limit"` if `max_candidates`
    earlier candidates are already accepted.

    Returns
    -------
    mean, positive_fraction, reasons: np.ndarray
        Per candidate.
    """
    mean = flux.mean(axis=(0, 1))
    positive_fraction = np.mean(flux.mean(axis=1) > 0.0, axis=0)
    reasons = np.where(
        ~(np.isfinite(mean) & (mean > 0.0)),
        "non_positive_flux",
        np.where(positive_fraction < persistence, "low_persistence", "accepted"),
    )
    if max_candidates is not None:
        accepted = np.flatnonzero(reasons == "accepted")
        reasons[accepted[max_candidates:]] = "round_limit"
    return mean, positive_fraction, reasons


def select_and_accept(
    score, measure, *, persistence, max_candidates=None, excluded=None, **selection
):
    """`select_candidates` and `flux_acceptance` for one round, re-selecting
    around flux-rejected candidates.

    A flux-rejected candidate (e.g. a hole, which the unsigned score ranks like
    a bump) can hide a lower real source as `"too_close"`. Its pixel is then
    excluded and the selection rerun on the same score, until it finds no new
    candidate or the new ones could only hit the round limit. Each extra pass
    costs one `measure` call over the current candidates, which keeps the
    apertures consistent; usually there is none.

    Parameters
    ----------
    measure: callable
        Candidate pixels in score order to their pivot fluxes
        `(n_samples, n_draws, n_candidates)`, e.g. `relaxation_weights`,
        `aperture_labels`, `removed_emission` and `pivot_flux` for them.
    persistence, max_candidates:
        As in `flux_acceptance`.
    excluded, **selection:
        As in `select_candidates`.

    Returns
    -------
    pixels, mean, positive_fraction, reasons: np.ndarray
        Every measured candidate in score order with its `flux_acceptance`
        result; candidates rejected in an earlier pass keep that result.
    """
    score = np.asarray(score)
    excluded = np.zeros(score.size, bool) if excluded is None else np.array(excluded)
    pixels, reasons = select_candidates(score, excluded=excluded, **selection)
    pixels = pixels[reasons == "accepted"]
    if pixels.size == 0:  # skip `measure`: relaxing nothing still costs a full pass
        return pixels, np.zeros(0), np.zeros(0), np.array([], dtype=str)
    rejected = []
    while True:
        mean, fraction, reasons = flux_acceptance(
            measure(pixels), persistence, max_candidates
        )
        bad = np.isin(reasons, ("non_positive_flux", "low_persistence"))
        excluded[pixels[bad]] = True
        found, why = select_candidates(score, excluded=excluded, **selection)
        found = found[why == "accepted"]
        new = np.setdiff1d(found, pixels)
        if new.size == 0 or (
            max_candidates is not None
            and np.sum(score[pixels[~bad]] > score[new].max()) >= max_candidates
        ):
            break
        rejected.append((pixels[bad], mean[bad], fraction[bad], reasons[bad]))
        pixels = found
    parts = rejected + [(pixels, mean, fraction, reasons)]
    pixels, mean, fraction, reasons = (np.concatenate(p) for p in zip(*parts))
    order = np.lexsort((pixels, -score[pixels]))
    return pixels[order], mean[order], fraction[order], reasons[order]
