# Copyright(C) 2026
# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause
# Authors: Matteo Guardiani

"""Candidate selection, relaxation and removed-flux handoff on HEALPix."""

import numpy as np
import pytest
from ducc0.healpix import Healpix_Base
from nifty.re.correlated_field import make_grid
from numpy.testing import assert_allclose, assert_array_equal
from scipy.stats import chi

import jubik as ju
from jubik import source_detection as sd
from jubik.excitations import ExcitationTransform

NSIDE = 16
BASE = Healpix_Base(NSIDE, "RING")
VECTORS = BASE.pix2vec(np.arange(BASE.npix()))


def _pixel(lon_deg, lat_deg):
    theta, phi = np.radians(90.0 - lat_deg), np.radians(lon_deg)
    return int(BASE.ang2pix(np.array([[theta, phi]]))[0])


def _distance(pixel):
    return np.arccos(np.clip(VECTORS @ VECTORS[pixel], -1.0, 1.0))


def test_multifrequency_score():
    maps = np.array([[3.0, 0.0], [4.0, -1.0], [0.0, 0.0]])
    assert_allclose(sd.multifrequency_score(maps, p=0.5), [5.0, 1.0] / chi.ppf(0.5, 3))


def test_local_maxima_break_ties_by_pixel_index():
    score = np.zeros(BASE.npix())
    peak = _pixel(40.0, 10.0)
    twin = int(BASE.neighbors(np.array([peak]))[0, 0])
    score[[peak, twin]] = 2.0
    other = _pixel(200.0, -60.0)
    score[other] = 1.0
    maxima = sd.local_maxima(score)
    assert min(peak, twin) in maxima and max(peak, twin) not in maxima
    assert other in maxima
    # The flat background keeps only the pixels below all their neighbours.
    assert set(maxima) - {min(peak, twin), other} == {0}


def test_select_candidates_reasons():
    deg = np.radians
    score = np.zeros(BASE.npix())
    tie = sorted([_pixel(120.0, 50.0), _pixel(300.0, 50.0)])
    spikes = {
        "a": (_pixel(0.0, 0.0), 6.0),
        "close_to_a": (_pixel(7.0, 0.0), 5.0),
        "near_source": (_pixel(90.0, 30.0), 4.5),
        "excluded": (_pixel(180.0, 0.0), 4.0),
        "b": (_pixel(270.0, -30.0), 3.0),
        "tie_low": (tie[0], 2.0),
        "tie_high": (tie[1], 2.0),
        "faint": (_pixel(45.0, -45.0), 0.5),
    }
    for pixel, value in spikes.values():
        score[pixel] = value
    excluded = _distance(spikes["excluded"][0]) < deg(5.0)

    pixels, reasons = sd.select_candidates(
        score,
        threshold=1.0,
        min_separation=deg(10.0),
        source_lonlat=[(deg(92.0), deg(31.0))],
        exclusion_radius=deg(5.0),
        excluded=excluded,
    )
    got = dict(zip(pixels.tolist(), reasons.tolist()))
    assert {name: got[pixel] for name, (pixel, _) in spikes.items()} == {
        "a": "accepted",
        "close_to_a": "too_close",
        "near_source": "existing_source",
        "excluded": "exclusion_region",
        "b": "accepted",
        "tie_low": "accepted",
        "tie_high": "accepted",
        "faint": "below_threshold",
    }
    # Descending score, ties by pixel index; the flat background comes last.
    assert_array_equal(pixels[:8], [pixel for pixel, _ in spikes.values()])
    assert set(reasons[8:]) == {"below_threshold"}

    with pytest.raises(ValueError, match="threshold"):
        sd.select_candidates(score, threshold=-1.0, min_separation=0.0)


def test_select_candidates_measures_distances_above_threshold_only(monkeypatch):
    deg = np.radians
    rng = np.random.default_rng(3)
    score = rng.uniform(0.0, 1.0, BASE.npix())  # many maxima below threshold
    bright, faint_near_source = _pixel(0.0, 0.0), _pixel(90.0, 30.0)
    score[bright], score[faint_near_source] = 5.0, 1.5
    sources = np.stack(
        [rng.uniform(-np.pi, np.pi, 1000), np.arcsin(rng.uniform(-1, 1, 1000))], -1
    )
    sources[0] = deg(90.0), deg(30.0)

    rows = []
    angle = sd._angle
    monkeypatch.setattr(sd, "_angle", lambda a, b: rows.append(len(a)) or angle(a, b))
    pixels, reasons = sd.select_candidates(
        score,
        threshold=2.0,
        min_separation=deg(10.0),
        source_lonlat=sources,
        exclusion_radius=deg(1.0),
    )
    assert max(rows) == 1  # only the bright maximum, never all maxima
    got = dict(zip(pixels.tolist(), reasons.tolist()))
    assert got[bright] == "accepted"
    # Below the threshold comes first, also next to an existing source.
    assert got[faint_near_source] == "below_threshold"
    assert set(reasons) == {"accepted", "below_threshold"}


def test_relaxation_weights_and_blend():
    deg = np.radians
    assert_allclose(
        sd.cosine_taper_weights(
            deg(np.array([0.0, 1.0, 1.5, 2.0, 5.0])), deg(1.0), deg(2.0)
        ),
        [1.0, 1.0, 0.5, 0.0, 0.0],
        atol=1e-12,
    )
    a, b = _pixel(0.0, 0.0), _pixel(3.0, 0.0)
    w = sd.relaxation_weights(NSIDE, [a, b], deg(1.0), deg(6.0))
    single = [sd.relaxation_weights(NSIDE, [c], deg(1.0), deg(6.0)) for c in (a, b)]
    assert_allclose(w, np.maximum(*single))
    assert_array_equal(sd.relaxation_weights(NSIDE, [], deg(1.0), deg(2.0)), 0.0)

    values = np.random.default_rng(0).normal(size=(2, BASE.npix()))
    relaxed = sd.relax_map(values, w, np.random.default_rng(1), scale=0.3)
    outside, core = w == 0.0, w == 1.0
    assert_array_equal(relaxed[:, outside], values[:, outside])
    assert np.all(np.abs(relaxed[:, core]) <= 0.3)
    assert_array_equal(
        relaxed, sd.relax_map(values, w, np.random.default_rng(1), scale=0.3)
    )

    with pytest.raises(ValueError, match="core_radius"):
        sd.cosine_taper_weights(0.0, 2.0, 1.0)


def test_aperture_labels_assign_overlaps_to_the_nearest():
    deg = np.radians
    a, b = _pixel(0.0, 0.0), _pixel(6.0, 0.0)
    labels = sd.aperture_labels(NSIDE, [a, b], deg(5.0))
    da, db = _distance(a), _distance(b)
    assert_array_equal(labels[(da < deg(5.0)) & (da < db)], 0)
    assert_array_equal(labels[(db < deg(5.0)) & (db < da)], 1)
    assert_array_equal(labels[(da >= deg(5.0)) & (db >= deg(5.0))], -1)
    assert_array_equal(sd.aperture_labels(NSIDE, [], deg(5.0)), -1)


class _Identity:
    """Excitation transform whose map is the latent itself."""

    def to_excitation_map(self, position):
        return position["xi"]

    def with_excitation_map(self, position, map):
        return {**position, "xi": map}


def test_removed_emission_and_pivot_flux():
    deg = np.radians
    spectrum = np.exp(-2.0 * np.array([-0.5, 0.0, 0.5]))
    diffuse = lambda p: spectrum[:, None] * np.exp(p["xi"])
    a, b = _pixel(0.0, 0.0), _pixel(4.0, 0.0)
    weights = sd.relaxation_weights(NSIDE, [a, b], deg(1.0), deg(3.0))
    labels = sd.aperture_labels(NSIDE, [a, b], deg(3.0))
    area = 4 * np.pi / BASE.npix()
    positions = [
        {"xi": s * 2.0 * np.exp(-0.5 * (_distance(a) / deg(2.0)) ** 2)}
        for s in (1.0, 0.5)
    ]

    # scale=0 relaxes to zero exactly, so the removed emission is known.
    fluxes, mean, std = sd.removed_emission(
        diffuse,
        positions,
        [_Identity()],
        weights,
        labels,
        area,
        np.random.default_rng(0),
        scale=0.0,
        n_draws=2,
    )
    assert fluxes.shape == (2, 2, 2, 3)
    removed = [diffuse(p) - diffuse({"xi": (1 - weights) * p["xi"]}) for p in positions]
    for s, r in enumerate(removed):
        for c in (0, 1):
            assert_allclose(
                fluxes[s, :, c],
                np.broadcast_to(r[:, labels == c].sum(1) * area, (2, 3)),
            )
    assert_allclose(mean, np.mean(removed, axis=0), atol=1e-14)
    assert_allclose(std, np.std(removed, axis=0), atol=1e-7)

    # The removed emission has the source spectrum, so the fit returns its pivot.
    f = sd.pivot_flux(fluxes, [-0.5, 0.0, 0.5], -2.0)
    assert_allclose(f, fluxes[..., 1], rtol=1e-12)


def test_flux_acceptance():
    flux = np.array(
        [
            # (n_samples=4, n_draws=2, n_candidates=4)
            [[1.0, -1.0, 5.0, np.nan], [1.0, -1.0, 5.0, 1.0]],
            [[1.0, -1.0, -1.0, 1.0], [1.0, -1.0, -1.0, 1.0]],
            [[1.0, 1.0, -1.0, 1.0], [-0.5, 1.0, -1.0, 1.0]],
            [[0.2, -0.5, -1.0, 1.0], [-0.4, -0.5, -1.0, 1.0]],
        ]
    )
    mean, fraction, reasons = sd.flux_acceptance(flux, persistence=0.5)
    assert_allclose(fraction[:3], [0.75, 0.25, 0.25])
    assert_array_equal(
        reasons,
        ["accepted", "non_positive_flux", "low_persistence", "non_positive_flux"],
    )
    assert mean[2] > 0

    # The limit counts accepted candidates only, in the given order.
    _, _, reasons = sd.flux_acceptance(flux[..., [1, 0, 0, 0]], 0.5, max_candidates=2)
    assert_array_equal(
        reasons, ["non_positive_flux", "accepted", "accepted", "round_limit"]
    )


def test_hole_does_not_use_up_the_round_limit():
    """The unsigned score ranks a hole above a real source; with a limit of
    one the round still adds the source."""
    deg = np.radians
    hole, source = _pixel(0.0, 0.0), _pixel(90.0, 0.0)
    xi = 2.0 * np.exp(-0.5 * (_distance(source) / deg(4.0)) ** 2) - 3.0 * np.exp(
        -0.5 * (_distance(hole) / deg(4.0)) ** 2
    )
    pixels, reasons = sd.select_candidates(
        np.abs(xi), threshold=1.0, min_separation=deg(10.0)
    )
    selected = pixels[reasons == "accepted"]
    assert_array_equal(selected, [hole, source])

    weights = sd.relaxation_weights(NSIDE, selected, deg(4.0), deg(8.0))
    labels = sd.aperture_labels(NSIDE, selected, deg(8.0))
    fluxes, _, _ = sd.removed_emission(
        lambda p: np.exp(p["xi"])[None],
        [{"xi": xi}],
        [_Identity()],
        weights,
        labels,
        4 * np.pi / BASE.npix(),
        np.random.default_rng(0),
        scale=0.3,
        n_draws=3,
    )
    f = sd.pivot_flux(fluxes, [0.0], 0.0)
    _, _, reasons = sd.flux_acceptance(f, 0.5, max_candidates=1)
    assert_array_equal(reasons, ["non_positive_flux", "accepted"])


def test_injected_excess_is_removed_at_one_degree_core():
    """The plan's aperture check at nside=128 (lmax=256): a Gaussian excess of
    sigma 0.5 deg in a normalized spatial xi map, relaxed with core 1 deg and
    outer radius 2 deg, loses at least 90% of its log-intensity there."""
    nside, deg = 128, np.radians
    grid = make_grid((nside,), None, "spherical")
    t = ExcitationTransform("xi", grid, zero_mode_active=False)
    pd = np.asarray(grid.harmonic_grid.power_distributor)
    amplitude = np.where(pd > 0, np.maximum(pd, 1) ** -1.4, 0.0)
    sht = ju.get_healpix_sht(nside, 2 * nside, 2 * nside)
    log_intensity = lambda p: np.asarray(sht(amplitude * p["xi"]))

    base = Healpix_Base(nside, "RING")
    centre = int(base.ang2pix(np.array([[1.1, 0.7]]))[0])
    vectors = base.pix2vec(np.arange(base.npix()))
    distance = np.arccos(np.clip(vectors @ vectors[centre], -1.0, 1.0))
    background = {"xi": np.random.default_rng(0).normal(size=pd.size)}
    excess = 5.0 * np.exp(-0.5 * (distance / deg(0.5)) ** 2)
    excitation = np.asarray(t.to_excitation_map(background))
    injected = t.with_excitation_map(background, excitation + excess)

    weights = sd.relaxation_weights(nside, [centre], deg(1.0), deg(2.0))

    def removed(p):
        # Same seed for both: the background and noise terms cancel exactly.
        relaxed = sd.relax_position(
            p, [t], weights, np.random.default_rng(1), scale=0.3
        )
        return log_intensity(p) - log_intensity(relaxed)

    inside = distance < deg(2.0)
    injected_excess = log_intensity(injected) - log_intensity(background)
    fraction = (removed(injected) - removed(background))[inside].sum() / (
        injected_excess[inside].sum()
    )
    assert 0.9 <= fraction <= 1.05
