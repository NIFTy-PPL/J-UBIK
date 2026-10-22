# Copyright(C) 2026
# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause
# Authors: Matteo Guardiani

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax
from scipy.stats import invgamma

import jubik as ju
from jubik.sky_model.movable_point_sources import (
    deposit_bilinear_cartesian,
    gnomonic_to_lonlat,
    healpix_bilinear_weights,
)

pmp = pytest.mark.parametrize

LOG_FREQUENCIES = np.log([0.5, 1.0, 2.0, 4.0])
REF = 1
SPECTRAL = dict(mean=(-2.4, 0.2), deviations=dict(process="wiener", sigma=(0.3, 0.1)))


def _build(anchors, geometry="cartesian", flux_prior=("lognormal", 1.0, 0.5), **kw):
    kw = dict(
        shape=(16, 12) if geometry == "cartesian" else (8,),
        distances=(0.1, 0.1) if geometry == "cartesian" else None,
        log_frequencies=LOG_FREQUENCIES,
        reference_frequency_index=REF,
        spectral_settings=SPECTRAL,
    ) | kw
    return ju.build_movable_point_sources(
        "src", anchors, 0.05, flux_prior, geometry, **kw
    )


# anomaly_detection `src/point_sources.py` (b237383), the renderer being
# ported. Sky axes (freq, y, x), locations (x, y) in pixel units.
def _reference_square_sky(background_sky, locations, fluxes):
    def insert(sky, args):
        location, flux = args
        pixel_index = jnp.floor(location).astype(int)
        delta_x, delta_y = jnp.modf(location)[0] - 0.5
        x_overlap = jnp.array(
            [jnp.minimum(0.0, delta_x), 1.0 - jnp.abs(delta_x), jnp.maximum(0.0, delta_x)]
        )
        y_overlap = jnp.array(
            [jnp.minimum(0.0, delta_y), 1.0 - jnp.abs(delta_y), jnp.maximum(0.0, delta_y)]
        )
        source = flux[:, None, None] * jnp.outer(jnp.abs(y_overlap), jnp.abs(x_overlap))
        start = (0, pixel_index[1], pixel_index[0])
        updated = lax.dynamic_slice(sky, start, source.shape) + source
        return lax.dynamic_update_slice(sky, updated, start), None

    sky = jnp.pad(background_sky, ((0, 0), (1, 1), (1, 1)))
    sky, _ = lax.scan(insert, sky, (locations, fluxes))
    return sky[:, 1:-1, 1:-1]


def test_cartesian_matches_anomaly_detection_renderer():
    rng = np.random.default_rng(1)
    shape = (16, 12)
    xy = rng.uniform(0.0, 1.0, (50, 2)) * np.array(shape[::-1])
    fluxes = rng.uniform(1.0, 30.0, (50, 3))
    expected = _reference_square_sky(jnp.zeros((3,) + shape), xy, fluxes)
    got = deposit_bilinear_cartesian(xy[:, ::-1], fluxes.T, shape)
    np.testing.assert_allclose(got, expected, atol=1e-12)


@pmp("geometry", ["cartesian", "spherical"])
def test_empty_catalogue(geometry):
    model = _build(np.zeros((0, 2)), geometry)
    p = model.init(jax.random.PRNGKey(0))
    sky = model(p)
    npix = (16, 12) if geometry == "cartesian" else (12 * 8**2,)
    assert sky.shape == (LOG_FREQUENCIES.size,) + npix
    assert np.all(sky == 0)
    assert model.positions(p).shape == (0, 2)


def test_overlapping_sources_add():
    shape = (8, 8)
    xy = np.array([[3.3, 4.6], [3.3, 4.6], [3.9, 4.1]])
    spectra = np.array([[1.0, 2.0, 5.0]])
    both = deposit_bilinear_cartesian(xy, spectra, shape)
    single = sum(
        deposit_bilinear_cartesian(xy[i : i + 1], spectra[:, i : i + 1], shape)
        for i in range(3)
    )
    np.testing.assert_allclose(both, single, atol=1e-14)
    np.testing.assert_allclose(both.sum(), 8.0)


@pmp("geometry", ["cartesian", "spherical"])
def test_flux_conservation_and_units(geometry):
    anchors = np.array([[0.7, 0.5], [0.42, 0.9], [0.41, 0.91]])
    if geometry == "spherical":
        anchors = np.array([[0.3, 1.5], [-3.1, -0.2], [2.0, -1.55]])
    model = _build(anchors, geometry)
    p = model.init(jax.random.PRNGKey(3))
    omega = 0.1 * 0.1 if geometry == "cartesian" else 4 * np.pi / (12 * 8**2)
    sky = model(p)
    assert np.all(sky >= 0)
    total = sky.reshape(sky.shape[0], -1).sum(axis=1) * omega
    np.testing.assert_allclose(total, model.spectra(p).sum(axis=1), rtol=1e-12)


def test_off_grid_flux_is_dropped_not_clamped():
    shape = (6, 5)
    spectra = np.ones((1, 1))
    # The ported renderer clamped: a source far off the grid landed on the edge
    clamped = _reference_square_sky(np.zeros((1,) + shape), np.array([[2.0, -3.0]]), spectra)
    assert clamped.sum() == 1
    # Far outside on every side: nothing lands on the edge pixels
    for xy in ([-3.0, 2.0], [9.0, 2.0], [2.0, -3.0], [2.0, 7.5]):
        assert np.all(deposit_bilinear_cartesian(np.array([xy]), spectra, shape) == 0)
    # Within the outer half of an edge pixel: the part beyond its centre is lost
    sky = deposit_bilinear_cartesian(np.array([[0.2, 2.5]]), spectra, shape)
    np.testing.assert_allclose(sky.sum(), 0.7)
    np.testing.assert_allclose(sky[0, 0, 2], 0.7)
    # Up to the outer pixel centres the flux is conserved
    sky = deposit_bilinear_cartesian(np.array([[0.5, 4.5]]), spectra, shape)
    np.testing.assert_allclose(sky[0, 0, 4], 1.0)


@pmp("geometry", ["cartesian", "spherical"])
def test_spectra_anchored_at_pivot(geometry):
    model = _build(np.array([[0.5, 0.5], [1.0, 0.2]]), geometry)
    for seed in range(3):
        p = model.init(jax.random.PRNGKey(seed))
        np.testing.assert_allclose(model.spectra(p)[REF], model.fluxes(p), rtol=1e-14)
        assert model.spectra(p).shape == (LOG_FREQUENCIES.size, 2)
    # Section 3 freezes the spectral latents by these domains, not by key names
    spectral = model.spectral_model
    assert set(spectral.log_spectral_behavior.domain) == {"src_spectral_index_mean"}
    assert set(spectral.spectral_index_deviations.domain) == {
        "src_spectral_deviations_sigma", "src_spectral_deviations_xi"
    }


def test_single_frequency_has_no_spectral_latents():
    model = _build(np.array([[0.5, 0.5]]), log_frequencies=[0.0], reference_frequency_index=0)
    assert set(model.domain) == {"src_flux", "src_position_offsets"}
    p = model.init(jax.random.PRNGKey(0))
    assert model(p).shape == (1, 16, 12)


@pmp("flux_prior, median", [
    (("lognormal", 2.0, 1.0), 2.0 / np.sqrt(1 + 0.25)),
    (("invgamma", 3.0, 0.5), invgamma.ppf(0.5, 3.0, scale=0.5)),
])
def test_flux_prior_inverse_and_median(flux_prior, median):
    model = _build(np.array([[0.5, 0.5], [1.0, 0.2]]), flux_prior=flux_prior)
    p = model.init(jax.random.PRNGKey(4))
    np.testing.assert_allclose(model.flux_invprior(model.fluxes(p)), p["src_flux"], atol=1e-6)
    p0 = jax.tree.map(jnp.zeros_like, p)
    np.testing.assert_allclose(model.fluxes(p0), median, rtol=1e-6)


@pmp("geometry", ["cartesian", "spherical"])
def test_appending_sources_keeps_existing_ones(geometry):
    anchors = np.array([[0.5, 0.5], [1.0, 0.2]])
    old, new = _build(anchors, geometry), _build(np.vstack([anchors, [[0.8, 0.3]]]), geometry)
    p = old.init(jax.random.PRNGKey(5))
    # Zero-pad every latent along the axis that grew (the source axis)
    p_new = {
        k: jnp.pad(v, np.reshape([(0, n - o) for o, n in zip(v.shape, s)], (-1, 2)))
        for k, v in p.items()
        for s in [new.domain[k].shape]
    }
    for f in ("positions", "fluxes"):
        np.testing.assert_array_equal(getattr(new, f)(p_new)[:2], getattr(old, f)(p))
    np.testing.assert_array_equal(new.spectra(p_new)[:, :2], old.spectra(p))
    np.testing.assert_allclose(new.positions(p_new)[2], [0.8, 0.3])


@pmp("geometry", ["cartesian", "spherical"])
def test_finite_jvp_vjp(geometry):
    model = _build(np.array([[0.73, 0.52], [0.31, 0.88]]), geometry)
    p = model.init(jax.random.PRNGKey(6))
    t = model.init(jax.random.PRNGKey(7))
    _, tangent = jax.jvp(model, (p,), (t,))
    sky, vjp = jax.vjp(model, p)
    (cotangent,) = vjp(jnp.ones_like(sky))
    assert np.all(np.isfinite(tangent))
    assert all(np.all(np.isfinite(v)) for v in jax.tree.leaves(cotangent))


def test_gnomonic_projection():
    anchors = np.array([[0.0, 0.0], [np.pi, 0.3], [1.0, -1.2], [2.0, np.pi / 2]])
    offsets = np.array([[0.01, 0.0], [0.02, 0.01], [-0.03, 0.04], [0.0, -0.05]])
    at_anchor = gnomonic_to_lonlat(anchors, 0 * offsets)
    np.testing.assert_allclose(at_anchor[:3], anchors[:3], atol=1e-15)
    np.testing.assert_allclose(at_anchor[3, 1], np.pi / 2)  # pole: lon is arbitrary
    lonlat = gnomonic_to_lonlat(anchors, offsets)
    np.testing.assert_allclose(lonlat[0], [np.arctan(0.01), 0.0], atol=1e-15)
    assert -np.pi < lonlat[1, 0] < -3.0  # east of lon = pi wraps
    # The angular distance from the anchor is arctan of the tangent-plane radius
    def unit(lonlat):
        lon, lat = lonlat.T
        return np.stack(
            [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], -1
        )
    distance = np.arccos(np.clip(np.sum(unit(anchors) * unit(np.asarray(lonlat)), -1), -1, 1))
    np.testing.assert_allclose(distance, np.arctan(np.hypot(*offsets.T)), rtol=1e-7)


def test_position_summary_across_longitude_wrap():
    model = _build(np.array([[np.pi, 0.0]]), "spherical")
    samples = [{"src_position_offsets": jnp.array([[s, 0.0]])} for s in (-0.02, 0.0, 0.02)]
    mean, std = model.position_summary(samples)
    np.testing.assert_allclose(np.abs(mean[0, 0]), np.pi)
    # the latents are scaled by position_std = 0.05
    np.testing.assert_allclose(std, [[0.05 * np.std([-0.02, 0.0, 0.02]), 0.0]])


@pmp("nside", [1, 2, 4, 16])
def test_healpix_weights_match_healpy(nside):
    hp = pytest.importorskip("healpy")
    # Random directions plus the cases that need care: the poles and polar
    # caps, every ring and points just off it, the equator-cap transition
    # z = +-2/3, and longitudes at and around the wrap.
    eps = 1e-9
    rng = np.random.default_rng(42)
    ring_theta = np.arccos(hp.ringinfo(nside, np.arange(1, 4 * nside))[2])
    theta = np.concatenate([
        np.arccos(rng.uniform(-1, 1, 200)),
        [0.0, np.pi, eps, np.pi - eps],
        rng.uniform(0, ring_theta[0], 10),
        rng.uniform(ring_theta[-1], np.pi, 10),
        ring_theta, ring_theta + eps, ring_theta - eps,
        np.arccos([2 / 3, -2 / 3]) + eps, np.arccos([2 / 3, -2 / 3]) - eps,
    ])
    phi = rng.uniform(-np.pi, 2 * np.pi, theta.size)
    phi[:12] = [0.0, -eps, 2 * np.pi - eps, np.pi, -np.pi + eps, eps] * 2

    pix, wgt = healpix_bilinear_weights(nside, jnp.asarray(theta), jnp.asarray(phi))
    assert np.all(wgt >= 0)
    np.testing.assert_allclose(wgt.sum(-1), 1.0, rtol=1e-14)

    # Compare as maps: where a point sits on a ring, a zero-weight pixel may differ
    def dense(pix, wgt):
        m = np.zeros((theta.size, 12 * nside**2))
        np.add.at(m, (np.arange(theta.size)[:, None], np.asarray(pix)), np.asarray(wgt))
        return m

    ref_pix, ref_wgt = hp.get_interp_weights(nside, theta, phi)
    expected = dense(ref_pix.T, ref_wgt.T)
    np.testing.assert_allclose(dense(pix, wgt), expected, atol=1e-13)


def test_healpix_weights_wrap_poles_and_transition():
    nside = 4
    # Longitude wrap
    theta = jnp.array([0.3, 1.2, 2.9])
    for shift in (2 * np.pi, -2 * np.pi):
        a = healpix_bilinear_weights(nside, theta, jnp.full(3, 0.1))
        b = healpix_bilinear_weights(nside, theta, jnp.full(3, 0.1 + shift))
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_allclose(a[1], b[1], atol=1e-14)
    # At the poles the four polar pixels share the flux equally
    npix = 12 * nside**2
    pix, wgt = healpix_bilinear_weights(nside, jnp.array([0.0, np.pi]), jnp.array([1.0, 1.0]))
    assert set(np.asarray(pix[0])) == {0, 1, 2, 3}
    assert set(np.asarray(pix[1])) == {npix - 4, npix - 3, npix - 2, npix - 1}
    np.testing.assert_allclose(wgt, 0.25)
    # Weights are continuous across rings and across z = 2/3
    for t in [np.arccos(2 / 3), 0.5, 2.0]:
        maps = []
        for eps in (-1e-10, 1e-10):
            p, w = healpix_bilinear_weights(nside, jnp.array([t + eps]), jnp.array([0.4]))
            maps.append(np.bincount(np.asarray(p[0]), np.asarray(w[0]), npix))
        np.testing.assert_allclose(maps[0], maps[1], atol=1e-8)


def test_spherical_position_gradient_away_from_knots():
    model = _build(np.array([[0.37, 0.61], [-2.2, -0.4]]), "spherical")
    p = model.init(jax.random.PRNGKey(8))
    weights = jax.random.uniform(jax.random.PRNGKey(9), model(p).shape)

    def loss(offsets):
        return jnp.sum(weights * model(p | {"src_position_offsets": offsets}))

    x = p["src_position_offsets"]
    grad = jax.grad(loss)(x)
    assert np.all(np.isfinite(grad)) and np.any(grad != 0)
    step = 1e-7 * jax.random.normal(jax.random.PRNGKey(10), x.shape)
    np.testing.assert_allclose(
        loss(x + step) - loss(x - step), 2 * jnp.vdot(grad, step), rtol=1e-5
    )
