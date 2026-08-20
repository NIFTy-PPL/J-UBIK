#!/usr/bin/env python3

# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause
# Authors: Hanieh Zandinejad

import jax.random as random
import pytest
import numpy as np

import jubik as ju

from jubik.sky_model.multifrequency.spectral_product_mf_point_sources import (
    _point_source_spatial_shape,
)

pmp = pytest.mark.parametrize


@pmp("shape", [(10,), (10, 10), (4, 5, 6)])
def test_point_source_spatial_shape_cartesian(shape):
    assert _point_source_spatial_shape(shape, "cartesian") == tuple(shape)


@pmp("nside", [4, 8, 16])
def test_point_source_spatial_shape_spherical(nside):
    npix = _point_source_spatial_shape((nside,), "spherical")
    assert npix == (12 * nside**2,)


@pmp("shape", [(4, 4), (4, 4, 4), ()])
def test_point_source_spatial_shape_spherical_wrong_shape(shape):
    with pytest.raises(ValueError):
        _point_source_spatial_shape(shape, "spherical")


def test_point_source_spatial_shape_unsupported_harmonic_type():
    with pytest.raises(ValueError):
        _point_source_spatial_shape((8,), "unsupported")


@pmp("seed", [12, 42])
@pmp("deviations_settings", [None, dict(process="wiener", sigma=(1.0, 0.1))])
def test_healpix_invgamma_sky_forward(seed, deviations_settings):
    nside = 8
    shape = (nside,)
    log_frequencies = np.array((0.1, 0.2, 0.6))
    reference_frequency_index = 0

    spectral_settings = dict(
        mean=(0.0, 1.0),
        deviations=deviations_settings,
        shared=False,
    )

    model = ju.build_mf_invgamma_sky(
        prefix="healpix_invgamma_test",
        alpha=2.0,
        q=0.08,
        shape=shape,
        log_frequencies=log_frequencies,
        reference_frequency_index=reference_frequency_index,
        spectral_settings=spectral_settings,
        harmonic_type="spherical",
    )

    rp = model.init(random.PRNGKey(seed))
    sky = model(rp)

    assert sky.shape == (log_frequencies.size, 12 * nside**2)
    assert np.all(np.isfinite(np.asarray(sky)))


def test_healpix_invgamma_sky_shared_spectral_index():
    nside = 4
    shape = (nside,)
    log_frequencies = np.array((0.1, 0.2, 0.6))

    spectral_settings = dict(
        mean=(0.0, 1.0),
        deviations=None,
        shared=True,
    )

    model = ju.build_mf_invgamma_sky(
        prefix="healpix_invgamma_shared_test",
        alpha=2.0,
        q=0.08,
        shape=shape,
        log_frequencies=log_frequencies,
        reference_frequency_index=0,
        spectral_settings=spectral_settings,
        harmonic_type="spherical",
    )

    rp = model.init(random.PRNGKey(0))
    sky = model(rp)

    assert sky.shape == (log_frequencies.size, 12 * nside**2)
    assert np.all(np.isfinite(np.asarray(sky)))
    assert model.spectral_index_distribution(rp).shape == ()
