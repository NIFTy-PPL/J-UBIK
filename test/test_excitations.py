# Copyright(C) 2026
# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause
# Authors: Matteo Guardiani

"""Normalized excitation maps of `SpectralProductSky` and their inverses."""

import jax
import jax.random as random
import nifty.re as jft
import numpy as np
import pytest
from nifty.re.correlated_field import make_grid
from numpy.testing import assert_allclose

import jubik as ju
from jubik.excitations import ExcitationTransform, build_excitation_transforms

pmp = pytest.mark.parametrize

AMPLITUDE = dict(
    fluctuations=(0.5, 0.01),
    loglogavgslope=(-3.0, 0.1),
    flexibility=None,
    asperity=None,
)


def _sky(harmonic_type, shape, log_frequencies=(0.1, 0.4, 0.8), deviations=True):
    return ju.build_simple_spectral_sky(
        prefix="sky",
        shape=shape,
        distances=(0.1,) * len(shape),
        log_frequencies=np.array(log_frequencies),
        reference_frequency_index=1 if len(log_frequencies) > 1 else 0,
        zero_mode_settings=(0.0, 1.0),
        spatial_amplitude_settings=AMPLITUDE,
        spectral_index_settings=dict(mean=(0.0, 1.0), fluctuations=(0.5, 0.01)),
        deviations_settings=(
            dict(process="wiener", sigma=(1.0, 0.1)) if deviations else None
        ),
        harmonic_type=harmonic_type,
    )


def test_fields_and_keys_follow_the_sky():
    transforms = build_excitation_transforms(_sky("spherical", (2,)))
    assert {k: t.key for k, t in transforms.items()} == {
        "spatial": "sky_spatial_xi",
        "spectral_index": "sky_spectral_index_xi",
        "deviations": "sky_spectral_deviations_xi",
    }
    assert set(
        build_excitation_transforms(_sky("spherical", (2,), deviations=False))
    ) == {
        "spatial",
        "spectral_index",
    }
    single = _sky("fourier", (4, 4), log_frequencies=(0.1,), deviations=False)
    assert set(build_excitation_transforms(single)) == {"spatial"}


@pmp("harmonic_type, shape", [("fourier", (16, 12)), ("spherical", (128,))])
def test_coefficient_round_trip(harmonic_type, shape):
    sky = _sky(harmonic_type, shape)
    p = jft.Vector(sky.init(random.PRNGKey(3)))
    transforms = build_excitation_transforms(sky)
    assert transforms["deviations"].to_excitation_map(p).shape[0] == 2
    for t in transforms.values():
        back = t.with_excitation_map(p, t.to_excitation_map(p))
        assert isinstance(back, type(p))
        assert_allclose(back[t.key], p[t.key], rtol=1e-10, atol=1e-10)
        # Other latents are untouched.
        assert all(np.array_equal(back[k], p[k]) for k in p.tree if k != t.key)


@pmp("harmonic_type, shape", [("fourier", (8, 8)), ("spherical", (8,))])
def test_zero_mode_is_kept(harmonic_type, shape):
    sky = _sky(harmonic_type, shape)
    p = sky.init(random.PRNGKey(5))
    transforms = build_excitation_transforms(sky)
    for t in transforms.values():
        # A constant offset is pure l=0: nothing changes.
        back = t.with_excitation_map(p, t.to_excitation_map(p) + 1.0)
        assert_allclose(back[t.key], p[t.key], rtol=1e-10, atol=1e-10)

    # A local edit moves the deviations, but not their l=0 coefficient.
    t = transforms["deviations"]
    edited = np.array(t.to_excitation_map(p))
    edited[:, 0] += 3.0
    back = t.with_excitation_map(p, edited)
    zero = (slice(None),) + (0,) * len(shape)
    assert_allclose(back[t.key][zero], p[t.key][zero], rtol=0, atol=0)
    assert not np.allclose(back[t.key], p[t.key])


@pmp("harmonic_type, shape", [("fourier", (8, 6)), ("spherical", (4,))])
@pmp("zero_mode_active", [False, True])
def test_unit_per_pixel_prior_variance(harmonic_type, shape, zero_mode_active):
    grid = make_grid(shape, (0.1,) * len(shape), harmonic_type)
    t = ExcitationTransform("xi", grid, zero_mode_active=zero_mode_active)
    hshape = grid.harmonic_grid.shape
    to_excitation_map = lambda xi: t.to_excitation_map({"xi": xi}).reshape(-1)

    # Exact: the prior covariance of the map is J J^T for white xi.
    jac = jax.jacfwd(to_excitation_map)(np.zeros(hshape)).reshape(-1, np.prod(hshape))
    assert_allclose(np.sum(jac**2, axis=1), 1.0, rtol=1e-10)

    # And on prior samples.
    xi = np.random.default_rng(0).normal(size=(4000,) + hshape)
    maps = np.asarray(jax.vmap(to_excitation_map)(xi))
    assert_allclose(maps.var(axis=0).mean(), 1.0, rtol=0.02)
    assert_allclose(maps.mean(axis=0), 0.0, atol=0.1)
