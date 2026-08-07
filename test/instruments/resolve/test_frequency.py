import numpy as np
import pytest
from numpy.testing import assert_array_equal

import jubik as ju
from jubik.instruments.resolve.data.data_modify.frequency import (
    exclude_frequency_ranges,
)
from jubik.instruments.resolve.parse.data.data_modify.frequency import (
    SpectralModify,
)

from generate_test_obs import generate_random_obs

pmp = pytest.mark.parametrize

FREQS = np.array([1.0e9, 1.1e9, 1.2e9, 1.3e9, 1.4e9, 1.5e9])


def build_obs(freqs=FREQS):
    pol_type = ju.polarization.PolarizationType(("LL", "RR"))
    return generate_random_obs(freqs, 20, [-1e2, 1e2], [-5, 5], pol_type)


def test_exclude_range_drops_channels_and_keeps_rest_bitwise():
    np.random.seed(42)
    obs = build_obs()
    # Drops the channels at 1.2e9 and 1.3e9.
    new = exclude_frequency_ranges(obs, [[1.15e9, 1.35e9]])

    keep = np.array([0, 1, 4, 5])
    assert new.nfreq == len(keep)
    assert_array_equal(new.freq, obs.freq[keep])
    assert_array_equal(new.vis.asnumpy(), obs.vis.asnumpy()[:, :, keep])
    assert_array_equal(new.weight.asnumpy(), obs.weight.asnumpy()[:, :, keep])


@pmp("ranges", (None, [], ()))
def test_exclude_no_ranges_is_noop(ranges):
    np.random.seed(43)
    obs = build_obs()
    new = exclude_frequency_ranges(obs, ranges)

    assert new is obs
    assert new.nfreq == obs.nfreq


def test_exclude_all_channels_raises():
    np.random.seed(44)
    obs = build_obs()
    with pytest.raises(ValueError):
        exclude_frequency_ranges(obs, [[0.5e9, 2.0e9]])


def test_exclude_non_overlapping_range_returns_obs_unchanged():
    np.random.seed(45)
    obs = build_obs()
    new = exclude_frequency_ranges(obs, [[2.0e9, 3.0e9]])

    assert new is obs
    assert new.nfreq == obs.nfreq


def test_exclude_bounds_are_inclusive():
    np.random.seed(46)
    obs = build_obs()
    # The range bounds coincide with channel centers, both get dropped.
    new = exclude_frequency_ranges(obs, [[1.1e9, 1.3e9]])

    keep = np.array([0, 4, 5])
    assert_array_equal(new.freq, obs.freq[keep])


def test_exclude_multiple_disjoint_ranges():
    np.random.seed(47)
    obs = build_obs()
    new = exclude_frequency_ranges(obs, [[0.9e9, 1.05e9], [1.35e9, 1.45e9]])

    keep = np.array([1, 2, 3, 5])
    assert_array_equal(new.freq, obs.freq[keep])
    assert_array_equal(new.vis.asnumpy(), obs.vis.asnumpy()[:, :, keep])
    assert_array_equal(new.weight.asnumpy(), obs.weight.asnumpy()[:, :, keep])


def test_spectral_modify_parses_exclude_frequency_ranges():
    spectral = {"exclude_frequency_ranges": [[1.1e9, 1.2e9], [1.4e9, 1.5e9]]}
    modify = SpectralModify.from_yaml_dict(spectral)

    assert modify.exclude_ranges == [[1.1e9, 1.2e9], [1.4e9, 1.5e9]]


def test_spectral_modify_exclude_ranges_default_is_none():
    modify = SpectralModify.from_yaml_dict({})

    assert modify.exclude_ranges is None


@pmp("entry", ([1.2e9, 1.1e9], [1.2e9, 1.2e9], [1.1e9, 1.2e9, 1.3e9]))
def test_spectral_modify_invalid_exclude_range_raises(entry):
    with pytest.raises(ValueError):
        SpectralModify.from_yaml_dict({"exclude_frequency_ranges": [entry]})
