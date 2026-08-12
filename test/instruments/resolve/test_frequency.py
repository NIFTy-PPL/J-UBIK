import numpy as np
import pytest
from astropy import units as u
from numpy.testing import assert_allclose, assert_array_equal

import jubik as ju
from jubik.instruments.resolve.data.data_modify.frequency import (
    exclude_frequency_ranges,
    freq_average_by_bins,
    freq_average_by_fdom_and_n_freq_chunks,
)
from jubik.instruments.resolve.data.observation import Observation
from jubik.instruments.resolve.parse.data.data_modify.frequency import (
    SpectralModify,
)

from generate_test_obs import generate_random_obs

pmp = pytest.mark.parametrize

FREQS = np.array([1.0e9, 1.1e9, 1.2e9, 1.3e9, 1.4e9, 1.5e9])


def build_obs(freqs=FREQS):
    pol_type = ju.polarization.PolarizationType(("LL", "RR"))
    return generate_random_obs(freqs, 20, [-1e2, 1e2], [-5, 5], pol_type)


def assert_frequency_groups_are_averaged(obs, averaged, groups):
    expected_freq = np.array([np.mean(obs.freq[ind]) for ind in groups])
    expected_vis = np.stack(
        [np.mean(obs.vis_val[..., ind], axis=2) for ind in groups], axis=2
    )
    expected_weight = np.stack(
        [
            len(ind) ** 2 / np.sum(1 / obs.weight_val[..., ind], axis=2)
            for ind in groups
        ],
        axis=2,
    )

    assert_allclose(averaged.freq, expected_freq)
    assert_allclose(averaged.vis_val, expected_vis)
    assert_allclose(averaged.weight_val, expected_weight)


@pmp("n_freq,n_bins", ((4, 2), (5, 2), (6, 2), (4, 1), (4, 4)))
def test_freq_average_by_bins_uses_every_channel(n_freq, n_bins):
    np.random.seed(40 + n_freq + n_bins)
    freqs = 1.0e9 + np.arange(n_freq) * 0.1e9
    obs = build_obs(freqs)

    averaged = freq_average_by_bins(obs, n_bins)
    groups = np.array_split(np.arange(n_freq), n_bins)

    assert_frequency_groups_are_averaged(obs, averaged, groups)


def test_freq_average_by_bins_none_is_noop():
    obs = build_obs()
    assert freq_average_by_bins(obs, None) is obs


def test_freq_average_by_bins_excludes_flagged_channels():
    obs = build_obs()
    vis = obs.vis_val.copy()
    weight = obs.weight_val.copy()

    # A flagged visibility may contain an arbitrary placeholder and must not
    # affect either the average or its propagated weight.
    vis[0, 0, 1] = 1.0e30
    weight[0, 0, 1] = 0.0
    weight[1, 0, :] = 0.0
    flagged_obs = Observation(
        obs.antenna_positions,
        vis,
        weight,
        obs.legacy_polarization,
        obs.freq,
        obs._auxiliary_tables,
    )

    averaged = freq_average_by_bins(flagged_obs, 1)
    valid = weight[0, 0] > 0.0
    expected_vis = np.mean(vis[0, 0, valid])
    expected_weight = valid.sum() ** 2 / np.sum(1.0 / weight[0, 0, valid])

    assert_allclose(averaged.vis_val[0, 0, 0], expected_vis)
    assert_allclose(averaged.weight_val[0, 0, 0], expected_weight)
    assert averaged.vis_val[1, 0, 0] == 0.0
    assert averaged.weight_val[1, 0, 0] == 0.0


@pmp("n_bins", (0, -1, len(FREQS) + 1))
def test_freq_average_by_bins_rejects_invalid_number_of_bins(n_bins):
    obs = build_obs()
    with pytest.raises(ValueError):
        freq_average_by_bins(obs, n_bins)


def test_frequency_domain_averaging_uses_every_channel_in_each_chunk():
    np.random.seed(48)
    freqs = 1.0e9 + np.arange(6) * 0.1e9
    obs = build_obs(freqs)
    sky_frequencies = ju.Color([0.95e9, 1.55e9] * u.Hz)

    averaged = freq_average_by_fdom_and_n_freq_chunks(sky_frequencies, obs, 2)
    groups = np.array_split(np.arange(obs.nfreq), 2)

    assert_frequency_groups_are_averaged(obs, averaged, groups)


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
