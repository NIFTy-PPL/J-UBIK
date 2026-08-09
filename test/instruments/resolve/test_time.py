import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import jubik as ju
from jubik.instruments.resolve.data.antenna_positions import AntennaPositions
from jubik.instruments.resolve.data.data_modify.precision import to_single_precision
from jubik.instruments.resolve.data.data_modify.time import (
    move_time,
    restrict_by_time,
    time_average,
    time_average_to_length_of_timebins,
)
from jubik.instruments.resolve.data.observation import Observation

from generate_test_obs import generate_random_obs

pmp = pytest.mark.parametrize

FREQS = np.array([1.0e9, 1.1e9])
POL_TYPE = ju.polarization.PolarizationType(("LL", "RR"))


def build_obs(times, ant1, ant2, freqs=FREQS, **kwargs):
    """Random observation with fully specified calibration information."""
    times = np.asarray(times, dtype=np.float64)
    ant1 = np.asarray(ant1, dtype=np.int64)
    ant2 = np.asarray(ant2, dtype=np.int64)
    return generate_random_obs(
        freqs,
        times.size,
        [-1e2, 1e2],
        [-5, 5],
        POL_TYPE,
        ant1=ant1,
        ant2=ant2,
        times=times,
        **kwargs,
    )


def manual_obs(times, ant1, ant2, vis, weight, uvw=None, freqs=FREQS):
    """Observation built from explicit visibility/weight arrays."""
    times = np.asarray(times, dtype=np.float64)
    ant1 = np.asarray(ant1, dtype=np.int64)
    ant2 = np.asarray(ant2, dtype=np.int64)
    vis = np.array(vis, dtype=np.complex128)
    weight = np.array(weight, dtype=np.float64)
    if uvw is None:
        uvw = np.arange(3 * times.size, dtype=np.float64).reshape(times.size, 3)
    antpos = AntennaPositions(np.array(uvw, dtype=np.float64), ant1, ant2, times)
    return Observation(
        antpos,
        vis,
        weight,
        POL_TYPE.get_legacy_polarization(),
        np.asarray(freqs, dtype=np.float64),
        None,
    )


def group_rows(obs, bins):
    """Reproduce the (ant1, ant2, bin) grouping of `time_average`.

    Returns the group keys in output order, the row -> bin map and, for every
    group, the indices of the input rows that belong to it. The output order of
    `time_average` is `np.lexsort` over the columns (ant1, ant2, bin), i.e. the
    bin is the primary and ant1 the least significant key.
    """
    row_to_bin = np.full(obs.nrow, -1, dtype=int)
    for ii, (lo, hi) in enumerate(bins):
        row_to_bin[np.logical_and(obs.time >= lo, obs.time < hi)] = ii
    assert np.all(row_to_bin >= 0)
    keys = sorted(
        set(zip(obs.ant1.tolist(), obs.ant2.tolist(), row_to_bin.tolist())),
        key=lambda kk: (kk[2], kk[1], kk[0]),
    )
    rows = [
        np.where((obs.ant1 == a1) & (obs.ant2 == a2) & (row_to_bin == bb))[0]
        for a1, a2, bb in keys
    ]
    return keys, row_to_bin, rows


# Times, antennas and time bins of the "rich" test case: baselines are
# interleaved (non-contiguous in row order), one group holds a single row.
RICH_TIMES = [0.0, 0.0, 1.0, 1.0, 2.0, 10.0, 11.0, 11.0]
RICH_ANT1 = [0, 0, 0, 1, 0, 0, 0, 1]
RICH_ANT2 = [1, 2, 1, 2, 1, 1, 1, 2]
RICH_BINS = [[0.0, 5.0], [10.0, 15.0]]


def build_rich_obs():
    return build_obs(RICH_TIMES, RICH_ANT1, RICH_ANT2)


# ---------------------------------------------------------------------------
# time_average: core averaging math
# ---------------------------------------------------------------------------


def test_time_average_weighted_mean_matches_hand_computed_value():
    np.random.seed(301)
    n_rows = 4
    shape = (2, n_rows, 2)
    vis = np.random.normal(size=shape) + 1j * np.random.normal(size=shape)
    weight = np.random.uniform(1.0, 10.0, size=shape)
    # One cell with hand-chosen numbers:
    #   bin 0 (rows 0, 1): (1*1 + 3*3) / (1 + 3) = 2.5  (real and imaginary)
    #   bin 1 (rows 2, 3): (10*2 + 0*2) / (2 + 2) = 5.0 (real and imaginary)
    vis[0, :, 0] = [1 + 1j, 3 + 3j, 10 + 0j, 0 + 10j]
    weight[0, :, 0] = [1.0, 3.0, 2.0, 2.0]
    obs = manual_obs(
        [0.0, 1.0, 10.0, 11.0], [0, 0, 0, 0], [1, 1, 1, 1], vis, weight
    )

    new = time_average(obs, [[0.0, 5.0], [10.0, 15.0]])

    assert new.nrow == 2
    assert_allclose(new.vis_val[0, :, 0], [2.5 + 2.5j, 5.0 + 5.0j])
    assert_allclose(new.weight_val[0, :, 0], [4.0, 4.0])
    # Every other cell follows the same inverse-variance weighted mean.
    for grp, rows in enumerate(([0, 1], [2, 3])):
        wgt = weight[:, rows, :]
        expected = np.sum(vis[:, rows, :] * wgt, axis=1) / np.sum(wgt, axis=1)
        assert_allclose(new.vis_val[:, grp, :], expected, rtol=1e-14)


def test_time_average_weight_is_the_sum_of_the_input_weights_per_group():
    np.random.seed(302)
    obs = build_rich_obs()
    keys, _, rows = group_rows(obs, RICH_BINS)

    new = time_average(obs, RICH_BINS)

    expected = np.stack(
        [np.sum(obs.weight_val[:, ind, :], axis=1) for ind in rows], axis=1
    )
    assert new.weight_val.shape == (obs.npol, len(keys), obs.nfreq)
    assert_allclose(new.weight_val, expected, rtol=1e-14)


def test_time_average_vis_is_the_weighted_mean_per_group():
    np.random.seed(303)
    obs = build_rich_obs()
    _, _, rows = group_rows(obs, RICH_BINS)

    new = time_average(obs, RICH_BINS)

    expected = np.stack(
        [
            np.sum(obs.vis_val[:, ind, :] * obs.weight_val[:, ind, :], axis=1)
            / np.sum(obs.weight_val[:, ind, :], axis=1)
            for ind in rows
        ],
        axis=1,
    )
    assert_allclose(new.vis_val, expected, rtol=1e-13)


def test_time_average_nrow_equals_number_of_unique_baseline_bin_combinations():
    np.random.seed(304)
    obs = build_rich_obs()
    keys, _, _ = group_rows(obs, RICH_BINS)
    # Two baselines in the first bin appear in non-contiguous rows, the groups
    # (0, 2, bin 0) and (1, 2, bin 0) hold a single row each.
    assert len(keys) == 5

    new = time_average(obs, RICH_BINS)

    assert new.nrow == 5
    assert_array_equal(new.ant1, [kk[0] for kk in keys])
    assert_array_equal(new.ant2, [kk[1] for kk in keys])
    assert new.vis_val.shape == (obs.npol, 5, obs.nfreq)


def test_time_average_uvw_is_the_plain_mean_of_the_group():
    np.random.seed(305)
    obs = build_rich_obs()
    _, _, rows = group_rows(obs, RICH_BINS)

    new = time_average(obs, RICH_BINS)

    expected = np.stack([np.mean(obs.uvw[ind], axis=0) for ind in rows])
    assert new.uvw.shape == (5, 3)
    assert_allclose(new.uvw, expected, rtol=1e-14)
    # It really is the unweighted mean, not the weight-weighted one.
    single_row_group = rows[1]
    assert single_row_group.size == 1
    assert_allclose(new.uvw[1], obs.uvw[single_row_group[0]])


def test_time_average_new_times_are_bin_means_and_sorted():
    np.random.seed(306)
    obs = build_rich_obs()
    keys, row_to_bin, _ = group_rows(obs, RICH_BINS)

    new = time_average(obs, RICH_BINS)

    # Plain (unweighted) mean over all rows of a bin, broadcast to the groups.
    bin_means = np.array(
        [np.mean(obs.time[row_to_bin == bb]) for bb in range(len(RICH_BINS))]
    )
    assert_allclose(bin_means, [4.0 / 5.0, 32.0 / 3.0])
    expected = bin_means[[kk[2] for kk in keys]]
    assert_allclose(new.time, expected, rtol=1e-14)
    # The lexsort uses the bin as primary key, hence the times come out sorted.
    assert np.all(np.diff(new.time) >= 0)


def test_time_average_timestamp_of_a_group_is_the_bin_mean_not_the_group_mean():
    """Locks in the current convention: all groups of a bin share one time."""
    np.random.seed(307)
    obs = build_obs(
        [0.0, 1.0, 2.0, 8.0, 9.0], [0, 0, 0, 2, 2], [1, 1, 1, 3, 3]
    )

    new = time_average(obs, [[0.0, 10.0]])

    assert new.nrow == 2
    assert_array_equal(new.ant1, [0, 2])
    # Mean over all five rows, even though group (2, 3) only covers [8, 9].
    assert_allclose(new.time, [4.0, 4.0])


def test_time_average_does_not_mutate_the_input_observation():
    np.random.seed(308)
    obs = build_rich_obs()
    vis, weight = obs.vis_val.copy(), obs.weight_val.copy()
    times, uvw = obs.time.copy(), obs.uvw.copy()

    new = time_average(obs, RICH_BINS)

    assert new is not obs
    assert isinstance(new, Observation)
    assert_array_equal(obs.vis_val, vis)
    assert_array_equal(obs.weight_val, weight)
    assert_array_equal(obs.time, times)
    assert_array_equal(obs.uvw, uvw)
    assert obs.nrow == len(RICH_TIMES)


def test_time_average_output_dtypes():
    np.random.seed(309)
    obs = build_rich_obs()

    new = time_average(obs, RICH_BINS)

    assert np.issubdtype(new.ant1.dtype, np.integer)
    assert np.issubdtype(new.ant2.dtype, np.integer)
    assert new.time.dtype == np.float64
    assert new.uvw.dtype == np.float64
    assert new.vis_val.dtype == np.complex128
    assert new.weight_val.dtype == np.float64


def test_time_average_preserves_single_precision():
    np.random.seed(310)
    obs = to_single_precision(build_rich_obs())

    new = time_average(obs, RICH_BINS)

    assert new.vis_val.dtype == np.complex64
    assert new.weight_val.dtype == np.float32
    assert new.is_single_precision()


# ---------------------------------------------------------------------------
# time_average: flagged data
# ---------------------------------------------------------------------------


def test_time_average_raises_on_a_bin_with_total_weight_zero():
    np.random.seed(311)
    shape = (2, 2, 2)
    vis = np.random.normal(size=shape) + 1j * np.random.normal(size=shape)
    weight = np.random.uniform(1.0, 10.0, size=shape)
    weight[0, :, 0] = 0.0  # every row of one (pol, group, freq) cell flagged
    obs = manual_obs([0.0, 1.0], [0, 0], [1, 1], vis, weight)

    with pytest.raises(ValueError, match="total weight 0"):
        time_average(obs, [[0.0, 5.0]])


def test_time_average_raises_when_one_baseline_is_completely_flagged():
    """A single fully flagged baseline aborts the whole call."""
    np.random.seed(312)
    shape = (2, 3, 2)
    vis = np.random.normal(size=shape) + 1j * np.random.normal(size=shape)
    weight = np.random.uniform(1.0, 10.0, size=shape)
    weight[:, 2, :] = 0.0  # the only row of baseline (0, 2)
    obs = manual_obs([0.0, 1.0, 2.0], [0, 0, 0], [1, 1, 2], vis, weight)

    with pytest.raises(ValueError, match="total weight 0"):
        time_average(obs, [[0.0, 5.0]])


def test_time_average_ignores_nan_visibilities_of_flagged_rows():
    np.random.seed(313)
    shape = (2, 2, 2)
    vis = np.random.normal(size=shape) + 1j * np.random.normal(size=shape)
    weight = np.random.uniform(1.0, 10.0, size=shape)
    weight[0, 0, 0] = 0.0
    vis[0, 0, 0] = np.nan  # legal: only vis[weight > 0] must be finite
    obs = manual_obs([0.0, 1.0], [0, 0], [1, 1], vis, weight)

    with np.errstate(all="raise"):
        new = time_average(obs, [[0.0, 5.0]])

    assert new.nrow == 1
    # Only the second row carries weight, so it survives unchanged.
    assert_allclose(new.vis_val[0, 0, 0], vis[0, 1, 0])
    assert_allclose(new.weight_val[0, 0, 0], weight[0, 1, 0])
    assert np.all(np.isfinite(new.vis_val))


def test_time_average_after_flags_to_nan_matches_the_unflagged_average():
    np.random.seed(314)
    shape = (2, 4, 2)
    vis = np.random.normal(size=shape) + 1j * np.random.normal(size=shape)
    weight = np.random.uniform(1.0, 10.0, size=shape)
    # Rows 0 and 2 are fully flagged, so every group keeps one good row.
    weight[:, 0, :] = 0.0
    weight[:, 2, :] = 0.0
    obs = manual_obs([0.0, 1.0, 10.0, 11.0], [0] * 4, [1] * 4, vis, weight)
    nan_obs = obs.flags_to_nan()
    assert np.any(np.isnan(nan_obs.vis_val))

    reference = time_average(obs, RICH_BINS)
    new = time_average(nan_obs, RICH_BINS)

    assert_allclose(new.vis_val[:, 0, :], vis[:, 1, :], rtol=1e-14)
    assert_allclose(new.vis_val[:, 1, :], vis[:, 3, :], rtol=1e-14)

    assert_allclose(new.vis_val, reference.vis_val, rtol=1e-13)
    assert_allclose(new.weight_val, reference.weight_val, rtol=1e-14)


def test_time_average_flagged_rows_do_not_enter_the_weighted_mean():
    np.random.seed(315)
    shape = (2, 3, 2)
    vis = np.random.normal(size=shape) + 1j * np.random.normal(size=shape)
    weight = np.random.uniform(1.0, 10.0, size=shape)
    weight[0, 1, 0] = 0.0
    obs = manual_obs([0.0, 1.0, 2.0], [0, 0, 0], [1, 1, 1], vis, weight)

    new = time_average(obs, [[0.0, 5.0]])

    keep = [0, 2]
    expected = np.sum(vis[0, keep, 0] * weight[0, keep, 0]) / np.sum(
        weight[0, keep, 0]
    )
    assert_allclose(new.vis_val[0, 0, 0], expected, rtol=1e-14)


# ---------------------------------------------------------------------------
# time_average: input validation
# ---------------------------------------------------------------------------


def test_time_average_rows_outside_every_bin_raise_a_descriptive_error():
    np.random.seed(316)
    obs = build_obs([0.0, 1.0, 100.0], [0, 0, 0], [1, 1, 1])

    with pytest.raises(ValueError, match="not covered"):
        time_average(obs, [[0.0, 5.0]])


def test_time_average_overlapping_bins_raise_a_descriptive_error():
    np.random.seed(317)
    obs = build_obs([0.0, 1.0, 2.0], [0, 0, 0], [1, 1, 1])

    with pytest.raises(ValueError, match="overlap"):
        time_average(obs, [[0.0, 5.0], [1.0, 3.0]])


def test_time_average_unsorted_times_across_bins_raise_a_descriptive_error():
    np.random.seed(318)
    obs = build_obs([0.0, 10.0, 1.0], [0, 0, 0], [1, 1, 1])

    with pytest.raises(ValueError, match="sorted|increas"):
        time_average(obs, [[0.0, 5.0], [10.0, 15.0]])


def test_time_average_accepts_antenna_indices_beyond_one_thousand():
    np.random.seed(319)
    obs = build_obs([0.0, 1.0], [0, 0], [1500, 1500])

    new = time_average(obs, [[0.0, 5.0]])

    assert new.nrow == 1
    assert_array_equal(new.ant1, [0])
    assert_array_equal(new.ant2, [1500])
    expected = np.sum(obs.vis_val * obs.weight_val, axis=1) / np.sum(
        obs.weight_val, axis=1
    )
    assert_allclose(new.vis_val[:, 0, :], expected, rtol=1e-14)


def test_time_average_skips_time_bins_without_any_row():
    np.random.seed(320)
    obs = build_obs([0.0, 20.0], [0, 0], [1, 1])
    bins = [[0.0, 5.0], [5.0, 10.0], [15.0, 25.0]]

    with np.errstate(all="raise"):
        new = time_average(obs, bins)

    assert new.nrow == 2
    assert_allclose(new.time, [0.0, 20.0])
    assert_allclose(new.vis_val[:, 0, :], obs.vis_val[:, 0, :], rtol=1e-14)
    assert_allclose(new.vis_val[:, 1, :], obs.vis_val[:, 1, :], rtol=1e-14)


# ---------------------------------------------------------------------------
# time_average_to_length_of_timebins
# ---------------------------------------------------------------------------


def test_time_average_to_length_of_timebins_none_is_noop():
    np.random.seed(321)
    obs = build_rich_obs()

    assert time_average_to_length_of_timebins(obs, None) is obs


def test_time_average_to_length_of_timebins_argument_is_a_bin_length():
    """`len_tbin` is the length of a bin, not the number of bins."""
    np.random.seed(322)
    times = np.arange(101, dtype=np.float64)
    obs = build_obs(times, np.zeros(101, dtype=int), np.ones(101, dtype=int))

    new = time_average_to_length_of_timebins(obs, 10)

    assert new.nrow == 11
    assert_allclose(new.time, list(np.arange(4.5, 100.0, 10.0)) + [100.0])


@pmp("n_times,len_tbin", ((11, 5), (8, 3), (10, 4), (7, 7), (6, 1)))
def test_time_average_to_length_of_timebins_covers_the_last_time_point(
    n_times, len_tbin
):
    np.random.seed(323 + n_times)
    times = np.arange(n_times, dtype=np.float64)
    obs = build_obs(times, np.zeros(n_times, dtype=int), np.ones(n_times, dtype=int))

    new = time_average_to_length_of_timebins(obs, len_tbin)

    # No row may get lost: the weights are conserved by the averaging.
    assert_allclose(
        np.sum(new.weight_val), np.sum(obs.weight_val), rtol=1e-13
    )
    expected_bins = int(np.ceil(n_times / len_tbin))
    assert new.nrow == expected_bins
    assert_allclose(new.time[-1], np.mean(times[(expected_bins - 1) * len_tbin :]))


def test_time_average_to_length_of_timebins_puts_edge_times_into_the_upper_bin():
    np.random.seed(324)
    times = np.array([0.0, 5.0, 10.0])
    obs = build_obs(times, [0, 0, 0], [1, 1, 1])

    new = time_average_to_length_of_timebins(obs, 5)

    # Bins are [0, 5), [5, 10), [10, 15): every time sits on an edge and lands
    # in the bin that starts there, so nothing is averaged together.
    assert new.nrow == 3
    assert_allclose(new.time, times)
    assert_allclose(new.vis_val, obs.vis_val, rtol=1e-14)


def test_time_average_to_length_of_timebins_with_a_large_time_offset():
    np.random.seed(325)
    t0 = 4.8e9  # seconds since MJD0, as in a real measurement set
    times = t0 + np.arange(6, dtype=np.float64)
    obs = build_obs(times, np.zeros(6, dtype=int), np.ones(6, dtype=int))

    new = time_average_to_length_of_timebins(obs, 2)

    assert new.nrow == 3
    assert_allclose(new.time, t0 + np.array([0.5, 2.5, 4.5]))


def test_time_average_to_length_of_timebins_below_the_time_resolution():
    """A bin shorter than the ulp of the timestamps cannot resolve anything."""
    np.random.seed(326)
    t0 = 4.8e9
    times = t0 + np.array([0.0, 1e-7, 2e-7])
    assert np.unique(times).size == 1  # below the float64 resolution at t0
    obs = build_obs(times, np.zeros(3, dtype=int), np.ones(3, dtype=int))

    with pytest.raises(ValueError):
        time_average_to_length_of_timebins(obs, 1e-7)


# ---------------------------------------------------------------------------
# restrict_by_time
# ---------------------------------------------------------------------------


def test_restrict_by_time_is_half_open():
    np.random.seed(327)
    obs = build_obs([0.0, 1.0, 2.0, 3.0], [0, 0, 0, 0], [1, 1, 1, 1])

    new = restrict_by_time(obs, 1.0, 3.0)

    # tmin is inclusive, tmax is exclusive.
    assert new.nrow == 2
    assert_array_equal(new.time, [1.0, 2.0])
    assert_array_equal(new.vis_val, obs.vis_val[:, 1:3])
    assert_array_equal(new.weight_val, obs.weight_val[:, 1:3])
    assert_array_equal(new.uvw, obs.uvw[1:3])


def test_restrict_by_time_with_index_returns_the_matching_slice():
    np.random.seed(328)
    obs = build_obs([0.0, 1.0, 2.0, 3.0], [0, 0, 0, 0], [1, 1, 1, 1])

    new, ind = restrict_by_time(obs, 1.0, 3.0, True)

    assert ind == slice(1, 3)
    assert_array_equal(obs.time[ind], new.time)
    assert_array_equal(obs.vis_val[:, ind], new.vis_val)


@pmp(
    "tmin,tmax,expected",
    (
        (0.0, 4.0, [0.0, 1.0, 2.0, 3.0]),
        (0.0, 0.0, []),
        (1.5, 2.5, [2.0]),
        (4.0, 5.0, []),
        (-1.0, 1.0, [0.0]),
    ),
)
def test_restrict_by_time_boundaries(tmin, tmax, expected):
    np.random.seed(329)
    obs = build_obs([0.0, 1.0, 2.0, 3.0], [0, 0, 0, 0], [1, 1, 1, 1])

    new = restrict_by_time(obs, tmin, tmax)

    assert_array_equal(new.time, expected)


def test_restrict_by_time_repeated_times_are_kept_together():
    np.random.seed(330)
    obs = build_obs([0.0, 1.0, 1.0, 2.0], [0, 0, 0, 0], [1, 1, 1, 1])

    new, ind = restrict_by_time(obs, 1.0, 2.0, True)

    assert ind == slice(1, 3)
    assert_array_equal(new.time, [1.0, 1.0])


def test_restrict_by_time_requires_sorted_times():
    np.random.seed(331)
    obs = build_obs([0.0, 2.0, 1.0], [0, 0, 0], [1, 1, 1])

    with pytest.raises(AssertionError, match="increase"):
        restrict_by_time(obs, 0.0, 3.0)


# ---------------------------------------------------------------------------
# move_time
# ---------------------------------------------------------------------------


@pmp("t0", (0.0, 5.0, -3.5, 1e9))
def test_move_time_shifts_the_times_and_keeps_the_data(t0):
    np.random.seed(332)
    obs = build_rich_obs()
    times = obs.time.copy()

    new = move_time(obs, t0)

    assert new is not obs
    assert_allclose(new.time, times + t0)
    assert_array_equal(new.vis_val, obs.vis_val)
    assert_array_equal(new.weight_val, obs.weight_val)
    assert_array_equal(new.uvw, obs.uvw)
    assert_array_equal(new.ant1, obs.ant1)
    assert_array_equal(new.ant2, obs.ant2)
    assert_array_equal(new.freq, obs.freq)
    # The input is left untouched.
    assert_array_equal(obs.time, times)


def test_move_time_is_additive():
    np.random.seed(333)
    obs = build_rich_obs()

    new = move_time(move_time(obs, 2.0), 3.0)

    assert_allclose(new.time, obs.time + 5.0)
