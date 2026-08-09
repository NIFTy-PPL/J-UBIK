import numpy as np
import pytest
from numpy.testing import assert_array_equal

import jubik as ju
from jubik.instruments.resolve.data.data_modify.visibility_subset import (
    select_random_visibility_subset,
)
from jubik.instruments.resolve.parse.data.data_modify.visibility_subset import (
    SelectSubset,
)

from generate_test_obs import generate_random_obs

pmp = pytest.mark.parametrize

FREQS = np.array([1.0e9, 1.1e9, 1.2e9])
POL = ju.polarization.PolarizationType(("LL", "RR"))

# Mask that `_generate_mask` produces for length=10 and percentage=0.5, i.e.
# np.sort(Generator(PCG64(42)).choice(np.arange(10), size=5, replace=False)).
GENERATED_MASK_10_50 = np.array([0, 3, 4, 5, 7])


def build_obs(n_rows=10, calib_info=False):
    """Random observation, optionally with antenna/time (calibration) info."""
    if calib_info:
        ant1 = np.arange(n_rows, dtype=np.int64) % 3
        ant2 = (np.arange(n_rows, dtype=np.int64) + 1) % 3
        times = np.arange(n_rows, dtype=np.float64) * 10.0
    else:
        ant1 = ant2 = times = None
    return generate_random_obs(
        FREQS,
        n_rows,
        [-1e2, 1e2],
        [-5, 5],
        POL,
        ant1=ant1,
        ant2=ant2,
        times=times,
    )


def assert_is_row_subset(obs, new, mask):
    """`new` contains exactly the rows `mask` of `obs`."""
    assert new.nrow == len(mask)
    assert_array_equal(new.vis.asnumpy(), obs.vis.asnumpy()[:, mask, :])
    assert_array_equal(new.weight.asnumpy(), obs.weight.asnumpy()[:, mask, :])
    assert_array_equal(new.antenna_positions.uvw, obs.antenna_positions.uvw[mask])
    assert_array_equal(new.freq, obs.freq)


def test_select_subset_none_is_noop():
    np.random.seed(70)
    obs = build_obs()

    assert select_random_visibility_subset(obs, None) is obs


def test_select_subset_is_deterministic():
    np.random.seed(71)
    obs = build_obs()

    first = select_random_visibility_subset(obs, SelectSubset(percentage=0.5))
    second = select_random_visibility_subset(obs, SelectSubset(percentage=0.5))

    assert_array_equal(first.vis.asnumpy(), second.vis.asnumpy())
    assert_array_equal(first.weight.asnumpy(), second.weight.asnumpy())
    assert_array_equal(
        first.antenna_positions.uvw, second.antenna_positions.uvw
    )
    # The hard-coded seed 42 selects these rows.
    assert_is_row_subset(obs, first, GENERATED_MASK_10_50)


@pmp("n_rows,percentage,expected", ((10, 0.25, 2), (10, 0.5, 5), (8, 0.5, 4), (7, 0.5, 3)))
def test_percentage_truncates_towards_zero(n_rows, percentage, expected):
    np.random.seed(72 + n_rows)
    obs = build_obs(n_rows)

    new = select_random_visibility_subset(obs, SelectSubset(percentage=percentage))

    assert new.nrow == expected
    assert new.npol == obs.npol
    assert new.nfreq == obs.nfreq


def test_percentage_one_keeps_all_rows_in_order():
    np.random.seed(73)
    obs = build_obs()

    new = select_random_visibility_subset(obs, SelectSubset(percentage=1.0))

    assert_is_row_subset(obs, new, np.arange(obs.nrow))


@pmp("percentage", (0, 0.0, 0.09))
def test_empty_selection_raises(percentage):
    np.random.seed(74)
    obs = build_obs()

    with pytest.raises(ValueError):
        select_random_visibility_subset(obs, SelectSubset(percentage=percentage))


@pmp("percentage", (25, 1.5, -0.1))
def test_percentage_outside_unit_interval_raises(percentage):
    np.random.seed(75)
    obs = build_obs()

    with pytest.raises(ValueError):
        select_random_visibility_subset(obs, SelectSubset(percentage=percentage))


def test_missing_percentage_without_existing_mask_raises(tmp_path):
    np.random.seed(76)
    obs = build_obs()
    # This is exactly what the yaml parser produces for a config that only
    # sets `mask_path`.
    select_subset = SelectSubset.from_yaml_dict(
        {"mask_path": str(tmp_path / "mask.npy")}
    )
    assert select_subset.percentage is None

    with pytest.raises(ValueError):
        select_random_visibility_subset(obs, select_subset)


def test_mask_is_saved_and_reloaded(tmp_path):
    np.random.seed(77)
    obs = build_obs(calib_info=True)
    mask_file = tmp_path / "nested" / "dirs" / "mask.npy"
    select_subset = SelectSubset(percentage=0.5, mask_path=str(mask_file))

    first = select_random_visibility_subset(obs, select_subset)

    assert mask_file.exists()
    assert_array_equal(np.load(mask_file), GENERATED_MASK_10_50)

    second = select_random_visibility_subset(obs, select_subset)

    assert first == second
    assert_is_row_subset(obs, second, GENERATED_MASK_10_50)


def test_existing_mask_is_used_instead_of_a_new_one(tmp_path):
    np.random.seed(78)
    obs = build_obs(calib_info=True)
    mask = np.array([1, 3, 4, 7, 9])
    mask_file = tmp_path / "mask.npy"
    np.save(mask_file, mask)

    new = select_random_visibility_subset(
        obs, SelectSubset(percentage=0.5, mask_path=str(mask_file))
    )

    assert_is_row_subset(obs, new, mask)
    assert_array_equal(new.antenna_positions.ant1, obs.antenna_positions.ant1[mask])
    assert_array_equal(new.antenna_positions.ant2, obs.antenna_positions.ant2[mask])
    assert_array_equal(new.antenna_positions.time, obs.antenna_positions.time[mask])


def test_mask_path_without_npy_suffix_round_trips(tmp_path):
    np.random.seed(79)
    obs = build_obs()
    # np.save appends `.npy`, so the existence check must look for that file.
    mask_path = tmp_path / "mask_without_suffix"
    select_subset = SelectSubset(percentage=0.5, mask_path=str(mask_path))

    select_random_visibility_subset(obs, select_subset)
    saved = tmp_path / "mask_without_suffix.npy"
    assert saved.exists()

    # Overwrite the stored mask; a second call must load it instead of
    # regenerating the seed-42 mask.
    other_mask = np.array([1, 3, 4, 7, 9])
    np.save(saved, other_mask)
    new = select_random_visibility_subset(obs, select_subset)

    assert_is_row_subset(obs, new, other_mask)


def test_stored_mask_out_of_bounds_raises(tmp_path):
    np.random.seed(80)
    obs_large = build_obs(100)
    obs_small = build_obs(10)
    mask_file = tmp_path / "mask.npy"
    select_subset = SelectSubset(percentage=0.5, mask_path=str(mask_file))

    select_random_visibility_subset(obs_large, select_subset)

    with pytest.raises(ValueError):
        select_random_visibility_subset(obs_small, select_subset)


def test_stored_mask_of_wrong_size_raises(tmp_path):
    np.random.seed(81)
    obs_small = build_obs(10)
    obs_large = build_obs(100)
    mask_file = tmp_path / "mask.npy"
    select_subset = SelectSubset(percentage=0.5, mask_path=str(mask_file))

    small = select_random_visibility_subset(obs_small, select_subset)
    assert small.nrow == 5

    # Reusing the 10-row mask for a 100-row observation would silently keep 5
    # instead of 50 rows.
    with pytest.raises(ValueError):
        select_random_visibility_subset(obs_large, select_subset)


def test_imaging_only_observation_keeps_nones():
    np.random.seed(82)
    obs = build_obs(calib_info=False)
    assert obs.antenna_positions.only_imaging

    new = select_random_visibility_subset(obs, SelectSubset(percentage=0.5))

    assert new.antenna_positions.only_imaging
    assert new.antenna_positions.ant1 is None
    assert new.antenna_positions.ant2 is None
    assert new.antenna_positions.time is None
    assert_is_row_subset(obs, new, GENERATED_MASK_10_50)


def test_input_observation_is_not_modified():
    np.random.seed(83)
    obs = build_obs(calib_info=True)
    vis_before = obs.vis.asnumpy().copy()
    weight_before = obs.weight.asnumpy().copy()

    new = select_random_visibility_subset(obs, SelectSubset(percentage=0.5))

    assert obs.nrow == 10
    assert_array_equal(obs.vis.asnumpy(), vis_before)
    assert_array_equal(obs.weight.asnumpy(), weight_before)
    assert new._vis is not obs._vis
    assert new.legacy_polarization == obs.legacy_polarization


def test_select_subset_from_yaml_dict_bare_float():
    select_subset = SelectSubset.from_yaml_dict(0.25)

    assert select_subset.percentage == 0.25
    assert select_subset.mask_path is None
