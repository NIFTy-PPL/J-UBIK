import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import jubik as ju
from jubik.instruments.resolve.data.auxiliary_table import AuxiliaryTable
from jubik.instruments.resolve.data.data_modify.flagging import (
    flag_baseline,
    flag_station,
    flag_weights,
)
from jubik.instruments.resolve.data.observation import Observation
from jubik.instruments.resolve.parse.data.data_modify.flagging import FlagWeights

from generate_test_obs import generate_random_obs

pmp = pytest.mark.parametrize

FREQS = np.array([1.0e9, 1.1e9, 1.2e9])

# Baselines of a four-antenna array, ant1 < ant2 for every row.
ANT1 = np.array([0, 0, 0, 1, 1, 2, 0, 1], dtype=np.int64)
ANT2 = np.array([1, 2, 3, 2, 3, 3, 1, 2], dtype=np.int64)
TIMES = np.arange(8, dtype=np.float64) * 10.0


def build_obs(freqs=FREQS, n_rows=8, with_calib=False, **kwargs):
    """Random observation, optionally carrying calibration information."""
    pol_type = ju.polarization.PolarizationType(("LL", "RR"))
    if with_calib:
        kwargs.update(ant1=ANT1[:n_rows], ant2=ANT2[:n_rows], times=TIMES[:n_rows])
    return generate_random_obs(freqs, n_rows, [-1e2, 1e2], [-5, 5], pol_type, **kwargs)


def replace(obs, weight=None, auxiliary_tables=None):
    """Copy of `obs` with new weights and/or auxiliary tables."""
    return Observation(
        obs.antenna_positions,
        obs.vis_val,
        obs.weight_val if weight is None else weight,
        obs.legacy_polarization,
        obs.freq,
        auxiliary_tables,
    )


def antenna_table(n_antennas=4):
    names = np.array([f"ANT{ii}" for ii in range(n_antennas)])
    return AuxiliaryTable({"NAME": names, "STATION": np.array(names)})


def assert_rows_kept(obs, new, keep):
    keep = np.asarray(keep)
    assert new.nrow == len(keep)
    assert_array_equal(new.vis_val, obs.vis_val[:, keep, :])
    assert_array_equal(new.weight_val, obs.weight_val[:, keep, :])
    assert_array_equal(new.antenna_positions.uvw, obs.antenna_positions.uvw[keep, :])


# --------------------------------------------------------------- flag_weights


def test_flag_weights_none_is_noop():
    np.random.seed(101)
    obs = build_obs()

    assert flag_weights(obs, None) is obs


def test_flag_weights_keeps_everything_when_all_weights_are_in_range():
    np.random.seed(102)
    obs = build_obs()
    # generate_random_obs draws the weights from U(1, 10).
    new = flag_weights(obs, FlagWeights(min=1e-12, max=1e12))

    assert_rows_kept(obs, new, np.arange(obs.nrow))


def test_flag_weights_drops_rows_with_any_out_of_range_entry():
    np.random.seed(103)
    obs = build_obs()
    wgt = obs.weight_val.copy()
    wgt[0, 2, 1] = 1e15  # too big
    wgt[1, 5, 0] = 1e-15  # too small
    obs = replace(obs, wgt)

    new = flag_weights(obs, FlagWeights(min=1e-12, max=1e12))

    assert_rows_kept(obs, new, [0, 1, 3, 4, 6, 7])


@pmp("index", ((0, 0, 0), (1, 3, 2), (0, 7, 1)))
def test_flag_weights_drops_exactly_one_row_per_corrupted_entry(index):
    np.random.seed(104)
    obs = build_obs()
    wgt = obs.weight_val.copy()
    wgt[index] = 1e15
    obs = replace(obs, wgt)

    new = flag_weights(obs, FlagWeights(min=1e-12, max=1e12))

    keep = [ii for ii in range(obs.nrow) if ii != index[1]]
    assert_rows_kept(obs, new, keep)


def test_flag_weights_bounds_are_exclusive():
    np.random.seed(105)
    # All random weights lie strictly inside [0.5, 2.0].
    obs = build_obs(n_rows=4, weight_range=[1.0, 1.9])
    wgt = obs.weight_val.copy()
    wgt[0, 0, 0] = 2.0  # == max, survives
    wgt[0, 1, 0] = 0.5  # == min, survives
    wgt[0, 2, 0] = 2.0 + 1e-9  # > max, dropped
    wgt[0, 3, 0] = 0.5 - 1e-9  # < min, dropped
    obs = replace(obs, wgt)

    new = flag_weights(obs, FlagWeights(min=0.5, max=2.0))

    assert_rows_kept(obs, new, [0, 1])


def test_flag_weights_drops_rows_that_contain_a_zero_weight():
    """A zero weight is the flag encoding, it is `< min` and drops the row."""
    np.random.seed(106)
    obs = build_obs(n_rows=4)
    wgt = obs.weight_val.copy()
    wgt[0, 1, 0] = 0.0
    obs = replace(obs, wgt)

    new = flag_weights(obs, FlagWeights())

    assert_rows_kept(obs, new, [0, 2, 3])


def test_flag_weights_masks_calibration_information_consistently():
    np.random.seed(107)
    obs = build_obs(with_calib=True)
    wgt = obs.weight_val.copy()
    wgt[0, 2, 1] = 1e15
    obs = replace(obs, wgt)

    new = flag_weights(obs, FlagWeights(min=1e-12, max=1e12))

    keep = np.array([0, 1, 3, 4, 5, 6, 7])
    assert_rows_kept(obs, new, keep)
    assert_array_equal(new.ant1, obs.ant1[keep])
    assert_array_equal(new.ant2, obs.ant2[keep])
    assert_array_equal(new.time, obs.time[keep])


def test_flag_weights_keeps_only_imaging_antenna_positions_without_calib_info():
    np.random.seed(108)
    obs = build_obs()
    wgt = obs.weight_val.copy()
    wgt[1, 0, 2] = 1e15
    obs = replace(obs, wgt)

    new = flag_weights(obs, FlagWeights(min=1e-12, max=1e12))

    assert new.antenna_positions.only_imaging
    assert new.ant1 is None
    assert new.ant2 is None
    assert new.time is None
    assert_rows_kept(obs, new, np.arange(1, obs.nrow))


def test_flag_weights_keeps_auxiliary_tables_and_frequencies():
    np.random.seed(109)
    obs = build_obs(with_calib=True)
    wgt = obs.weight_val.copy()
    wgt[0, 3, 0] = 1e15
    obs = replace(obs, wgt, {"ANTENNA": antenna_table()})

    new = flag_weights(obs, FlagWeights(min=1e-12, max=1e12))

    assert_array_equal(new.freq, obs.freq)
    assert new.auxiliary_table("ANTENNA") == obs.auxiliary_table("ANTENNA")


def test_flag_weights_does_not_modify_the_input():
    np.random.seed(110)
    obs = build_obs(with_calib=True)
    wgt = obs.weight_val.copy()
    wgt[0, 2, 1] = 1e15
    obs = replace(obs, wgt)
    before = obs.weight_val.copy()

    flag_weights(obs, FlagWeights(min=1e-12, max=1e12))

    assert obs.nrow == 8
    assert_array_equal(obs.weight_val, before)


def test_flag_weights_raises_if_all_rows_would_be_removed():
    np.random.seed(111)
    obs = build_obs(n_rows=4)
    # All weights are drawn from U(1, 10), so every row is below `min`.
    with pytest.raises(ValueError):
        flag_weights(obs, FlagWeights(min=100.0, max=1e12))


# ---------------------------------------------------- flag_baseline / station


@pmp("func,args", ((flag_baseline, (0, 1)), (flag_station, (1,))))
def test_flagging_without_calibration_information_raises(func, args):
    np.random.seed(112)
    obs = build_obs()

    with pytest.raises(RuntimeError):
        func(obs, *args)


def test_flag_baseline_zeros_only_the_matching_rows():
    np.random.seed(113)
    obs = build_obs(with_calib=True)
    obs = replace(obs, auxiliary_tables={"ANTENNA": antenna_table()})
    before = obs.weight_val.copy()

    new = flag_baseline(obs, 0, 1)

    ind = np.array([True, False, False, False, False, False, True, False])
    assert new is not obs
    assert_array_equal(new.weight_val[:, ind, :], 0.0)
    assert_array_equal(new.weight_val[:, ~ind, :], obs.weight_val[:, ~ind, :])
    assert_array_equal(new.vis_val, obs.vis_val)
    assert_array_equal(obs.weight_val, before)


def test_flag_station_zeros_every_row_touching_the_antenna():
    np.random.seed(114)
    obs = build_obs(with_calib=True)
    obs = replace(obs, auxiliary_tables={"ANTENNA": antenna_table()})
    before = obs.weight_val.copy()

    new = flag_station(obs, 3)

    ind = np.logical_or(obs.ant1 == 3, obs.ant2 == 3)
    assert_array_equal(ind, [False, False, True, False, True, True, False, False])
    assert_array_equal(new.weight_val[:, ind, :], 0.0)
    assert_array_equal(new.weight_val[:, ~ind, :], obs.weight_val[:, ~ind, :])
    assert_array_equal(obs.weight_val, before)


@pmp("func,args", ((flag_baseline, (0, 1)), (flag_station, (0,))))
def test_flagging_without_antenna_table_works_if_nothing_matches(func, args):
    """No row matches, so no antenna name has to be looked up."""
    np.random.seed(115)
    ant1 = np.array([1, 1, 2, 2], dtype=np.int64)
    ant2 = np.array([2, 3, 3, 3], dtype=np.int64)
    pol_type = ju.polarization.PolarizationType(("LL", "RR"))
    obs = generate_random_obs(
        FREQS,
        4,
        [-1e2, 1e2],
        [-5, 5],
        pol_type,
        ant1=ant1,
        ant2=ant2,
        times=np.arange(4, dtype=np.float64),
    )
    assert obs._auxiliary_tables is None

    new = func(obs, *args)

    assert_array_equal(new.weight_val, obs.weight_val)


@pmp("func,args", ((flag_baseline, (0, 7)), (flag_station, (7,))))
def test_flagging_an_antenna_index_that_is_not_in_the_table_is_a_noop(func, args):
    """Index 7 is out of range for the 4-antenna table, but matches no row."""
    np.random.seed(116)
    obs = build_obs(with_calib=True)
    obs = replace(obs, auxiliary_tables={"ANTENNA": antenna_table(4)})

    new = func(obs, *args)

    assert_array_equal(new.weight_val, obs.weight_val)


@pmp("func,args", ((flag_baseline, (0, 1)), (flag_station, (1,))))
def test_flagging_rejects_data_that_is_not_ordered_ant1_smaller_ant2(func, args):
    np.random.seed(117)
    ant1 = np.array([0, 1, 1, 2], dtype=np.int64)
    ant2 = np.array([1, 1, 2, 3], dtype=np.int64)  # row 1 is an autocorrelation
    pol_type = ju.polarization.PolarizationType(("LL", "RR"))
    obs = generate_random_obs(
        FREQS,
        4,
        [-1e2, 1e2],
        [-5, 5],
        pol_type,
        ant1=ant1,
        ant2=ant2,
        times=np.arange(4, dtype=np.float64),
    )

    with pytest.raises(RuntimeError):
        func(obs, *args)


def test_flag_baseline_of_all_rows_gives_an_entirely_flagged_observation():
    np.random.seed(118)
    ant1 = np.zeros(4, dtype=np.int64)
    ant2 = np.ones(4, dtype=np.int64)
    pol_type = ju.polarization.PolarizationType(("LL", "RR"))
    obs = generate_random_obs(
        FREQS,
        4,
        [-1e2, 1e2],
        [-5, 5],
        pol_type,
        ant1=ant1,
        ant2=ant2,
        times=np.arange(4, dtype=np.float64),
    )
    obs = replace(obs, auxiliary_tables={"ANTENNA": antenna_table()})

    new = flag_baseline(obs, 0, 1)

    assert new.nrow == obs.nrow
    assert_allclose(new.weight_val, 0.0)
    assert_array_equal(new.antenna_positions.uvw, obs.antenna_positions.uvw)
