import numpy as np
import pytest
from astropy import units as u
from numpy.testing import assert_allclose, assert_array_equal

import jubik as ju
import jubik.instruments.resolve as rve
from jubik.instruments.resolve.data.data_modify.flagging import flag_weights
from jubik.instruments.resolve.data.data_modify.frequency import (
    exclude_frequency_ranges,
    freq_average_by_fdom_and_n_freq_chunks,
)
from jubik.instruments.resolve.data.data_modify.phase_center import shift_phase_center
from jubik.instruments.resolve.data.data_modify.pipeline import modify_observation
from jubik.instruments.resolve.data.data_modify.time import (
    time_average_to_length_of_timebins,
)
from jubik.instruments.resolve.parse.data.data_modify.flagging import FlagWeights
from jubik.instruments.resolve.parse.data.data_modify.pipeline import ObservationModify
from jubik.instruments.resolve.parse.data.data_modify.phase_center import (
    ShiftObservation,
)

from generate_test_obs import generate_random_obs

pmp = pytest.mark.parametrize

SPEEDOFLIGHT = 299792458.0


def pol_type():
    return ju.polarization.PolarizationType(("LL", "RR"))


def build_obs(freqs, n_rows=6, ant1=None, ant2=None, times=None):
    return generate_random_obs(
        np.asarray(freqs),
        n_rows,
        [-1.0e3, 1.0e3],
        [-5.0, 5.0],
        pol_type(),
        ant1=ant1,
        ant2=ant2,
        times=times,
    )


def with_weight(obs, index, value):
    """Copy of `obs` with a single weight entry replaced."""
    weight = obs.weight_val.copy()
    weight[index] = value
    return rve.Observation(
        obs.antenna_positions,
        obs.vis_val.copy(),
        weight,
        obs.legacy_polarization,
        obs.freq,
        None,
    )


def continuous_sky(fmin, fmax):
    return ju.Color([fmin, fmax] * u.Hz)


# ---------------------------------------------------------------------------
# The default configuration
# ---------------------------------------------------------------------------


def test_default_modify_is_identity():
    np.random.seed(200)
    freqs = 1.0e9 + np.arange(3) * 0.1e9
    obs = build_obs(freqs)
    modify = ObservationModify.from_yaml_dict({})(0)

    new = modify_observation(continuous_sky(0.95e9, 1.25e9), obs, modify)

    assert_array_equal(new.vis_val, obs.vis_val)
    assert_array_equal(new.weight_val, obs.weight_val)
    assert_array_equal(new.freq, obs.freq)
    assert new.antenna_positions is obs.antenna_positions
    # The only documented step of a default config is the cast to double
    # precision, which is a no-op for data that already is double precision.
    assert new.vis_val.dtype == np.complex128
    assert modify.to_double_precision


def test_default_modify_upcasts_single_precision_data():
    np.random.seed(201)
    freqs = 1.0e9 + np.arange(2) * 0.1e9
    obs = build_obs(freqs)
    single = rve.Observation(
        obs.antenna_positions,
        obs.vis_val.astype(np.complex64),
        obs.weight_val.astype(np.float32),
        obs.legacy_polarization,
        obs.freq,
        None,
    )
    modify = ObservationModify.from_yaml_dict({})(0)

    new = modify_observation(continuous_sky(0.95e9, 1.15e9), single, modify)

    assert new.vis_val.dtype == np.complex128
    assert new.weight_val.dtype == np.float64
    assert_allclose(new.vis_val, obs.vis_val.astype(np.complex64))


def test_modify_does_not_mutate_its_input_and_is_idempotent():
    np.random.seed(202)
    freqs = 1.0e9 + np.arange(2) * 0.1e9
    obs = build_obs(freqs, n_rows=5)
    vis_before = obs.vis_val.copy()
    weight_before = obs.weight_val.copy()
    modify = ObservationModify.from_yaml_dict({"weight_modify": {"percentage": 0.1}})(0)
    sky = continuous_sky(0.95e9, 1.15e9)

    first = modify_observation(sky, obs, modify)

    assert_array_equal(obs.vis_val, vis_before)
    assert_array_equal(obs.weight_val, weight_before)

    second = modify_observation(sky, obs, modify)
    assert_array_equal(first.weight_val, second.weight_val)
    assert_array_equal(first.vis_val, second.vis_val)


# ---------------------------------------------------------------------------
# Step ordering: flagging happens before averaging
# ---------------------------------------------------------------------------


def test_flag_weights_runs_before_frequency_averaging():
    """A weight outlier is diluted by the harmonic frequency average.

    Flag-then-average drops the offending row, average-then-flag does not, so
    the two orders differ. The pipeline must produce the flag-first result.
    """
    np.random.seed(203)
    freqs = 1.0e9 + np.arange(6) * 0.1e9
    obs = with_weight(build_obs(freqs, n_rows=6), (0, 3, 2), 1.0e14)
    sky = continuous_sky(0.95e9, 1.55e9)
    setting = FlagWeights(min=1e-12, max=1e12)
    modify = ObservationModify.from_yaml_dict(
        {"flag_weights": {"min": 1e-12, "max": 1e12}, "spectral": {"bins": 2}}
    )(0)

    new = modify_observation(sky, obs, modify)

    flag_first = freq_average_by_fdom_and_n_freq_chunks(
        sky, flag_weights(obs, setting), 2
    )
    average_first = flag_weights(
        freq_average_by_fdom_and_n_freq_chunks(sky, obs, 2), setting
    )

    # The two orders really differ: averaging dilutes the outlier.
    assert flag_first.nrow == 5
    assert average_first.nrow == 6
    assert np.max(average_first.weight_val) < 1e12

    assert new.nrow == 5
    assert_array_equal(new.vis_val, flag_first.vis_val)
    assert_array_equal(new.weight_val, flag_first.weight_val)
    assert_array_equal(new.freq, flag_first.freq)


def test_flag_weights_runs_before_time_averaging():
    """The same for the time average, which sums the weights.

    A too small weight survives the sum, so average-then-flag keeps a row that
    flag-then-average removes from the average.
    """
    np.random.seed(204)
    freqs = 1.0e9 + np.arange(2) * 0.1e9
    ant1 = np.array([0, 1, 0, 1, 0, 1])
    ant2 = np.array([2, 3, 2, 3, 2, 3])
    times = np.array([0.0, 0.0, 10.0, 10.0, 20.0, 20.0])
    obs = with_weight(
        build_obs(freqs, n_rows=6, ant1=ant1, ant2=ant2, times=times),
        (0, 0, 0),
        1.0e-6,
    )
    sky = continuous_sky(0.95e9, 1.15e9)
    setting = FlagWeights(min=1e-3, max=1e12)
    modify = ObservationModify.from_yaml_dict(
        {"flag_weights": {"min": 1e-3, "max": 1e12}, "time_bins": 30}
    )(0)

    new = modify_observation(sky, obs, modify)

    flag_first = time_average_to_length_of_timebins(flag_weights(obs, setting), 30)
    average_first = flag_weights(
        time_average_to_length_of_timebins(obs, 30), setting
    )

    # Both orders keep the two baselines, but the averaged values differ.
    assert flag_first.nrow == 2
    assert average_first.nrow == 2
    assert not np.allclose(flag_first.vis_val, average_first.vis_val)

    assert_array_equal(new.vis_val, flag_first.vis_val)
    assert_array_equal(new.weight_val, flag_first.weight_val)


# ---------------------------------------------------------------------------
# Step ordering: frequency exclusion before frequency averaging
# ---------------------------------------------------------------------------


def test_exclude_frequency_ranges_runs_before_frequency_averaging():
    np.random.seed(205)
    freqs = 1.0e9 + np.arange(6) * 0.1e9
    obs = build_obs(freqs)
    sky = continuous_sky(0.95e9, 1.55e9)
    ranges = [[1.15e9, 1.35e9]]
    modify = ObservationModify.from_yaml_dict(
        {"spectral": {"bins": 1, "exclude_frequency_ranges": ranges}}
    )(0)

    new = modify_observation(sky, obs, modify)

    keep = np.array([0, 1, 4, 5])
    # One output channel, the average of the four surviving channels only.
    assert new.nfreq == 1
    assert_allclose(new.freq, [np.mean(freqs[keep])])
    assert_allclose(new.vis_val[:, :, 0], np.mean(obs.vis_val[:, :, keep], axis=2))
    assert_allclose(
        new.weight_val[:, :, 0],
        len(keep) ** 2 / np.sum(1 / obs.weight_val[:, :, keep], axis=2),
    )
    # The excluded channels never enter the average.
    assert not np.allclose(new.vis_val[:, :, 0], np.mean(obs.vis_val, axis=2))

    # The other order is not even well defined here: the average of all six
    # channels sits at 1.25e9, i.e. inside the excluded range.
    averaged = freq_average_by_fdom_and_n_freq_chunks(sky, obs, 1)
    with pytest.raises(ValueError):
        exclude_frequency_ranges(averaged, ranges)


# ---------------------------------------------------------------------------
# Frequency reversal
# ---------------------------------------------------------------------------


def test_descending_frequencies_are_reversed():
    np.random.seed(206)
    freqs = (1.0e9 + np.arange(4) * 0.1e9)[::-1]
    obs = build_obs(freqs)
    modify = ObservationModify.from_yaml_dict({})(0)

    new = modify_observation(continuous_sky(0.95e9, 1.35e9), obs, modify)

    assert_array_equal(new.freq, freqs[::-1])
    assert np.all(np.diff(new.freq) > 0)
    assert_array_equal(new.vis_val, obs.vis_val[:, :, ::-1])
    assert_array_equal(new.weight_val, obs.weight_val[:, :, ::-1])


def test_ascending_frequencies_are_not_reversed():
    np.random.seed(207)
    freqs = 1.0e9 + np.arange(4) * 0.1e9
    obs = build_obs(freqs)
    modify = ObservationModify.from_yaml_dict({})(0)

    new = modify_observation(continuous_sky(0.95e9, 1.35e9), obs, modify)

    assert_array_equal(new.freq, freqs)
    assert_array_equal(new.vis_val, obs.vis_val)


def test_single_channel_observation_is_not_reversed():
    np.random.seed(208)
    obs = build_obs(np.array([1.0e9]))
    modify = ObservationModify.from_yaml_dict({})(0)

    new = modify_observation(continuous_sky(0.95e9, 1.05e9), obs, modify)

    assert_array_equal(new.freq, obs.freq)
    assert_array_equal(new.vis_val, obs.vis_val)


# ---------------------------------------------------------------------------
# Phase center shift
# ---------------------------------------------------------------------------


def test_shift_is_applied_by_the_pipeline():
    np.random.seed(209)
    freqs = 1.0e9 + np.arange(2) * 0.1e9
    obs = build_obs(freqs, n_rows=4)
    sx, sy = 2.0e-4, -5.0e-4
    modify = ObservationModify.from_yaml_dict(
        {"shift": {"data_templates": [[sx, sy]], "unit": "rad"}}
    )(0)

    new = modify_observation(continuous_sky(0.95e9, 1.15e9), obs, modify)

    expected = shift_phase_center(obs, ShiftObservation([sx, sy] * u.rad))
    assert_array_equal(new.vis_val, expected.vis_val)
    assert_array_equal(new.weight_val, obs.weight_val)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Known ordering problem, reported to the maintainers: "
        "`shift_phase_center` is the last step of `modify_observation` and "
        "therefore runs after the frequency average. The phase factor varies "
        "across the averaged channels, so the source is decorrelated before it "
        "can be moved onto the phase center. A phase-center shift has to be "
        "applied before any averaging."
    ),
)
def test_shift_should_be_applied_before_frequency_averaging():
    np.random.seed(210)
    freqs = np.linspace(1.0e9, 2.0e9, 8)
    obs = build_obs(freqs, n_rows=8)
    # A single point source at the sky offset (sx, 0), see
    # test_phase_center.test_response_phase_convention...
    sx = 4.0e-4
    uu = obs.uvw[:, 0, None] * freqs[None] / SPEEDOFLIGHT
    vis = np.broadcast_to(
        np.exp(-2j * np.pi * uu * sx)[None], obs.vis_val.shape
    ).astype(np.complex128)
    point = rve.Observation(
        obs.antenna_positions,
        vis.copy(),
        np.ones(vis.shape),
        obs.legacy_polarization,
        freqs,
        None,
    )
    sky = continuous_sky(0.95e9, 2.05e9)
    modify = ObservationModify.from_yaml_dict(
        {
            "spectral": {"bins": 1},
            "shift": {"data_templates": [[sx, 0.0]], "unit": "rad"},
        }
    )(0)

    new = modify_observation(sky, point, modify)

    # Shifting first and averaging afterwards gives unit amplitude everywhere.
    shifted = shift_phase_center(point, ShiftObservation([sx, 0.0] * u.rad))
    shift_first = freq_average_by_fdom_and_n_freq_chunks(sky, shifted, 1)
    assert_allclose(np.abs(shift_first.vis_val), 1.0, atol=1e-12)

    assert_allclose(np.abs(new.vis_val), 1.0, atol=1e-6)
