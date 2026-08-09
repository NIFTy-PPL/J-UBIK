import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import jubik as ju
from jubik.instruments.resolve.data.data_modify.precision import to_single_precision
from jubik.instruments.resolve.data.data_modify.weights import (
    systematic_error_budget,
)
from jubik.instruments.resolve.data.observation import Observation
from jubik.instruments.resolve.parse.data.data_modify.weights import (
    SystematicErrorBudget,
)

from generate_test_obs import generate_random_obs

pmp = pytest.mark.parametrize

FREQS = np.array([1.0e9, 1.1e9, 1.2e9])


def build_obs(freqs=FREQS, n_rows=8, **kwargs):
    pol_type = ju.polarization.PolarizationType(("LL", "RR"))
    return generate_random_obs(freqs, n_rows, [-1e2, 1e2], [-5, 5], pol_type, **kwargs)


def expected_weight(weight_old, vis, percentage):
    """1 / (sigma**2 + (percentage * |A|)**2), flagged entries stay flagged."""
    good = weight_old > 0.0
    out = np.zeros_like(weight_old)
    out[good] = 1.0 / (
        1.0 / weight_old[good] + (percentage * np.abs(vis[good])) ** 2
    )
    return out


def test_systematic_none_is_noop():
    np.random.seed(201)
    obs = build_obs()

    assert systematic_error_budget(obs, None) is obs


@pmp("percentage", (0.01, 0.05, 0.5))
def test_systematic_applies_the_documented_formula(percentage):
    np.random.seed(202)
    obs = build_obs()
    weight_old = obs.weight_val.copy()
    vis = obs.vis_val.copy()

    new = systematic_error_budget(obs, SystematicErrorBudget(percentage))

    assert_allclose(
        new.weight_val,
        1.0 / (1.0 / weight_old + (percentage * np.abs(vis)) ** 2),
        rtol=1e-14,
    )
    # A systematic error can only decrease the weights.
    assert np.all(new.weight_val < weight_old)


def test_systematic_with_zero_percentage_keeps_the_weights():
    np.random.seed(203)
    obs = build_obs()
    weight_old = obs.weight_val.copy()

    new = systematic_error_budget(obs, SystematicErrorBudget(0.0))

    assert_allclose(new.weight_val, weight_old, rtol=1e-14)


def test_systematic_returns_a_new_observation_and_leaves_the_input_untouched():
    np.random.seed(204)
    obs = build_obs()
    weight_old = obs.weight_val.copy()

    new = systematic_error_budget(obs, SystematicErrorBudget(0.05))

    assert new is not obs
    assert isinstance(new, Observation)
    assert_array_equal(obs.weight_val, weight_old)
    assert not obs.weight_val.flags.writeable
    assert not new.weight_val.flags.writeable
    assert np.any(new.weight_val != weight_old)


def test_systematic_keeps_visibilities_frequencies_and_antenna_positions():
    np.random.seed(205)
    obs = build_obs()

    new = systematic_error_budget(obs, SystematicErrorBudget(0.05))

    assert_array_equal(new.vis_val, obs.vis_val)
    assert_array_equal(new.freq, obs.freq)
    assert new.antenna_positions == obs.antenna_positions
    assert new.nrow == obs.nrow
    assert new.npol == obs.npol
    assert new.nfreq == obs.nfreq


def test_systematic_keeps_flagged_entries_at_exactly_zero():
    np.random.seed(206)
    obs = build_obs(fraction_flagged=0.3)
    flagged = obs.weight_val == 0.0
    assert flagged.sum() > 0
    weight_old = obs.weight_val.copy()
    vis = obs.vis_val.copy()

    with np.errstate(all="raise"):
        new = systematic_error_budget(obs, SystematicErrorBudget(0.05))

    assert_array_equal(new.weight_val[flagged], 0.0)
    assert np.all(np.isfinite(new.weight_val))
    assert_allclose(new.weight_val, expected_weight(weight_old, vis, 0.05), rtol=1e-14)
    assert_array_equal(new.flags_val, obs.flags_val)


def test_systematic_does_not_produce_nan_for_nan_visibilities_at_flagged_points():
    np.random.seed(207)
    obs = build_obs(fraction_flagged=0.3).flags_to_nan()
    flagged = obs.weight_val == 0.0
    assert np.any(np.isnan(obs.vis_val))

    with np.errstate(all="raise"):
        new = systematic_error_budget(obs, SystematicErrorBudget(0.05))

    assert np.all(np.isfinite(new.weight_val))
    assert_array_equal(new.weight_val[flagged], 0.0)


def test_systematic_preserves_single_precision():
    np.random.seed(208)
    obs = to_single_precision(build_obs())

    new = systematic_error_budget(obs, SystematicErrorBudget(0.05))

    assert new.weight_val.dtype == np.float32
    assert new.is_single_precision()


def test_systematic_error_budget_parses_percentage():
    assert SystematicErrorBudget.from_yaml_dict({"percentage": 0.05}).percentage == 0.05
    assert SystematicErrorBudget.from_yaml_dict({}) is None
