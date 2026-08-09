import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import jubik.instruments.resolve as rve
from jubik.instruments.resolve.data.data_modify.polarization import (
    average_stokesi,
    restrict_to_polarization,
    restrict_to_stokesi,
)
from jubik.polarization import Polarization, PolarizationType

from generate_test_obs import generate_random_obs

pmp = pytest.mark.parametrize

FREQS = np.array([1.0e9, 1.1e9, 1.2e9])
N_ROWS = 7


def build_obs(polarization_type, fraction_flagged=0.0, seed=42):
    np.random.seed(seed)
    return generate_random_obs(
        FREQS,
        N_ROWS,
        [-1e2, 1e2],
        [-5, 5],
        polarization_type,
        fraction_flagged=fraction_flagged,
    )


def build_obs_from_arrays(vis, weight, polarization):
    uvw = np.random.uniform(-1e2, 1e2, size=(vis.shape[1], 3))
    antpos = rve.AntennaPositions(uvw, None, None, None)
    return rve.Observation(antpos, vis, weight, polarization, FREQS, None)


# ---------------------------------------------------------------- restrict_to_stokesi


def test_restrict_to_stokesi_is_noop_for_stokes_i_data():
    obs = build_obs(PolarizationType.I)
    assert restrict_to_stokesi(obs) is obs


@pmp(
    "polarization_type,keep,labels",
    (
        (PolarizationType.RR_RL_LR_LL, [3, 0], ("LL", "RR")),
        (PolarizationType.XX_XY_YX_YY, [0, 3], ("XX", "YY")),
        (PolarizationType.RR_LL, [1, 0], ("LL", "RR")),
        (PolarizationType.XX_YY, [0, 1], ("XX", "YY")),
    ),
)
def test_restrict_to_stokesi_selects_parallel_hands(polarization_type, keep, labels):
    obs = build_obs(polarization_type)
    new = restrict_to_stokesi(obs)

    assert new.npol == 2
    assert new.vis.domain[0].labels == labels
    assert_array_equal(new.vis_val, obs.vis_val[keep])
    assert_array_equal(new.weight_val, obs.weight_val[keep])


def test_restrict_to_stokesi_keeps_metadata():
    obs = build_obs(PolarizationType.RR_RL_LR_LL)
    new = restrict_to_stokesi(obs)

    assert new.nrow == obs.nrow
    assert new.nfreq == obs.nfreq
    assert_array_equal(new.freq, obs.freq)
    assert new.antenna_positions is obs.antenna_positions


def test_restrict_to_stokesi_is_idempotent():
    obs = build_obs(PolarizationType.XX_XY_YX_YY)
    once = restrict_to_stokesi(obs)
    twice = restrict_to_stokesi(once)

    assert_array_equal(twice.vis_val, once.vis_val)
    assert twice.legacy_polarization == once.legacy_polarization


def test_restrict_to_stokesi_raises_without_both_parallel_hands():
    # RR and RL only: LL is missing, so Stokes I cannot be formed.
    obs = build_obs(PolarizationType.RR)
    with pytest.raises(ValueError):
        restrict_to_stokesi(obs)


# ----------------------------------------------------------- restrict_to_polarization


@pmp("label,index", (("RR", 0), ("RL", 1), ("LR", 2), ("LL", 3)))
def test_restrict_to_polarization_extracts_single_product(label, index):
    obs = build_obs(PolarizationType.RR_RL_LR_LL)
    new = restrict_to_polarization(obs, label)

    assert new.npol == 1
    assert_array_equal(new.vis_val, obs.vis_val[index : index + 1])
    assert_array_equal(new.weight_val, obs.weight_val[index : index + 1])


def test_restrict_to_polarization_keeps_metadata():
    obs = build_obs(PolarizationType.XX_XY_YX_YY)
    new = restrict_to_polarization(obs, "XY")

    assert new.nrow == obs.nrow
    assert_array_equal(new.freq, obs.freq)
    assert new.antenna_positions is obs.antenna_positions


def test_restrict_to_polarization_unknown_label_raises():
    obs = build_obs(PolarizationType.RR_RL_LR_LL)
    with pytest.raises(ValueError):
        restrict_to_polarization(obs, "XX")


@pmp("label", ("RR", "RL", "LR", "LL"))
def test_restrict_to_polarization_label_is_hardcoded_to_xx(label):
    # FIXME in polarization.py: the label of the result is always Polarization([9])
    # ("XX"), regardless of which product was selected. Pin the current behaviour so
    # the fix shows up as a failing test.
    obs = build_obs(PolarizationType.RR_RL_LR_LL)
    new = restrict_to_polarization(obs, label)

    assert new.legacy_polarization == Polarization([9])
    assert new.vis.domain[0].labels == ("XX",)


# ------------------------------------------------------------------- average_stokesi


def test_average_stokesi_is_noop_for_stokes_i_data():
    obs = build_obs(PolarizationType.I)
    assert average_stokesi(obs) is obs


@pmp("polarization_type", (PolarizationType.LL_RR, PolarizationType.XX_YY))
def test_average_stokesi_is_inverse_variance_weighted_mean(polarization_type):
    obs = build_obs(polarization_type)
    new = average_stokesi(obs)

    weight = obs.weight_val
    expected_weight = np.sum(weight, axis=0)[None]
    expected_vis = np.sum(weight * obs.vis_val, axis=0)[None] / expected_weight

    assert new.npol == 1
    assert new.vis.domain[0].labels == ("I",)
    assert new.legacy_polarization == Polarization.trivial()
    assert_allclose(new.vis_val, expected_vis)
    assert_allclose(new.weight_val, expected_weight)


def test_average_stokesi_of_equal_visibilities_reproduces_them():
    single = np.random.normal(size=(1, N_ROWS, len(FREQS))) + 1j * np.random.normal(
        size=(1, N_ROWS, len(FREQS))
    )
    vis = np.concatenate([single, single], axis=0)
    weight = np.random.uniform(1.0, 10.0, size=vis.shape)
    obs = build_obs_from_arrays(vis, weight, Polarization([8, 5]))

    new = average_stokesi(obs)

    assert_allclose(new.vis_val, single)
    assert_allclose(new.weight_val, np.sum(weight, axis=0)[None])


def test_average_stokesi_zeroes_fully_flagged_data_points():
    shape = (2, N_ROWS, len(FREQS))
    vis = np.random.normal(size=shape) + 1j * np.random.normal(size=shape)
    weight = np.ones(shape)
    # Flag one data point in both hands, one in a single hand only.
    weight[:, 0, 0] = 0.0
    weight[0, 1, 0] = 0.0
    obs = build_obs_from_arrays(vis, weight, Polarization([8, 5]))

    new = average_stokesi(obs)

    assert new.vis_val[0, 0, 0] == 0.0
    assert new.weight_val[0, 0, 0] == 0.0
    # Only the second hand survives at (1, 0), so its visibility is passed through.
    assert_allclose(new.vis_val[0, 1, 0], vis[1, 1, 0])
    assert new.weight_val[0, 1, 0] == 1.0
    assert np.all(np.isfinite(new.vis_val))


def test_average_stokesi_rejects_crosshanded_data():
    obs = build_obs(PolarizationType.RR_RL_LR_LL)
    with pytest.raises(AssertionError):
        average_stokesi(obs)


def test_average_stokesi_rejects_unordered_parallel_hands():
    # restrict_to_stokes_i() yields (LL, RR); (RR, LL) does not compare equal.
    obs = build_obs(PolarizationType.RR_LL)
    with pytest.raises(AssertionError):
        average_stokesi(obs)


def test_restrict_then_average_gives_stokes_i():
    obs = build_obs(PolarizationType.XX_XY_YX_YY)
    new = average_stokesi(restrict_to_stokesi(obs))

    assert new.npol == 1
    assert new.vis.domain[0].labels == ("I",)
