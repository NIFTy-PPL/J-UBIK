"""Tests for the small data_modify helpers: precision, autocorrelations and
the conversion to the classic `resolve` Observation."""

import importlib
import importlib.util
import sys

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import jubik as ju
from jubik.instruments.resolve.data.observation import Observation
from jubik.instruments.resolve.data.data_modify.autocorrelations import (
    remove_autocorrelations,
    restrict_to_autocorrelations,
)
from jubik.instruments.resolve.data.data_modify.precision import (
    to_double_precision,
    to_single_precision,
)
from jubik.polarization import Polarization

from generate_test_obs import generate_random_obs

pmp = pytest.mark.parametrize

FREQS = np.array([1.0e9, 1.1e9, 1.2e9])
POL = ju.polarization.PolarizationType(("XX", "YY"))

# Rows 0, 3, 6, 9 are autocorrelations (ant1 == ant2).
ANT1 = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3], dtype=np.int64)
ANT2 = np.array([0, 1, 2, 1, 2, 3, 2, 3, 0, 3], dtype=np.int64)
AUTO_ROWS = np.array([0, 3, 6, 9])
CROSS_ROWS = np.array([1, 2, 4, 5, 7, 8])


def build_obs(n_rows=10, calib_info=True):
    if calib_info:
        ant1, ant2 = ANT1[:n_rows], ANT2[:n_rows]
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


def with_weight(obs, weight):
    """New observation with the same visibilities but the given weights."""
    return Observation(
        obs._antpos,
        obs.vis.asnumpy(),
        weight,
        obs.legacy_polarization,
        obs.freq,
        None,
    )


# ---------------------------------------------------------------- precision


def test_to_single_precision_dtypes_and_values():
    np.random.seed(90)
    obs = build_obs()

    single = to_single_precision(obs)

    assert single._vis.dtype == np.complex64
    assert single._weight.dtype == np.float32
    assert not single.is_double_precision()
    assert_allclose(single.vis.asnumpy(), obs.vis.asnumpy(), rtol=1e-6)
    assert_allclose(single.weight.asnumpy(), obs.weight.asnumpy(), rtol=1e-6)


def test_single_double_round_trip():
    np.random.seed(91)
    obs = build_obs()

    back = to_double_precision(to_single_precision(obs))

    assert back._vis.dtype == np.complex128
    assert back._weight.dtype == np.float64
    assert back.is_double_precision()
    assert_allclose(back.vis.asnumpy(), obs.vis.asnumpy(), rtol=1e-6)
    assert_allclose(back.weight.asnumpy(), obs.weight.asnumpy(), rtol=1e-6)
    assert_array_equal(back.antenna_positions.uvw, obs.antenna_positions.uvw)


@pmp("convert", (to_double_precision, to_single_precision))
def test_conversion_to_own_dtype_does_not_copy(convert):
    np.random.seed(92)
    obs = convert(build_obs())

    same = convert(obs)

    assert same._vis is obs._vis
    assert same._weight is obs._weight


@pmp("convert", (to_double_precision, to_single_precision))
def test_conversion_preserves_metadata(convert):
    np.random.seed(93)
    obs = build_obs()

    new = convert(obs)

    assert new._antpos is obs._antpos
    assert new.legacy_polarization == obs.legacy_polarization
    assert_array_equal(new.freq, obs.freq)
    assert new.nrow == obs.nrow


def test_to_single_precision_rejects_underflowing_weights():
    np.random.seed(94)
    obs = with_weight(build_obs(), np.full((2, 10, 3), 1e-300))
    assert obs.n_data_effective() == 60

    # float32 rounds these weights to exactly 0, which is the flagging
    # convention: all data would be silently flagged.
    with pytest.raises(ValueError):
        to_single_precision(obs)


def test_to_single_precision_rejects_overflowing_weights():
    np.random.seed(95)
    obs = with_weight(build_obs(), np.full((2, 10, 3), 1e39))

    with pytest.raises(ValueError):
        to_single_precision(obs)


def test_to_single_precision_keeps_flagged_rows_flagged():
    np.random.seed(96)
    weight = np.random.uniform(1.0, 10.0, size=(2, 10, 3))
    weight[0, 0, 0] = 0.0
    obs = with_weight(build_obs(), weight)

    single = to_single_precision(obs)

    assert_array_equal(single.flags_val, obs.flags_val)
    assert single.n_data_effective() == obs.n_data_effective()


# --------------------------------------------------------- autocorrelations


def test_restrict_to_autocorrelations():
    np.random.seed(97)
    obs = build_obs()

    auto = restrict_to_autocorrelations(obs)

    assert auto.nrow == len(AUTO_ROWS)
    assert_array_equal(auto.antenna_positions.ant1, auto.antenna_positions.ant2)
    assert_array_equal(auto.antenna_positions.ant1, ANT1[AUTO_ROWS])
    assert_array_equal(auto.antenna_positions.time, obs.antenna_positions.time[AUTO_ROWS])
    assert_array_equal(auto.vis.asnumpy(), obs.vis.asnumpy()[:, AUTO_ROWS, :])


def test_remove_autocorrelations():
    np.random.seed(98)
    obs = build_obs()

    cross = remove_autocorrelations(obs)

    assert cross.nrow == len(CROSS_ROWS)
    assert np.all(cross.antenna_positions.ant1 != cross.antenna_positions.ant2)
    assert_array_equal(cross.antenna_positions.ant1, ANT1[CROSS_ROWS])
    assert_array_equal(cross.vis.asnumpy(), obs.vis.asnumpy()[:, CROSS_ROWS, :])


def test_autocorrelations_partition_the_observation():
    np.random.seed(99)
    obs = build_obs()

    auto = restrict_to_autocorrelations(obs)
    cross = remove_autocorrelations(obs)

    assert auto.nrow + cross.nrow == obs.nrow
    uvw = np.concatenate(
        [auto.antenna_positions.uvw, cross.antenna_positions.uvw], axis=0
    )
    assert_array_equal(
        np.sort(uvw, axis=0), np.sort(obs.antenna_positions.uvw, axis=0)
    )


@pmp("func", (restrict_to_autocorrelations, remove_autocorrelations))
def test_autocorrelations_require_antenna_information(func):
    np.random.seed(100)
    obs = build_obs(calib_info=False)
    assert obs.antenna_positions.only_imaging

    with pytest.raises(ValueError):
        func(obs)


# ------------------------------------------------------ classic observation

CLASSIC_MODULE = "jubik.instruments.resolve.data.data_modify.classic_observation"


def test_data_modify_imports_without_classic_resolve():
    """The `try/except ImportError` guard in `data_modify/__init__.py` must
    keep the package importable when the classic `resolve` is missing."""
    dm = importlib.import_module("jubik.instruments.resolve.data.data_modify")
    resolve_available = importlib.util.find_spec("resolve") is not None

    assert hasattr(dm, "modify_observation")
    assert hasattr(dm, "convert_to_classic_observation") == resolve_available


class _Recorder:
    """Minimal stand-in for the classic `resolve` objects."""

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    @classmethod
    def from_list(cls, lst):
        return cls(lst)


@pytest.fixture
def classic_module():
    """Import `classic_observation` against a stub `resolve` package.

    The classic `resolve` is not necessarily installed; the stub lets us test
    the conversion logic itself (which arguments are handed over) without it.
    """
    stub = type(sys)("resolve")
    stub.AuxiliaryTable = type("AuxiliaryTable", (_Recorder,), {})
    stub.AntennaPositions = type("AntennaPositions", (_Recorder,), {})
    stub.Polarization = type("Polarization", (_Recorder,), {})
    stub.Observation = type("Observation", (_Recorder,), {})

    saved = {key: sys.modules.get(key) for key in ("resolve", CLASSIC_MODULE)}
    sys.modules["resolve"] = stub
    sys.modules.pop(CLASSIC_MODULE, None)
    try:
        yield importlib.import_module(CLASSIC_MODULE)
    finally:
        for key, value in saved.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value


def test_convert_without_auxiliary_tables(classic_module):
    np.random.seed(101)
    obs = build_obs()
    assert obs._auxiliary_tables is None

    classic = classic_module.convert_to_classic_observation(obs)

    assert classic.args[3].args[0] == obs.legacy_polarization.to_list()
    assert_array_equal(classic.args[1], obs.vis.asnumpy())
    assert_array_equal(classic.args[2], obs.weight.asnumpy())
    assert_array_equal(classic.args[4], obs.freq)
    assert classic.args[5] == {}


def test_convert_keeps_unusual_polarization_order(classic_module):
    np.random.seed(102)
    base = build_obs()
    # ("YY", "XX") is a legal CORR_TYPE ordering but no PolarizationType.
    obs = Observation(
        base._antpos,
        base.vis.asnumpy(),
        base.weight.asnumpy(),
        Polarization([12, 9]),
        base.freq,
        None,
    )

    classic = classic_module.convert_to_classic_observation(obs)

    assert classic.args[3].args[0] == [12, 9]


def test_convert_with_real_resolve():
    rve = pytest.importorskip("resolve")
    np.random.seed(103)
    obs = build_obs()
    convert = importlib.import_module(CLASSIC_MODULE).convert_to_classic_observation

    classic = convert(obs)

    assert isinstance(classic, rve.Observation)
    assert classic.nrow == obs.nrow
    assert_array_equal(classic.vis.val, obs.vis.asnumpy())
