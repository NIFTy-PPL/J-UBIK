"""Observation arrays and masking without classic NIFTy structures."""

import numpy as np
import pytest
from astropy.constants import c
from numpy.testing import assert_allclose, assert_array_equal

from jubik.instruments.resolve.data import AntennaPositions, Observation
from jubik.polarization import Polarization


@pytest.fixture
def observation():
    vis = np.array([[[3 + 4j, complex(np.nan)], [2j, 1j]]])
    weight = np.array([[[4.0, 0.0], [9.0, 1.0]]])
    # Observation where the nan visibility entry is flagged
    return Observation(
        AntennaPositions(np.array([[3.0, 4.0, 0.0], [0.0, 0.0, 10.0]])),
        vis,
        weight,
        Polarization.trivial(),
        np.array([1.0e9, 2.0e9]),
        None,
    )


def test_arrays_and_masking(observation):
    obs = observation
    assert isinstance(obs.vis, np.ndarray)
    assert isinstance(obs.weight, np.ndarray)
    assert not obs.vis.flags.writeable
    assert not obs.weight.flags.writeable
    assert_array_equal(obs.flags, ~obs.mask)
    assert_array_equal(obs.apply_flags(obs.vis), [3 + 4j, 2j, 1j])
    assert_array_equal(obs.mask_operator(obs.vis), obs.apply_flags(obs.vis))
    assert obs.n_data_effective() == 3
    assert obs.fraction_useful() == 0.75
    assert obs.max_snr() == 10.0
    assert obs.polarization_labels == ("I",)
    with pytest.raises(ValueError, match="shape"):
        obs.apply_flags(np.zeros(4))


def test_baseline_lengths_and_flagged_nan(observation):
    obs = observation
    expected = np.array([5.0, 10.0])[:, None] * obs.freq[None, :] / c.to_value("m/s")
    assert_allclose(obs.effective_uvwlen()[0], expected)
    flagged = obs.flags_to_nan()
    assert np.isnan(flagged.vis[obs.flags]).all()
    assert_array_equal(flagged.vis[obs.mask], obs.vis[obs.mask])


@pytest.mark.parametrize(
    "alias, replacement",
    [
        ("vis_val", "vis"),
        ("weight_val", "weight"),
        ("flags_val", "flags"),
        ("mask_val", "mask"),
    ],
)
def test_deprecated_array_aliases(observation, alias, replacement):
    with pytest.warns(
        DeprecationWarning, match=f"{alias} is deprecated; use {replacement}"
    ) as records:
        value = getattr(observation, alias)
    assert records[0].filename == __file__
    assert_array_equal(value, getattr(observation, replacement))
    if alias in ("vis_val", "weight_val"):
        assert value is getattr(observation, replacement)


def test_mask_operator_adjoint_and_jax(observation):
    import jax
    import jax.numpy as jnp

    obs = observation
    op = obs.mask_operator
    x = np.arange(obs.vis.size, dtype=float).reshape(obs.vis.shape) + 1j
    y = np.array([2 + 3j, 4 - 1j, 5j])
    expected = np.zeros_like(x)
    expected[obs.mask] = y
    transpose = jax.linear_transpose(op, jnp.zeros(x.shape, dtype=jnp.complex64))
    restored = transpose(jnp.asarray(y, dtype=jnp.complex64))[0]
    assert_allclose(restored, expected)
    assert_allclose(np.vdot(op(x), y), np.vdot(x, restored), rtol=1e-6)
    assert_allclose(jax.jit(op)(jnp.asarray(x)), op(x), rtol=1e-6)
    assert_allclose(
        jax.jit(transpose)(jnp.asarray(y, dtype=jnp.complex64))[0], expected
    )
