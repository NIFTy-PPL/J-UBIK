"""NIFTy-free polarization metadata, forward conversions, and adjoints."""

from dataclasses import FrozenInstanceError

import jax
from jax import Array
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

from jubik.polarization import Polarization, PolarizationType
from jubik.instruments.resolve.polarization_space import (
    PolarizationSpace, build_polarization_conversion, polarization_matrix,
)


@pytest.mark.parametrize("metadata", [
    ("RR", "LL"), ["RR", "LL"], PolarizationType.RR_LL,
    Polarization([5, 8]), PolarizationSpace(("RR", "LL")),
])
def test_metadata_adapters(metadata):
    space = PolarizationSpace.from_polarization(metadata)
    assert space.labels == ("RR", "LL")
    assert space.shape == (2,)
    assert space.size == len(space) == 2
    assert space.axis_index("LL") == 1
    assert space.labels_eq(("LL", "RR"))
    assert space.to_polarization_type() is PolarizationType.RR_LL
    assert space.to_polarization().to_list() == [5, 8]


def test_immutable_ordered_metadata():
    labels = ["RR", "LL"]
    space = PolarizationSpace(labels)
    labels.reverse()
    assert space == PolarizationSpace(("RR", "LL"))
    assert space != PolarizationSpace(("LL", "RR"))
    assert len({space, PolarizationSpace(("RR", "LL"))}) == 1
    with pytest.raises(FrozenInstanceError):
        space.labels = ("I",)


@pytest.mark.parametrize("labels", [(), ("I", "I"), ("unknown",)])
def test_invalid_metadata(labels):
    with pytest.raises(ValueError):
        PolarizationSpace(labels)


def test_legacy_and_named_metadata_adapters():
    stokes = PolarizationSpace.from_polarization(Polarization.trivial())
    assert stokes == PolarizationSpace("I") == PolarizationSpace.from_polarization(PolarizationType.I)
    assert stokes.to_polarization() == Polarization.trivial()
    assert PolarizationSpace(("I", "V")).to_polarization().to_list() == [1, 4]
    # Containers support combinations not yet named by PolarizationType.
    with pytest.raises(ValueError):
        PolarizationSpace(("I", "V")).to_polarization_type()


@pytest.mark.parametrize("outputs", [
    ("LL", "RL", "RR", "LR"), ("YY", "XY", "XX", "YX"),
])
def test_model_products_and_hermitian_adjoint(outputs):
    source = PolarizationSpace(PolarizationType.IQUV)
    target = PolarizationSpace(outputs)
    convert = source.conversion_to_target(target, is_callable=True, assume_missing_stokes_zero=True)
    rng = np.random.default_rng(12)
    values = jnp.asarray(rng.normal(size=(4, 2, 3)) + 1j*rng.normal(size=(4, 2, 3)))
    i, q, u, v = values
    products = dict(RR=i+v, LL=i-v, RL=q+1j*u, LR=q-1j*u,
                    XX=i+q, YY=i-q, XY=u+1j*v, YX=u-1j*v)
    actual = jax.jit(convert)(values)
    assert_allclose(actual, np.stack([products[label] for label in outputs]))
    cotangent = jnp.asarray(rng.normal(size=actual.shape) + 1j*rng.normal(size=actual.shape))
    # JAX linear_transpose is bilinear for complex arrays. Conjugate both
    # sides to obtain the Hermitian adjoint of the actual conversion callable.
    transpose = jax.linear_transpose(convert, jnp.zeros_like(values))
    adjoint = jnp.conj(transpose(jnp.conj(cotangent))[0])
    assert_allclose(np.vdot(actual, cotangent), np.vdot(values, adjoint), rtol=1e-6)


def test_exact_conversion_and_model_adjoint_have_distinct_normalizations():
    stokes = PolarizationSpace("I")
    correlations = PolarizationSpace(("RR", "LL"))
    values = jnp.stack((jnp.full((2, 3), 2.), jnp.full((2, 3), 6.))).astype(complex)
    exact = correlations.conversion_to_target(stokes, is_callable=True)
    model = stokes.conversion_to_target(correlations, is_callable=True, assume_missing_stokes_zero=True)
    assert_allclose(exact(values), 4)
    transpose = jax.linear_transpose(model, jnp.zeros((1, 2, 3), dtype=complex))
    assert_allclose(jnp.conj(transpose(jnp.conj(values))[0]), 8)
    with pytest.raises(ValueError, match="cannot determine"):
        stokes.conversion_to_target(correlations, is_callable=True)


def test_omitted_stokes_and_nonleading_axis():
    source = PolarizationSpace(("I", "Q", "U"))
    target = PolarizationSpace(PolarizationType.RR_RL_LR_LL)
    convert = source.conversion_to_target(target, is_callable=True, assume_missing_stokes_zero=True, axis=1)
    values = jnp.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
    i, q, u = values.T
    expected = np.stack((i, q+1j*u, q-1j*u, i), axis=1)
    assert_allclose(jax.jit(convert)(values), expected)


def test_shared_matrix_and_builder_accept_spaces():
    source = PolarizationSpace(("LL", "RR"))
    target = PolarizationSpace(("V", "I"))
    matrix = source.conversion_to_target(target)
    assert isinstance(matrix, Array)
    assert_allclose(matrix, [[-.5, .5], [.5, .5]])
    assert_allclose(polarization_matrix(source, target), matrix)
    assert_allclose(build_polarization_conversion(source, target)(jnp.array([6., 10.])), [2., 8.])


@pytest.mark.parametrize("inputs,outputs,expected", [
    (("LL", "RR"), ("V", "I"), [[-.5, .5], [.5, .5]]),
    (("LR", "RL"), ("U", "Q"), [[.5j, -.5j], [.5, .5]]),
    (("YX", "XY"), ("V", "U"), [[.5j, -.5j], [.5, .5]]),
    (("YY", "XX"), ("Q", "I"), [[-.5, .5], [.5, .5]]),
    (("Q", "I"), ("YY", "XX"), [[-1, 1], [1, 1]]),
])
def test_polarization_matrix_ordered_subsets(inputs, outputs, expected):
    assert_allclose(polarization_matrix(inputs, outputs), expected, atol=1e-14)


@pytest.mark.parametrize("inputs,outputs", [
    (("RL",), ("I",)), (("RR",), ("I",)),
    (("I",), ("RR",)), (("RR", "LL"), ("Q",)),
])
def test_polarization_matrix_rejects_underdetermined_conversion(inputs, outputs):
    with pytest.raises(ValueError, match="cannot determine"):
        polarization_matrix(inputs, outputs)


def test_stokes_forward_model_expansion_is_explicit():
    assert_allclose(polarization_matrix(("I",), ("RR", "RL", "LR", "LL"), assume_missing_stokes_zero=True),
                    [[1], [0], [0], [1]])
    with pytest.raises(ValueError, match="must be Stokes"):
        polarization_matrix(("RR",), ("I",), assume_missing_stokes_zero=True)


@pytest.mark.parametrize("labels", [("I", "I"), (), ("unknown",)])
def test_polarization_matrix_validates_labels(labels):
    with pytest.raises(ValueError):
        polarization_matrix(labels, ("I",))


def test_polarization_matrix_rejects_mixed_input_bases():
    with pytest.raises(ValueError, match="one polarization basis"):
        polarization_matrix(("I", "RR"), ("V",))
