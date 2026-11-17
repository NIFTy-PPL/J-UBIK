"""NIFTy-free polarization metadata, forward conversions, and adjoints."""

import jax
from jax import Array
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

from jubik.polarization import Polarization, PolarizationType
from jubik.instruments.resolve.polarization_conversion import (
    build_polarization_conversion, polarization_matrix,
)


@pytest.mark.parametrize("metadata", [
    ("RR", "LL"), ["RR", "LL"], PolarizationType.RR_LL,
    Polarization([5, 8]),
])
def test_conversion_accepts_metadata(metadata):
    assert_allclose(polarization_matrix(metadata, "I"), [[.5, .5]])


def test_conversion_accepts_legacy_trivial_polarization():
    assert_allclose(polarization_matrix(Polarization.trivial(), "I"), [[1]])


@pytest.mark.parametrize("outputs", [
    ("LL", "RL", "RR", "LR"), ("YY", "XY", "XX", "YX"),
])
def test_model_products_and_hermitian_adjoint(outputs):
    source = PolarizationType.IQUV
    target = outputs
    convert = build_polarization_conversion(source, target, assume_missing_stokes_zero=True)
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
    stokes = "I"
    correlations = ("RR", "LL")
    values = jnp.stack((jnp.full((2, 3), 2.), jnp.full((2, 3), 6.))).astype(complex)
    exact = build_polarization_conversion(correlations, stokes)
    model = build_polarization_conversion(stokes, correlations, assume_missing_stokes_zero=True)
    assert_allclose(exact(values), 4)
    transpose = jax.linear_transpose(model, jnp.zeros((1, 2, 3), dtype=complex))
    assert_allclose(jnp.conj(transpose(jnp.conj(values))[0]), 8)
    with pytest.raises(ValueError, match="cannot determine"):
        build_polarization_conversion(stokes, correlations)


def test_omitted_stokes_and_nonleading_axis():
    source = ("I", "Q", "U")
    target = PolarizationType.RR_RL_LR_LL
    convert = build_polarization_conversion(source, target, assume_missing_stokes_zero=True, axis=1)
    values = jnp.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
    i, q, u = values.T
    expected = np.stack((i, q+1j*u, q-1j*u, i), axis=1)
    assert_allclose(jax.jit(convert)(values), expected)


def test_shared_matrix_and_builder_accept_labels():
    source = ("LL", "RR")
    target = ("V", "I")
    matrix = polarization_matrix(source, target)
    assert isinstance(matrix, Array)
    assert_allclose(matrix, [[-.5, .5], [.5, .5]])
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


def test_zero_filled_model_rejects_invisible_sky():
    with pytest.raises(ValueError, match="conversion matrix is all zero"):
        polarization_matrix(
            ("Q", "U"), ("LL", "RR"), assume_missing_stokes_zero=True
        )


def test_zero_filled_model_allows_unused_stokes_components():
    assert_allclose(
        polarization_matrix(
            ("I", "Q", "U", "V"), ("RR", "LL"),
            assume_missing_stokes_zero=True,
        ),
        [[1, 0, 0, 1], [1, 0, 0, -1]],
    )


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
