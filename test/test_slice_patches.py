import jax
import numpy as np
import pytest
from jax import numpy as jnp

import jubik0 as ju


@pytest.fixture
def shape():
    return (36, 36)


@pytest.fixture
def n_patches():
    return 16


@pytest.fixture
def overlap():
    return 2


@pytest.fixture
def x(shape):
    return np.ones(shape)


@pytest.fixture
def sliced_patches(x, shape, n_patches, overlap):
    return ju.slice_patches(x, shape, n_patches, overlap)


@pytest.fixture
def f(shape, n_patches, overlap):
    def slice_patcher(field):
        return ju.slice_patches(field, shape, n_patches, overlap)

    return slice_patcher


@pytest.fixture
def fadj(f, x):
    return jax.linear_transpose(f, x)


def test_slice_patches(sliced_patches):
    assert sliced_patches is not None
    assert isinstance(sliced_patches, jnp.ndarray)


def test_adjointness(f, fadj, shape):
    a = np.random.rand(shape[0], shape[1])
    res = f(a)
    b = np.random.rand(res.shape[0], res.shape[1], res.shape[2])

    # forward
    res1 = np.vdot(b, f(a))
    # backward
    res2 = np.vdot(fadj(b), a)

    np.testing.assert_allclose(res1, res2)
