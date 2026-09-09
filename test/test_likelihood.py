import numpy as np
import pytest

from jubik.likelihood import build_gaussian_likelihood


def test_gaussian_likelihood_accepts_matching_shapes():
    data = np.ones((3, 4))
    std = 0.5 * np.ones((3, 4))
    likelihood = build_gaussian_likelihood(data, std)
    assert likelihood.domain.shape == data.shape


def test_gaussian_likelihood_accepts_float_std():
    data = np.ones((3, 4))
    likelihood = build_gaussian_likelihood(data, 0.5)
    assert likelihood.domain.shape == data.shape


@pytest.mark.parametrize(
    "std_shape",
    [
        (2,),  # different first axis
        (3, 1),  # broadcast-compatible, still a mismatch
        (1, 4),  # broadcast-compatible, same second axis
        (3, 4, 1),  # extra trailing axis
    ],
)
def test_gaussian_likelihood_rejects_shape_mismatch(std_shape):
    data = np.ones((3, 4))
    std = np.ones(std_shape)
    with pytest.raises(AssertionError, match="Shape mismatch"):
        build_gaussian_likelihood(data, std)
