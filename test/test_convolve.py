import numpy as np
import pytest

import jubik as ju
from jubik.convolve import _bilinear_weights


def test_bilinear_weights_even_shape_properties():
    w = _bilinear_weights((4, 4))

    assert w.shape == (4, 4)
    assert np.isfinite(w).all()
    assert np.all(w >= 0)
    np.testing.assert_allclose(w, w[::-1, :])
    np.testing.assert_allclose(w, w[:, ::-1])


def test_bilinear_weights_odd_shape_raises():
    with pytest.raises(ValueError):
        _bilinear_weights((3, 3))


def test_integrate_constant_field_exact():
    x = np.ones((3, 4))
    domain = ju.Domain(shape=(3, 4), distances=(0.5, 0.25))

    res = ju.integrate(x, domain, axes=(0, 1))

    assert res == pytest.approx(3 * 4 * 0.5 * 0.25)


def test_convolve_identity_kernel_returns_input_for_unit_cell_volume():
    rng = np.random.default_rng(0)
    signal = rng.normal(size=(4, 4))
    kernel = np.zeros((4, 4))
    kernel[0, 0] = 1.0
    domain = ju.Domain(shape=(4, 4), distances=(1.0, 1.0))

    res = ju.convolve(kernel, signal, domain, axes=(-2, -1))

    np.testing.assert_allclose(np.asarray(res), signal, rtol=1e-10, atol=1e-10)


def test_convolve_broadcasting_branch_prints_and_preserves_channel_shape(capsys):
    kernel = np.zeros((4, 4))
    kernel[0, 0] = 1.0
    signal = np.stack([np.ones((4, 4)), 2.0 * np.ones((4, 4))])
    domain = ju.Domain(shape=(4, 4), distances=(1.0, 1.0))

    res = ju.convolve(kernel, signal, domain, axes=(-2, -1))
    captured = capsys.readouterr()

    assert "Dimension Inconsistency. Broadcasting PSFs" in captured.out
    assert np.asarray(res).shape == signal.shape
    np.testing.assert_allclose(np.asarray(res), signal, rtol=1e-10, atol=1e-10)
