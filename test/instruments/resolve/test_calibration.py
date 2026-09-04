import jax.numpy as jnp
import nifty.re as jft
import pytest
from numpy.testing import assert_allclose

from jubik.instruments.resolve.calibration.distributor import CalibrationInterpolator


def _identity_model(shape):
    domain = {"calibration": jft.ShapeWithDtype(shape, jnp.float64)}
    return jft.Model(lambda x: x["calibration"], domain=domain)


def test_calibration_interpolator_normalizes_absolute_times():
    values = jnp.arange(3.0).reshape(1, 1, 3, 1)
    interpolator = CalibrationInterpolator(
        model=_identity_model(values.shape),
        time_col=jnp.array([5.0e9, 5.0e9 + 1.0, 5.0e9 + 2.0]),
        dt=1.0,
        n_corr=1,
        n_ant=1,
        n_freq=1,
    )

    assert_allclose(interpolator({"calibration": values}), values)


def test_calibration_interpolator_normalizes_absolute_frequencies():
    values = jnp.arange(6.0).reshape(1, 1, 2, 3)
    interpolator = CalibrationInterpolator(
        model=_identity_model(values.shape),
        time_col=jnp.array([5.0e9, 5.0e9 + 2.0]),
        dt=2.0,
        n_corr=1,
        n_ant=1,
        n_freq=3,
        freq_col=jnp.array([1.0e11, 1.0e11 + 1.0e6, 1.0e11 + 2.0e6]),
        df=1.0e6,
    )

    assert_allclose(interpolator({"calibration": values}), values)


@pytest.mark.parametrize("dt", (0.0, -1.0))
def test_calibration_interpolator_rejects_invalid_time_spacing(dt):
    with pytest.raises(ValueError, match="dt.*positive"):
        CalibrationInterpolator(
            model=_identity_model((1, 1, 1, 1)),
            time_col=jnp.array([0.0]),
            dt=dt,
            n_corr=1,
            n_ant=1,
            n_freq=1,
        )
