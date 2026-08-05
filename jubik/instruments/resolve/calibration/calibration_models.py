import nifty.re as jft
import jax.numpy as jnp

from .distributor import CalibrationDistributor


def call_fixed_covariance(model_vis, cop, mask):
    data_model = model_vis * cop

    return data_model[mask]


def call_variable_covariance(model_vis, cop, mask, log_inv_cov):
    data_model = model_vis * cop
    inv_std = jnp.exp(0.5 * log_inv_cov)

    return (data_model[mask], inv_std[mask])


class DirectionIndependentCalibrationModel(jft.Model):
    """
    Forward model for direction-independent calibration.

    The model applies the complex antenna-based calibration operator to a set of
    model visibilities and returns the predicted visibilities at the unflagged
    observation points.

    Optionally, a model for the logarithm of the inverse noise variance can be
    supplied. In this case, the corresponding inverse noise standard deviations
    are predicted and returned alongside the calibrated visibilities.

    Parameters
    ----------
    cop : CalibrationDistributor
        Calibration operator producing complex baseline gains for each
        visibility.
    model_visibilities : jnp.ndarray
        Predicted visibilities of the sky model before calibration. Its shape
        must be compatible with the output of `cop`.
    mask : jnp.ndarray
        Boolean mask selecting the unflagged visibilities. Entries evaluating
        to ``True`` are retained in the output.
    log_inverse_covariance_model : jft.Model, optional
        Model predicting the logarithm of the inverse noise variance
        (inverse covariance for diagonal noise). If omitted, only the
        calibrated visibilities are returned.

    Notes
    -----
    The calibrated visibility model is computed as

        V_pred = G * V_model,

    where `G` denotes the complex baseline calibration factors returned by
    `cop`.
    """

    def __init__(
        self,
        cop: CalibrationDistributor,
        model_visibilities: jnp.ndarray,
        mask: jnp.ndarray,
        log_inverse_covariance_model: jft.Model | None = None,
    ):

        if log_inverse_covariance_model is None:
            self._call = lambda primals: call_fixed_covariance(
                model_vis=model_visibilities,
                cop=cop(primals),
                mask=mask,
            )

            super().__init__(init=cop.init)
        else:
            self._call = lambda primals: call_variable_covariance(
                model_vis=model_visibilities,
                cop=cop(primals),
                mask=mask,
                log_inv_cov=log_inverse_covariance_model(primals),
            )

            super().__init__(init=self._cop.init | self._log_inv_cov.init)

    def __call__(self, primals):
        return self._call(primals)
