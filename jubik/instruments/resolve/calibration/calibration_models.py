import nifty.re as jft
import jax.numpy as jnp

from .distributor import CalibrationDistributor


class ModelCalibrationLikelihoodFixedCovariance(jft.Model):
    """
    Provides a flagged data model for calibration

    Parameters
    ----------
    cop: CalibrationDistribution
        Calibration operator
    model_visibilities: jnp.ndarray
        Assumed visibilities of the point source.
    mask: jnp.array
        Mask as boolean numpy array for good visibilites
    """

    def __init__(
        self,
        cop: CalibrationDistributor,
        model_visibilities: jnp.ndarray,
        mask: jnp.ndarray,
    ):
        self._cop = cop
        self._vis = model_visibilities
        self._mask = mask

        super().__init__(init=self._cop.init)

    def __call__(self, x):
        data_model = self._vis * self._cop(x)
        flagged_data_model = data_model[self._mask]

        return flagged_data_model


class ModelCalibrationLikelihoodVariableCovariance(jft.Model):
    """
    Provides a combined flagged data model and flagged inverse covariance model for calibration

    Parameters
    ----------
    cop: CalibrationDistribution
        Calibration operator
    model_visibilities: jnp.ndarray
        Assumed visibilities of the point source.
    log_inverse_covariance_model: jft.Model
        Model for log inverse covariance
    mask: jnp.array
        Mask as boolean numpy array for good visibilites
    """

    def __init__(
        self,
        cop: CalibrationDistributor,
        model_visibilities: jnp.ndarray,
        log_inverse_covariance_model: jft.Model,
        mask: jnp.ndarray,
    ):
        self._cop = cop
        self._vis = model_visibilities
        self._mask = mask
        self._log_inv_cov = log_inverse_covariance_model

        super().__init__(init=self._cop.init | self._log_inv_cov.init)

    def __call__(self, x):
        data_model = self._vis * self._cop(x)
        flagged_data_model = data_model[self._mask]

        inv_std = jnp.exp(0.5 * self._log_inv_cov(x))
        flagged_inv_std = inv_std[self._mask]

        return (flagged_data_model, flagged_inv_std)
