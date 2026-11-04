import nifty.re as jft


class InverseStandardDeviation(jft.Model):
    """The inverse noise std model and its correction factors.

    `correction_model` holds the per baseline factors, `antenna_correction_model`
    the per antenna factors; either is None when not inferred.
    """

    def __init__(
        self,
        inverse_std_model: jft.Model,
        correction_model: jft.Model | None,
        antenna_correction_model: jft.Model | None = None,
    ):
        self._log_inverse_covariance_model = inverse_std_model
        self.correction_model = correction_model
        self.antenna_correction_model = antenna_correction_model

        super().__init__(domain=inverse_std_model.domain)

    def __call__(self, x):
        return self._log_inverse_covariance_model(x)
