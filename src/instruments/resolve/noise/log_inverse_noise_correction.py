import nifty.re as jft


class LogInverseNoiseCovariance(jft.Model):
    def __init__(
        self,
        log_inverse_covariance_model: jft.Model,
        correction_model: jft.Model | None,
    ):
        self._log_inverse_covariance_model = log_inverse_covariance_model
        self.correction_model = correction_model

        super().__init__(domain=log_inverse_covariance_model.domain)

    def __call__(self, x):
        return self._log_inverse_covariance_model(x)
