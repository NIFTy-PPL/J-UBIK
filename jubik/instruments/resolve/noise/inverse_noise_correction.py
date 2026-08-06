import nifty.re as jft


class InverseStandardDeviation(jft.Model):
    def __init__(
        self,
        inverse_std_model: jft.Model,
        correction_model: jft.Model | None,
    ):
        self._log_inverse_covariance_model = inverse_std_model
        self.correction_model = correction_model

        super().__init__(domain=inverse_std_model.domain)

    def __call__(self, x):
        return self._log_inverse_covariance_model(x)
