import nifty8.re as jft

from jax.numpy import array
from ..parametric_model import build_parametric_prior
from typing import Tuple


class ShiftModel(jft.Model):
    def __init__(self, prior_model: jft.Model, to_pixel_frac: Tuple[float]):
        self.to_pixel_frac = to_pixel_frac
        self.prior_model = prior_model
        self._key = next(iter(prior_model.domain.keys()))

        super().__init__(domain=prior_model.domain)

    def __call__(self, x):
        arr = self.prior_model(x) / self.to_pixel_frac
        arr = arr[::-1, :, :]
        return {self._key: arr}


def build_shift_model(
    domain_key: str,
    mean_sigma: Tuple[float],
    subsample_distances: Tuple[float]
) -> ShiftModel:
    '''The shift model is a Gaussian distribution over the positions (x, y).
    (x, y) <- Gaussian(mean, sigma)

    Parameters
    ----------
    domain_key: str

    mean_sigma: Tuple[float]
        The mean and sigma for the Gaussian distribution of shift model.
    '''
    shape = (2, 1, 1)
    prior_parameters = ('normal', *mean_sigma)

    to_pixel_frac = array(subsample_distances).reshape(shape)

    domain = {domain_key: jft.ShapeWithDtype(shape)}
    shift_prior = build_parametric_prior(domain_key, prior_parameters, shape)
    return ShiftModel(jft.Model(shift_prior, domain=domain), to_pixel_frac)