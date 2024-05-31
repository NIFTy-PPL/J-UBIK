import nifty8.re as jft

from jax.numpy import array
from ..parametric_model import build_parametric_prior
from typing import Tuple


class ShiftModel(jft.Model):
    def __init__(self, prior_model: jft.Model, pix_distance: Tuple[float]):
        assert len(prior_model.target.keys()) == 1

        self.pix_distance = pix_distance
        self.prior_model = prior_model
        self._tkey = next(iter(prior_model.target.keys()))

        super().__init__(domain=prior_model.domain)

    def __call__(self, x):
        return {
            self._tkey: self.prior_model(x)[self._tkey] / self.pix_distance
        }


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
    pix_distance = array(subsample_distances).reshape(shape)

    # Build Prior
    shift_prior = build_parametric_prior(
        domain_key, ('normal', *mean_sigma), shape)
    shift_prior = jft.wrap_left(shift_prior, domain_key + '_t')
    shift_prior_model = jft.Model(
        shift_prior, domain={domain_key: jft.ShapeWithDtype(shape)})

    return ShiftModel(shift_prior_model, pix_distance)
