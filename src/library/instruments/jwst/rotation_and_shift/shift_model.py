import nifty8.re as jft

from jax.numpy import array
from ..parametric_model import build_parametric_prior
from typing import Tuple
from numpy.typing import ArrayLike


class ShiftModel(jft.Model):
    def __init__(self, prior_model: jft.Model, pix_distance: Tuple[float]):
        self.pix_distance = pix_distance
        self.prior_model = prior_model

        # FIXME: HACK, the target shape is actually prescribed by the coords.
        # However, one does not necesserily know the coordinates shape upon
        # shift model creation.
        super().__init__(domain=prior_model.domain, target=prior_model.target)

    def __call__(self, params: dict, coords: ArrayLike) -> ArrayLike:
        return coords + self.prior_model(params) / self.pix_distance


def build_shift_model(
    domain_key: str,
    mean_sigma: Tuple[float],
    pix_distances: Tuple[float]
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
    pix_distance = array(pix_distances).reshape(shape)

    # Build Prior
    shift_prior = build_parametric_prior(
        domain_key, ('normal', *mean_sigma), shape)
    shift_prior_model = jft.Model(
        shift_prior, domain={domain_key: jft.ShapeWithDtype(shape)})

    return ShiftModel(shift_prior_model, pix_distance)
