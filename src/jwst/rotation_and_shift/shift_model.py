import nifty8.re as jft

from ..parametric_model import build_parametric_prior
from typing import Tuple


def build_shift_model(domain_key: str, mean_sigma: Tuple[float]):
    '''The shift model is a Gaussian distribution over the positions (x, y).
    (x, y) <- Gaussian(mean, sigma)

    Parameters
    ----------
    domain_key: str

    mean_sigma: Tuple[float]
        The mean and sigma for the Gaussian distribution of shift model.
    '''
    shape = (2,)
    prior_parameters = ('normal', *mean_sigma)

    shift_model = build_parametric_prior(
        domain_key, prior_parameters, shape)
    domain = {domain_key: jft.ShapeWithDtype(shape)}
    return jft.Model(shift_model, domain=domain)
