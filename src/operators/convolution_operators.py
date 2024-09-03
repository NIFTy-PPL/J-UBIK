import numpy as np
import nifty8 as ift

from .zero_padder import MarginZeroPadder
from ..library.utils import convolve_field_operator


def _bilinear_weights(domain):
    """
    weights for the OverlapAdd Interpolation
    """
    psize = domain.shape[0]
    if psize/2 != int(psize/2):
        raise ValueError("this should happen")
    a = np.linspace(0, 1, int(psize/2), dtype="float64")
    b = np.concatenate([a, np.flip(a)])
    c = np.outer(b, b)
    return ift.Field.from_raw(domain, c)


def _get_weights(domain):
    """
    distribution the weights to the patch-space. Part of vectorization
    """
    weights = _bilinear_weights(domain[1])
    explode = ift.ContractionOperator(domain, spaces=0).adjoint
    return explode(weights)