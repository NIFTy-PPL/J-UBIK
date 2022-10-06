import numpy as np

import nifty8 as ift

def _bilinear_weights(domain):
    psize = domain.shape[0]
    if psize/2 != int(psize/2):
        raise ValueError("this should happen")
    a = np.linspace(0, 1, int(psize/2), dtype="float64")
    b = np.concatenate([a, np.flip(a)])
    c = np.outer(b,b)
    return ift.Field.from_raw(domain, c)

def get_weights(domain):
    weights = _bilinear_weights(domain[1])
    explode = ift.ContractionOperator(domain, spaces=0).adjoint
    return explode(weights)
