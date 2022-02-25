import numpy as np

import nifty8 as ift

def get_patch_weights(domain):
    psize = domain.shape[0]
    a = np.linspace(0, 1, int(psize/2))
    b = np.concatenate([a, np.flip(a)])
    c = np.outer(b,b)
    res = ift.Field.from_raw(domain, c)
    return res

def get_weights(domain):
    weights = get_patch_weights(domain[1])
    explode = ift.ContractionOperator(domain, spaces=0)
    res = explode.adjoint(weights)
    return res
